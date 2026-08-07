#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "kl_r025_official_to_r020",
    "kl_base_full_to_r020",
    "kl_base_full_to_r025_official",
    "js_r020_r025_official",
    "kl_r025_nested_to_r020",
    "js_r020_r025_nested",
    "kl_base_full_to_r025_nested",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the complete 88-job ratio-pair sweep.")
    parser.add_argument(
        "--config-manifest",
        type=Path,
        default=ROOT / "configs" / "ratio_pairs" / "manifest.json",
    )
    parser.add_argument(
        "--job-manifest",
        type=Path,
        default=ROOT / "outputs" / "slurm_jobs_ratio_pairs_88_20260729.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ratio_pairs_mmstar_clean100_max1024" / "analysis",
    )
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sample_map(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in directory.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(payload["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate sample ID {sample_id} under {directory}")
        result[sample_id] = payload
    return result


def audit_slurm(job_manifest: Path) -> dict[str, Any]:
    with job_manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 88:
        raise ValueError(f"Expected 88 Slurm rows, found {len(rows)}")
    ids = ",".join(row["job_id"] for row in rows)
    output = subprocess.check_output(
        [
            "sacct",
            "-n",
            "-P",
            "-X",
            "-j",
            ids,
            "--format=JobIDRaw,State,ExitCode,Elapsed,AllocTRES",
        ],
        text=True,
    )
    accounting: dict[str, list[str]] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if fields and fields[0] in {row["job_id"] for row in rows}:
            accounting[fields[0]] = fields[1:]
    missing = [row["job_id"] for row in rows if row["job_id"] not in accounting]
    failed = {
        row["job_id"]: accounting.get(row["job_id"])
        for row in rows
        if accounting.get(row["job_id"], [None, None])[:2] != ["COMPLETED", "0:0"]
    }
    if missing or failed:
        raise RuntimeError(f"Slurm audit failed: missing={missing}, failed={failed}")
    elapsed_seconds = []
    for values in accounting.values():
        hours, minutes, seconds = (int(part) for part in values[2].split(":"))
        elapsed_seconds.append(hours * 3600 + minutes * 60 + seconds)
    return {
        "job_count": len(rows),
        "completed_exit_0_count": len(accounting),
        "first_job_id": rows[0]["job_id"],
        "last_job_id": rows[-1]["job_id"],
        "min_elapsed_seconds": min(elapsed_seconds),
        "max_elapsed_seconds": max(elapsed_seconds),
        "mean_elapsed_seconds": sum(elapsed_seconds) / len(elapsed_seconds),
    }


def main() -> int:
    args = parse_args()
    entries = json.loads(args.config_manifest.resolve().read_text(encoding="utf-8"))
    if len(entries) != 8:
        raise ValueError(f"Expected 8 method/pair configs, found {len(entries)}")
    expected_ids: set[str] | None = None
    cohort_hashes: set[str] = set()
    record_count = 0
    ratio_counts: Counter[str] = Counter()
    truncations: list[dict[str, Any]] = []
    format_incomplete = 0
    max_equivalence_error = 0.0
    max_peak_memory = 0.0
    issues: list[str] = []

    by_key = {(str(entry["pair"]), str(entry["method"])): entry for entry in entries}
    for entry in entries:
        cfg = json.loads(Path(entry["config"]).read_text(encoding="utf-8"))
        sample_path = Path(cfg["samples"])
        cohort_hashes.add(sha256_file(sample_path))
        ids = {
            str(json.loads(line)["sample_id"])
            for line in sample_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        expected_ids = ids if expected_ids is None else expected_ids
        if ids != expected_ids or len(ids) != 100:
            issues.append(f"Cohort mismatch for {entry['pair']} {entry['method']}")
        output_root = Path(entry["output_root"])
        for checkpoint in cfg["checkpoint_steps"]:
            label = str(checkpoint["label"])
            sample_map = load_sample_map(output_root / label / "samples")
            errors = list((output_root / label / "errors").glob("*.json"))
            if set(sample_map) != expected_ids:
                issues.append(
                    f"Sample IDs mismatch for {entry['pair']} {entry['method']} {label}"
                )
            if errors:
                issues.append(f"Error files for {entry['pair']} {entry['method']} {label}: {len(errors)}")
            for payload in sample_map.values():
                record_count += 1
                ratio_counts[str(entry["pair"])] += 1
                low = float(payload["ratio_pair"]["low_retention_ratio"])
                high = float(payload["ratio_pair"]["high_retention_ratio"])
                if low != float(entry["low_retention_ratio"]) or high != float(entry["high_retention_ratio"]):
                    issues.append(f"Ratio metadata mismatch in {entry['pair']} {entry['method']} {label}")
                token_count = int(payload["generated_token_count"])
                if token_count != len(payload["generated_token_ids"]):
                    issues.append(f"Token count mismatch for sample {payload['sample_id']}")
                if not payload["official_fixed_r020_equivalence"]["allclose"]:
                    issues.append(f"Fixed-context mismatch for sample {payload['sample_id']}")
                max_equivalence_error = max(
                    max_equivalence_error,
                    float(payload["official_fixed_r020_equivalence"]["max_abs_logit_error"]),
                )
                if not payload["masks"]["low_subset_high_nested"]:
                    issues.append(f"Nested-mask failure for sample {payload['sample_id']}")
                low_indices = payload["masks"]["low_retained_indices"]
                high_indices = payload["masks"]["high_nested_retained_indices"]
                if not set(low_indices).issubset(set(high_indices)):
                    issues.append(f"Nested identity failure for sample {payload['sample_id']}")
                if len(low_indices) != payload["visual_tokens"]["low_official"]:
                    issues.append(f"Low token-count failure for sample {payload['sample_id']}")
                if len(high_indices) != payload["visual_tokens"]["high_nested"]:
                    issues.append(f"High token-count failure for sample {payload['sample_id']}")
                for metric in METRICS:
                    values = payload["metrics"][metric]
                    if len(values) != token_count:
                        issues.append(f"Metric length failure {metric} sample {payload['sample_id']}")
                    if any(not math.isfinite(value) or value < 0.0 for value in values):
                        issues.append(f"Invalid metric {metric} sample {payload['sample_id']}")
                if payload["hit_max_new_tokens"]:
                    truncations.append(
                        {
                            "pair": entry["pair"],
                            "method": entry["method"],
                            "checkpoint_label": label,
                            "sample_id": str(payload["sample_id"]),
                            "generated_token_count": token_count,
                        }
                    )
                generated_text = str(payload["generated_text"])
                if not all(tag in generated_text for tag in ("<think>", "</think>", "<answer>", "</answer>")):
                    format_incomplete += 1
                max_peak_memory = max(max_peak_memory, float(payload["peak_gpu_allocated_gib"]))

    step0_exact: dict[str, bool] = {}
    for pair in sorted({str(entry["pair"]) for entry in entries}):
        opsd_entry = by_key[(pair, "opsd")]
        sft_entry = by_key[(pair, "sft")]
        opsd = load_sample_map(Path(opsd_entry["output_root"]) / "step_0" / "samples")
        sft = load_sample_map(Path(sft_entry["output_root"]) / "step_0" / "samples")
        exact = True
        for sample_id in sorted(expected_ids or set()):
            left, right = opsd[sample_id], sft[sample_id]
            exact &= left["generated_token_ids"] == right["generated_token_ids"]
            exact &= left["masks"]["low_retained_indices"] == right["masks"]["low_retained_indices"]
            exact &= all(left["metrics"][metric] == right["metrics"][metric] for metric in METRICS)
        step0_exact[pair] = bool(exact)
        if not exact:
            issues.append(f"SFT/OPSD step-0 mismatch for {pair}")

    repeated_low_exact: dict[str, bool] = {}
    for method in ("opsd", "sft"):
        first = by_key[("r015_r0175", method)]
        second = by_key[("r015_r020", method)]
        cfg = json.loads(Path(first["config"]).read_text(encoding="utf-8"))
        exact = True
        for checkpoint in cfg["checkpoint_steps"]:
            label = str(checkpoint["label"])
            left = load_sample_map(Path(first["output_root"]) / label / "samples")
            right = load_sample_map(Path(second["output_root"]) / label / "samples")
            for sample_id in sorted(expected_ids or set()):
                exact &= left[sample_id]["generated_token_ids"] == right[sample_id]["generated_token_ids"]
                exact &= left[sample_id]["masks"]["low_retained_indices"] == right[sample_id]["masks"]["low_retained_indices"]
                exact &= left[sample_id]["metrics"]["kl_base_full_to_r020"] == right[sample_id]["metrics"]["kl_base_full_to_r020"]
        repeated_low_exact[method] = bool(exact)
        if not exact:
            issues.append(f"Repeated 15% low-side mismatch for {method}")

    slurm = audit_slurm(args.job_manifest.resolve())
    if record_count != 8800:
        issues.append(f"Expected 8800 records, found {record_count}")
    if set(ratio_counts.values()) != {2200} or len(ratio_counts) != 4:
        issues.append(f"Unexpected per-pair counts: {dict(ratio_counts)}")
    if len(cohort_hashes) != 1:
        issues.append(f"Multiple cohort hashes: {sorted(cohort_hashes)}")
    payload = {
        "passed": not issues,
        "issues": issues,
        "record_count": record_count,
        "expected_record_count": 8800,
        "ratio_record_counts": dict(sorted(ratio_counts.items())),
        "cohort_sha256": next(iter(cohort_hashes)),
        "step0_sft_opsd_exact": step0_exact,
        "repeated_low_15pct_exact": repeated_low_exact,
        "truncation_count": len(truncations),
        "truncations": truncations,
        "format_incomplete_count": format_incomplete,
        "max_fixed_context_logit_error": max_equivalence_error,
        "max_peak_gpu_allocated_gib": max_peak_memory,
        "slurm": slurm,
    }
    output_dir = args.output_dir.resolve()
    atomic_write(output_dir / "quality_audit.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = [
        "# Ratio-Pair Sweep Quality Audit",
        "",
        f"- Passed: **{payload['passed']}**",
        f"- Records: {record_count}/8800",
        f"- Cohort SHA256: `{payload['cohort_sha256']}`",
        f"- Per-pair records: `{payload['ratio_record_counts']}`",
        f"- Slurm: {slurm['completed_exit_0_count']}/88 completed with exit code 0",
        f"- SFT/OPSD step-0 exact: `{step0_exact}`",
        f"- Repeated 15% low-side exact: `{repeated_low_exact}`",
        f"- Fixed-context maximum logit error: {max_equivalence_error}",
        f"- Truncations: {len(truncations)}",
        f"- Incomplete think/answer format records: {format_incomplete}",
        f"- Peak allocated GPU memory: {max_peak_memory:.2f} GiB",
        "",
        "## Issues",
        "",
        *(f"- {issue}" for issue in issues),
        *( ["- None."] if not issues else [] ),
        "",
    ]
    atomic_write(output_dir / "QUALITY_AUDIT.md", "\n".join(report))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
