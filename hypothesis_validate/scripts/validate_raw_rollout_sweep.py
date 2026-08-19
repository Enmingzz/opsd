#!/usr/bin/env python
"""Validate completeness and raw-only invariants for the 20-run sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_FIELDS = {
    "parsed_answer",
    "correct",
    "qwen_correct",
    "qwen_extracted_answer",
    "judge_output",
    "post_processed_output",
}
FORBIDDEN_OPEN_PROMPT_TEXT = (
    "\noptions:\n",
    "please select the correct answer",
    "provide the correct option letter",
)
BENCHMARK_DIRS = {
    "MathVista_MINI": "mathvista_mini",
    "MMStar_OpenEnded": "mmstar_open_ended",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the 20-experiment raw rollout sweep.")
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    root = args.sweep_root.expanduser().resolve()
    manifest_path = (args.manifest or root / "effective_experiment_manifest.tsv").resolve()
    output_path = (args.output or root / "sweep_validation.json").resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        experiments = list(csv.DictReader(handle, delimiter="\t"))
    if len(experiments) != 20:
        raise RuntimeError(f"Expected 20 effective experiments, got {len(experiments)}.")

    experiment_reports: list[dict[str, Any]] = []
    identity_by_benchmark: dict[str, dict[str, str]] = defaultdict(dict)
    failures: list[str] = []
    incomplete = 0
    for experiment in experiments:
        benchmark = experiment["benchmark"]
        ratio_tag = experiment["ratio_tag"]
        decode_mode = experiment["decode_mode"]
        expected_mode = "greedy" if decode_mode == "greedy" else "sample"
        expected_per_sample = 1 if decode_mode == "greedy" else 64
        expected_records = 100 * expected_per_sample
        run_dir = root / BENCHMARK_DIRS[benchmark] / ratio_tag / decode_mode
        raw_path = run_dir / "raw_outputs.jsonl"
        manifest_file = run_dir / "run_manifest.json"
        run_validation_file = run_dir / "raw_validation.json"
        count = 0
        unique_keys: set[tuple[str, int]] = set()
        sample_counts: dict[str, int] = defaultdict(int)
        prefix_hashes: dict[str, set[str | None]] = defaultdict(set)
        forbidden_fields: set[str] = set()
        retention_values: list[float] = []
        local_failures: list[str] = []
        if not raw_path.exists():
            incomplete += 1
            local_failures.append("raw_outputs_missing")
        else:
            with raw_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    count += 1
                    sample_id = str(row["sample_id"])
                    key = (sample_id, int(row["rollout_index"]))
                    if key in unique_keys:
                        local_failures.append(f"duplicate_record:{key}")
                    unique_keys.add(key)
                    sample_counts[sample_id] += 1
                    prefix_hashes[sample_id].add(row.get("prefix_hash"))
                    forbidden_fields.update(FORBIDDEN_FIELDS.intersection(row))
                    if row.get("benchmark") != benchmark:
                        local_failures.append(f"benchmark_mismatch:line={line_number}")
                    if row.get("decode_mode") != expected_mode:
                        local_failures.append(f"decode_mode_mismatch:line={line_number}")
                    if not isinstance(row.get("generated_token_ids"), list):
                        local_failures.append(f"missing_token_ids:line={line_number}")
                    if not isinstance(row.get("raw_generated_text"), str):
                        local_failures.append(f"missing_raw_text:line={line_number}")
                    if benchmark == "MMStar_OpenEnded":
                        prompt = str(row.get("prompt", "")).lower()
                        matched = [text for text in FORBIDDEN_OPEN_PROMPT_TEXT if text in prompt]
                        if matched:
                            local_failures.append(f"option_prompt_leak:line={line_number}:{matched}")
                    identity = canonical_hash(
                        {
                            "image_sha256": row["image_sha256"],
                            "question": row["question"],
                            "prompt": row["prompt"],
                            "reference_answer": row.get("reference_answer"),
                        }
                    )
                    previous = identity_by_benchmark[benchmark].setdefault(sample_id, identity)
                    if previous != identity:
                        local_failures.append(f"cross_condition_identity_mismatch:{sample_id}")
                    if row["pruning"] == "visionzip":
                        retention_values.append(float(row["realized_retention_ratio"]))
        complete = count == expected_records
        if not complete:
            incomplete += int(raw_path.exists())
        if forbidden_fields:
            local_failures.append(f"forbidden_fields:{sorted(forbidden_fields)}")
        if complete:
            if len(sample_counts) != 100 or any(value != expected_per_sample for value in sample_counts.values()):
                local_failures.append("per_sample_record_count_mismatch")
            if decode_mode == "sample64" and any(len(values) != 1 for values in prefix_hashes.values()):
                local_failures.append("sample_prefix_hash_mismatch")
            if retention_values:
                target = float(experiment["ratio"])
                if any(abs(value - target) > 0.011 for value in retention_values):
                    local_failures.append("realized_retention_out_of_tolerance")
        if run_validation_file.exists():
            run_validation = json.loads(run_validation_file.read_text(encoding="utf-8"))
            if run_validation.get("status") != "passed":
                local_failures.append("runner_validation_failed")
        elif complete:
            local_failures.append("runner_validation_missing")
        if manifest_file.exists():
            run_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if run_manifest.get("post_processing_applied") is not False:
                local_failures.append("manifest_postprocessing_not_false")
        elif raw_path.exists():
            local_failures.append("run_manifest_missing")
        failures.extend(f"{benchmark}/{ratio_tag}/{decode_mode}:{item}" for item in set(local_failures))
        experiment_reports.append(
            {
                **experiment,
                "run_dir": str(run_dir),
                "expected_records": expected_records,
                "observed_records": count,
                "complete": complete,
                "runner_validation_exists": run_validation_file.exists(),
                "local_failures": sorted(set(local_failures)),
            }
        )

    substantive_failures = [
        failure
        for failure in failures
        if not failure.endswith("raw_outputs_missing")
        and "runner_validation_missing" not in failure
    ]
    status = "passed" if incomplete == 0 and not failures else "incomplete" if not substantive_failures else "failed"
    report = {
        "status": status,
        "raw_only": True,
        "expected_experiments": 20,
        "complete_experiments": sum(report["complete"] for report in experiment_reports),
        "incomplete_experiments": incomplete,
        "failure_count": len(substantive_failures),
        "failures": sorted(substantive_failures),
        "experiments": experiment_reports,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({key: report[key] for key in report if key != "experiments"}, indent=2))
    if status == "failed":
        return 1
    if status == "incomplete" and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
