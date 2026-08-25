#!/usr/bin/env python3
"""Validate and consolidate the complete checkpoint/intervention campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "outputs"
DEFAULT_SAMPLES = (
    HERE.parents[1]
    / "data/openmmreasoner_llava_cot_holdout1k_decontam_v1_seed42/holdout1k_metric_samples.jsonl"
)
STEPS = tuple(range(0, 10241, 1024))
DELTAS = ("d01", "d02", "d05")
PAIRS = ("A", "B", "C")
METRICS = ("jsd", "forward_kl", "reverse_kl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sample_filename(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16] + ".json"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def candidate_paths(step_root: Path, kind: str, filename: str) -> list[Path]:
    direct = step_root / kind / "samples" / filename
    candidates = [direct] if direct.is_file() else []
    candidates.extend(sorted((step_root / "shards").glob(f"*/{kind}/samples/{filename}")))
    return candidates


def nested_metrics_close(first: Any, second: Any) -> bool:
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            nested_metrics_close(first[key], second[key]) for key in first
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            nested_metrics_close(left, right) for left, right in zip(first, second)
        )
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(float(first), float(second), rel_tol=1e-5, abs_tol=1e-7)
    return first == second


def choose_duplicate_safe(paths: list[Path], kind: str, sample_id: str) -> tuple[Path, int]:
    if not paths:
        raise FileNotFoundError(f"Missing {kind} for {sample_id}")
    payloads = [path.read_bytes() for path in paths]
    if any(payload != payloads[0] for payload in payloads[1:]):
        parsed = [json.loads(payload) for payload in payloads]
        key = "generated_token_ids"
        if any(row.get(key) != parsed[0].get(key) for row in parsed[1:]):
            raise ValueError(f"Conflicting duplicate {kind} token IDs for {sample_id}: {paths}")
        if kind == "scores" and any(
            not nested_metrics_close(row.get("metrics"), parsed[0].get("metrics"))
            for row in parsed[1:]
        ):
            raise ValueError(f"Conflicting duplicate metrics for {sample_id}: {paths}")
    return paths[0], len(paths) - 1


def validate_score(score: dict[str, Any], step: int, sample_id: str) -> int:
    if int(score["checkpoint_step"]) != step or str(score["sample_id"]) != sample_id:
        raise ValueError(f"Identity mismatch for step={step} sample={sample_id}")
    count = int(score["generated_tokens"])
    if count <= 0 or len(score["generated_token_ids"]) != count:
        raise ValueError(f"Invalid generated token count for {sample_id}")
    if len(score["generated_token_text"]) != count or len(score["is_eos"]) != count:
        raise ValueError(f"Generated token metadata mismatch for {sample_id}")
    for delta in DELTAS:
        for pair in PAIRS:
            for metric in METRICS:
                values = score["metrics"][delta][pair][metric]
                if len(values) != count:
                    raise ValueError(f"Length mismatch: {sample_id}/{delta}/{pair}/{metric}")
                if any(not math.isfinite(float(value)) or float(value) < -1e-7 for value in values):
                    raise ValueError(f"Invalid divergence: {sample_id}/{delta}/{pair}/{metric}")
    for metric in METRICS:
        reference = score["metrics"]["d01"]["A"][metric]
        if any(score["metrics"][delta]["A"][metric] != reference for delta in ("d02", "d05")):
            raise ValueError(f"A was not reused exactly: {sample_id}/{metric}")
    counts = [int(score["native_student_visual_tokens"])] + [
        int(score["interventions"][delta]["native_student_plus_visual_tokens"])
        for delta in DELTAS
    ]
    if not counts[0] < counts[1] < counts[2] < counts[3]:
        raise ValueError(f"Native token counts are not strictly increasing for {sample_id}: {counts}")
    return count


def token_tables(score: dict[str, Any]):
    import pyarrow as pa

    count = int(score["generated_tokens"])
    columns: dict[str, Any] = {
        "checkpoint_step": [int(score["checkpoint_step"])] * count,
        "sample_index": [int(score["sample_index"])] * count,
        "sample_id": [str(score["sample_id"])] * count,
        "token_index": list(range(count)),
        "token_id": [int(value) for value in score["generated_token_ids"]],
        "token_text": [str(value) for value in score["generated_token_text"]],
        "is_eos": [bool(value) for value in score["is_eos"]],
        "student_ratio": [0.10] * count,
    }
    for delta in DELTAS:
        for pair in PAIRS:
            for metric in METRICS:
                columns[f"{delta}_{pair}_{metric}"] = score["metrics"][delta][pair][metric]
    return pa.Table.from_pydict(columns)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    samples = read_jsonl(args.samples.resolve())
    if len(samples) != 1000:
        raise ValueError(f"Expected canonical LCOT-1K, found {len(samples)} rows")
    destination = output_root / "consolidated"
    destination.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    parquet_tmp = destination / f".per_token_metrics_all_steps.parquet.tmp.{os.getpid()}"
    rollout_tmp = destination / f".rollouts_all_steps.jsonl.tmp.{os.getpid()}"
    score_tmp = destination / f".scores_all_steps.jsonl.tmp.{os.getpid()}"
    writer: pq.ParquetWriter | None = None
    duplicate_files = 0
    total_tokens = 0
    total_truncated = 0
    completed_cells = 0
    missing: list[dict[str, Any]] = []
    per_step: dict[str, Any] = {}
    try:
        with rollout_tmp.open("w", encoding="utf-8") as rollout_handle, score_tmp.open(
            "w", encoding="utf-8"
        ) as score_handle:
            for step in STEPS:
                step_root = output_root / f"step_{step:06d}"
                step_samples = 0
                step_tokens = 0
                step_truncated = 0
                for expected_index, source in enumerate(samples):
                    sample_id = str(source["sample_id"])
                    filename = sample_filename(sample_id)
                    try:
                        rollout_path, rollout_duplicates = choose_duplicate_safe(
                            candidate_paths(step_root, "rollouts", filename), "rollouts", sample_id
                        )
                        score_path, score_duplicates = choose_duplicate_safe(
                            candidate_paths(step_root, "scores", filename), "scores", sample_id
                        )
                    except FileNotFoundError:
                        missing.append({"checkpoint_step": step, "sample_id": sample_id})
                        if args.allow_partial:
                            continue
                        raise
                    duplicate_files += rollout_duplicates + score_duplicates
                    rollout = read_json(rollout_path)
                    score = read_json(score_path)
                    count = validate_score(score, step, sample_id)
                    if int(score["sample_index"]) != expected_index:
                        raise ValueError(f"Sample order mismatch for {sample_id} at step {step}")
                    if rollout["generated_token_ids"] != score["generated_token_ids"]:
                        raise ValueError(f"Rollout/score prefix mismatch for {sample_id} at step {step}")
                    rollout_handle.write(json.dumps(rollout, ensure_ascii=False, sort_keys=True) + "\n")
                    score_handle.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")
                    table = token_tables(score)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            parquet_tmp,
                            table.schema,
                            compression="zstd",
                            use_dictionary=["sample_id", "token_text"],
                        )
                    writer.write_table(table)
                    total_tokens += count
                    total_truncated += int(count >= 1024)
                    step_tokens += count
                    step_truncated += int(count >= 1024)
                    step_samples += 1
                    completed_cells += 1
                per_step[str(step)] = {
                    "samples": step_samples,
                    "tokens": step_tokens,
                    "mean_generated_tokens": (
                        float(step_tokens) / float(step_samples) if step_samples else None
                    ),
                    "max_length_rollouts": step_truncated,
                }
    finally:
        if writer is not None:
            writer.close()

    rollout_tmp.replace(destination / "rollouts_all_steps.jsonl")
    score_tmp.replace(destination / "scores_all_steps.jsonl")
    if writer is not None:
        parquet_tmp.replace(destination / "per_token_metrics_all_steps.parquet")
    complete = completed_cells == len(STEPS) * len(samples)
    manifest = {
        "schema_version": "r010_lcot1k_fixed_teacher_delta_campaign_v1",
        "status": "complete" if complete else "partial",
        "checkpoint_steps": list(STEPS),
        "sample_count_per_step": len(samples),
        "checkpoint_sample_cells": completed_cells,
        "expected_checkpoint_sample_cells": len(STEPS) * len(samples),
        "token_rows": total_tokens,
        "max_length_rollouts": total_truncated,
        "per_token_metric_count": 27,
        "interventions": [0.01, 0.02, 0.05],
        "teacher_source": "fixed step-0 Qwen2.5-VL-7B base, full visual tokens",
        "student_prefix": "checkpoint-specific greedy r010 rollout",
        "duplicate_files_verified": duplicate_files,
        "missing": missing,
        "per_step": per_step,
    }
    atomic_text(destination / "campaign_manifest.json", json.dumps(manifest, indent=2) + "\n")
    if not complete and not args.allow_partial:
        raise RuntimeError(f"Campaign incomplete: {completed_cells}/{len(STEPS) * len(samples)}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
