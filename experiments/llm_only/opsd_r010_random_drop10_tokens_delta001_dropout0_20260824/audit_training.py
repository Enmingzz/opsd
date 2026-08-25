#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=10240)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.run_dir / "final_training_audit.json"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}; pass --overwrite")

    errors = [str(path) for path in args.run_dir.glob("rank*_errors.jsonl") if path.stat().st_size]
    if errors:
        raise AssertionError(f"Non-empty rank error logs: {errors}")
    indices: list[int] = []
    metrics: list[dict] = []
    for rank in range(4):
        for row in read_rows(args.run_dir / f"rank{rank}_sample_assignments.jsonl"):
            assert all(float(ratio) == 0.1 for ratio in row["retention_ratios"])
            indices.extend(int(index) for index in row["global_indices"])
        metrics.extend(read_rows(args.run_dir / f"rank{rank}_native_budget_metrics.jsonl"))
    assert sorted(indices) == list(range(args.expected_steps))
    assert len(metrics) == args.expected_steps

    for row in metrics:
        valid = int(row["native_token_partition_valid_tokens"])
        eligible = int(row["native_token_partition_eligible_tokens"])
        dropped = int(row["native_token_partition_random_dropped_tokens"])
        kept = int(row["native_token_partition_selected_tokens"])
        assert row["loss_type"] == "opsd_nogt_token_random_drop20_forward_kl"
        assert row["native_token_partition_top_fraction"] == 0.1
        assert row["native_token_partition_min_teacher_kl"] == 0.0
        assert row["native_token_partition_below_kl_floor_tokens"] == 0
        assert row["native_token_partition_excluded_low_kl_tokens"] == 0
        assert valid == eligible == dropped + kept
        assert dropped == math.ceil(0.10 * valid)
        assert kept > 0
        assert math.isfinite(row["weighted_kl_loss"])
        assert math.isfinite(row["unweighted_kl_loss"])

    training_rows = read_rows(args.run_dir / "training_log.jsonl")
    assert int(training_rows[-1]["step"]) == args.expected_steps
    snapshots = [path for path in (args.run_dir / "eval_snapshots").glob("step_*") if (path / "COMPLETE").is_file()]
    assert len(snapshots) == args.expected_steps // 256 + 1
    assert (args.run_dir / "final" / "COMPLETE").is_file()
    payload = {
        "status": "passed",
        "global_samples": len(indices),
        "optimizer_updates": args.expected_steps // 32,
        "mean_actual_dropped_fraction": sum(
            row["native_token_partition_random_dropped_tokens"]
            / row["native_token_partition_valid_tokens"]
            for row in metrics
        ) / len(metrics),
        "mean_retained_to_full_loss_ratio": sum(
            row["native_weighted_to_unweighted_kl_ratio"] for row in metrics
        ) / len(metrics),
        "eval_snapshot_count": len(snapshots),
        "final_checkpoint": str((args.run_dir / "final").resolve()),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
