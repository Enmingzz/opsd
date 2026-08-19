#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rows(path: Path) -> list[dict]:
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

    nonempty_errors = [str(p) for p in args.run_dir.glob("rank*_errors.jsonl") if p.stat().st_size]
    if nonempty_errors:
        raise AssertionError(f"Non-empty rank error logs: {nonempty_errors}")

    indices: list[int] = []
    ratio_count = 0
    for rank in range(4):
        for row in rows(args.run_dir / f"rank{rank}_sample_assignments.jsonl"):
            ratios = [float(value) for value in row["retention_ratios"]]
            if any(value != 0.10 for value in ratios):
                raise AssertionError(f"Unexpected ratios on rank {rank}: {ratios}")
            ratio_count += len(ratios)
            indices.extend(int(value) for value in row["global_indices"])
    if sorted(indices) != list(range(args.expected_steps)):
        raise AssertionError("Global sample assignments are missing or duplicated")
    if ratio_count != args.expected_steps:
        raise AssertionError(f"Expected {args.expected_steps} r010 assignments, got {ratio_count}")

    training_rows = rows(args.run_dir / "training_log.jsonl")
    losses = [float(row["loss"]) for row in training_rows]
    if int(training_rows[-1]["step"]) != args.expected_steps:
        raise AssertionError(f"Final step is {training_rows[-1]['step']}")
    if not all(math.isfinite(loss) for loss in losses):
        raise AssertionError("Training loss contains non-finite values")
    snapshots = list((args.run_dir / "eval_snapshots").glob("step_*"))
    if len(snapshots) != args.expected_steps // 256 + 1:
        raise AssertionError(f"Unexpected snapshot count: {len(snapshots)}")
    if not (args.run_dir / "final").exists():
        raise AssertionError("Missing final checkpoint pointer")

    payload = {
        "status": "passed",
        "global_samples": len(indices),
        "optimizer_updates": args.expected_steps // 32,
        "ratio_counts": {"0.10": ratio_count},
        "final_step": int(training_rows[-1]["step"]),
        "loss_min": min(losses),
        "loss_mean": sum(losses) / len(losses),
        "loss_max": max(losses),
        "eval_snapshot_count": len(snapshots),
        "final_checkpoint": str((args.run_dir / "final").resolve()),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
