#!/usr/bin/env python3
"""Audit one completed segment of the 20k retraining runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OPSD_ROOT = EXPERIMENT_ROOT.parents[2]
PROJECT_ROOT = OPSD_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from opsd.visionzip_aokvqa.paired_sampling import paired_retention_ratio  # noqa: E402
from opsd.visionzip_aokvqa.train import checkpoint_contract_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("progressive", "sft"), required=True)
    parser.add_argument("--expected-stop", type=int, required=True)
    parser.add_argument("--expect-final", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def optimizer_steps(path: Path) -> set[int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(len(state["state"]) == 392, f"optimizer tensor count mismatch: {len(state['state'])}")
    return {
        int(value["step"].item() if hasattr(value["step"], "item") else value["step"])
        for value in state["state"].values()
    }


def expected_progressive_ratio(index: int) -> float:
    if index < 4_992:
        return 0.4
    if index < 9_984:
        return 0.3
    if index < 14_976:
        return 0.2
    return 0.1


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    stop = int(args.expected_stop)
    world_size = 4 if "SLURM_JOB_ID" in os.environ else int(os.environ.get("WORLD_SIZE", "1"))
    state_path = output / f"resume_checkpoints/step_{stop:06d}/trainer_state.json"
    require(state_path.is_file(), f"missing trainer state at step {stop}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    world_size = int(state["world_size"])
    require(stop % world_size == 0, "stop is not world-size aligned")
    require(int(state["global_step"]) == stop, "trainer state step mismatch")
    require(state["config_contract_sha256"] == checkpoint_contract_sha256(cfg), "checkpoint contract mismatch")

    dataset_rows = read_jsonl(Path(cfg["dataset"]["name"]))
    dataset_ids = [str(row.get("sample_id", row.get("id", ""))) for row in dataset_rows]
    require(len(dataset_ids) == 20_000, "dataset no longer contains 20k rows")
    verification = json.loads((output / "training_data_verification.json").read_text(encoding="utf-8"))
    verification_checks = verification.get("checks", {})
    require(verification_checks and all(verification_checks.values()), "runtime data verification failed")

    training_rows = read_jsonl(output / "training_log.jsonl")
    require(len(training_rows) == stop // world_size, "rank-0 training log length mismatch")
    require(int(training_rows[-1]["step"]) == stop, "training log final step mismatch")
    require(all(math.isfinite(float(row["loss"])) for row in training_rows), "non-finite loss found")
    require(
        all(math.isclose(float(row["learning_rate"]), 2.0e-5, rel_tol=0.0, abs_tol=1e-12) for row in training_rows),
        "learning rate is not fixed at 2e-5",
    )
    require(not any(path.stat().st_size for path in output.glob("rank*_errors.jsonl")), "rank error log is nonempty")

    observed: dict[int, tuple[str, float]] = {}
    for rank in range(world_size):
        rows = read_jsonl(output / f"rank{rank}_sample_assignments.jsonl")
        require(len(rows) == stop // world_size, f"rank {rank} sampling log length mismatch")
        for row in rows:
            for index, sample_id, ratio in zip(
                row["global_indices"], row["sample_ids"], row["retention_ratios"]
            ):
                index = int(index)
                require(index not in observed, f"duplicate global index {index}")
                observed[index] = (str(sample_id), float(ratio))
    require(set(observed) == set(range(stop)), "global sample coverage is not contiguous")
    ratio_counts: Counter[float] = Counter()
    for index in range(stop):
        sample_id, ratio = observed[index]
        require(sample_id == dataset_ids[index], f"sample order mismatch at {index}")
        if args.kind == "progressive":
            expected_ratio = expected_progressive_ratio(index)
        else:
            expected_ratio = paired_retention_ratio(
                cfg["pruning"]["train_retention_ratios"],
                seed=int(cfg["paired_sampling"]["ratio_seed"]),
                global_index=index,
                sample_id=sample_id,
                namespace=cfg["paired_sampling"]["namespace"],
            )
        require(math.isclose(ratio, expected_ratio), f"ratio mismatch at {index}: {ratio} != {expected_ratio}")
        ratio_counts[ratio] += 1

    checkpoint = state_path.parent
    require((checkpoint / "COMPLETE").is_file(), "checkpoint is incomplete")
    effective_batch_size = (
        world_size
        * int(cfg["training"]["micro_batch_size"])
        * int(cfg["training"]["gradient_accumulation_steps"])
    )
    require(
        optimizer_steps(checkpoint / "optimizer.pt") == {stop // effective_batch_size},
        "AdamW step count mismatch",
    )
    adapter_cfg = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    require(float(adapter_cfg["lora_dropout"]) == 0.0, "saved LoRA dropout is nonzero")
    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        adapter_keys = list(handle.keys())
    require(len(adapter_keys) == 392, f"adapter tensor count mismatch: {len(adapter_keys)}")
    require(
        not any(".visual." in key or "merger" in key for key in adapter_keys),
        "adapter contains vision or merger tensors",
    )
    if args.kind == "progressive":
        ema = torch.load(checkpoint / "ema_shadow.pt", map_location="cpu", weights_only=True)
        require(isinstance(ema, dict) and len(ema) == 392, "EMA shadow mismatch")
        del ema
    else:
        require(not (checkpoint / "ema_shadow.pt").exists(), "SFT unexpectedly saved EMA state")

    if args.expect_final:
        require((output / "final").is_symlink(), "final checkpoint link is missing")
        require((output / "final").resolve() == checkpoint.resolve(), "final link targets wrong checkpoint")
    else:
        require(not (output / "final").exists(), "intermediate segment incorrectly created final")
        marker = output / f"segment_complete_step_{stop:06d}.json"
        require(marker.is_file(), "intermediate segment marker is missing")

    snapshot_interval = int(cfg["checkpointing"]["eval_snapshot_every"])
    resume_interval = int(cfg["checkpointing"]["resumable_every"])
    expected_snapshots = [0] + list(range(snapshot_interval, stop + 1, snapshot_interval))
    actual_snapshots = sorted(
        int(path.parent.name.split("_")[-1]) for path in output.glob("eval_snapshots/step_*/COMPLETE")
    )
    require(actual_snapshots == expected_snapshots, "evaluation snapshot sequence mismatch")
    expected_resume = set(range(resume_interval, stop + 1, resume_interval))
    expected_resume.add(stop)
    for marker in output.glob("segment_complete_step_*.json"):
        marker_step = int(marker.stem.split("_")[-1])
        if marker_step <= stop:
            expected_resume.add(marker_step)
    actual_resume = sorted(
        int(path.parent.name.split("_")[-1]) for path in output.glob("resume_checkpoints/step_*/COMPLETE")
    )
    require(actual_resume == sorted(expected_resume), "resumable checkpoint sequence mismatch")

    report = {
        "status": "passed",
        "kind": args.kind,
        "expected_stop": stop,
        "world_size": world_size,
        "effective_batch_size": effective_batch_size,
        "optimizer_steps": stop // effective_batch_size,
        "all_losses_finite": True,
        "learning_rate": 2.0e-5,
        "lora_dropout": 0.0,
        "parameter_scope": "language_decoder_only",
        "adapter_tensors": len(adapter_keys),
        "ema_tensors": 392 if args.kind == "progressive" else 0,
        "sample_order_exact": True,
        "ratio_assignments_exact": True,
        "ratio_counts": {str(key): value for key, value in sorted(ratio_counts.items())},
        "checkpoint": str(checkpoint),
        "final": bool(args.expect_final),
    }
    report_path = output / f"post_training_audit_step_{stop:06d}.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
