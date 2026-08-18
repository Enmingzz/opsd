#!/usr/bin/env python3
"""Audit a completed exact-resume smoke or production run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open


PARENT_OUTPUT = Path(
    "/scratch/enmingzz/outputs/llm_only/native_budget_weighting_dropout0_pair_20260730/"
    "original_opsd_dropout0"
)
PARENT_CHECKPOINT = PARENT_OUTPUT / "resume_checkpoints/step_009984"
PARENT_STEP = 9984
PARENT_LOG_ROWS = 2496
WORLD_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def optimizer_steps(path: Path) -> set[int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    values = {
        int(item["step"].item() if hasattr(item["step"], "item") else item["step"])
        for item in state["state"].values()
        if "step" in item
    }
    if len(state["state"]) != 392:
        raise RuntimeError(f"optimizer state count mismatch: {len(state['state'])}")
    return values


def expected_periodic_steps(start: int, end: int, interval: int) -> list[int]:
    return [step for step in range(start + WORLD_SIZE, end + 1, WORLD_SIZE) if step % interval == 0]


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    final_step = int(cfg["training"]["max_steps"])
    remaining_samples = final_step - PARENT_STEP
    if remaining_samples <= 0 or remaining_samples % 32 != 0:
        raise RuntimeError(f"invalid continuation length: {remaining_samples}")
    local_rows = remaining_samples // WORLD_SIZE
    optimizer_updates_added = remaining_samples // 32
    if cfg["dataset"].get("shuffle") is not False:
        raise RuntimeError("exact-resume dataset must have shuffle=false")
    dataset_rows = [
        json.loads(line)
        for line in Path(cfg["dataset"]["name"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(dataset_rows) != int(cfg["dataset"]["expected_rows"]):
        raise RuntimeError(f"ordered dataset row mismatch: {len(dataset_rows)}")
    dataset_sample_ids = [
        str(row.get("sample_id", row.get("id", ""))) for row in dataset_rows
    ]

    parent_rows = [
        json.loads(line)
        for line in PARENT_OUTPUT.joinpath("training_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [
        json.loads(line)
        for line in output.joinpath("training_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(parent_rows) != PARENT_LOG_ROWS:
        raise RuntimeError(f"unexpected parent log rows: {len(parent_rows)}")
    if rows[:PARENT_LOG_ROWS] != parent_rows:
        raise RuntimeError("parent training-log prefix changed")
    new_rows = rows[PARENT_LOG_ROWS:]
    if len(new_rows) != local_rows:
        raise RuntimeError(f"continuation log rows mismatch: {len(new_rows)} != {local_rows}")
    if int(new_rows[0]["step"]) != PARENT_STEP + WORLD_SIZE:
        raise RuntimeError(f"first resumed log step mismatch: {new_rows[0]['step']}")
    if int(rows[-1]["step"]) != final_step:
        raise RuntimeError(f"final log step mismatch: {rows[-1]['step']} != {final_step}")
    if not all(math.isfinite(float(row["loss"])) for row in new_rows):
        raise RuntimeError("non-finite resumed loss found")
    if not all(math.isclose(float(row["learning_rate"]), 2.0e-5, rel_tol=0.0, abs_tol=1e-12) for row in new_rows):
        raise RuntimeError("resumed learning rate was not fixed at 2e-5")
    if any(row.get("native_budget_weighting_enabled") is True for row in new_rows):
        raise RuntimeError("weighted OPSD unexpectedly enabled")
    if any(output.glob("rank*_errors.jsonl")):
        raise RuntimeError("rank error log exists")

    for rank in range(WORLD_SIZE):
        parent_rank = [
            json.loads(line)
            for line in PARENT_OUTPUT.joinpath(f"rank{rank}_paired_sampling.jsonl").read_text().splitlines()
            if line.strip()
        ]
        rank_rows = [
            json.loads(line)
            for line in output.joinpath(f"rank{rank}_paired_sampling.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if rank_rows[:PARENT_LOG_ROWS] != parent_rank:
            raise RuntimeError(f"rank {rank} paired-sampling prefix changed")
        rank_new = rank_rows[PARENT_LOG_ROWS:]
        if len(rank_new) != local_rows:
            raise RuntimeError(f"rank {rank} resumed sampling rows mismatch")
        if int(rank_new[0]["global_indices"][0]) != PARENT_STEP + rank:
            raise RuntimeError(f"rank {rank} first dataset index mismatch: {rank_new[0]['global_indices']}")
        for row in rank_new:
            for global_index, sample_id in zip(row["global_indices"], row["sample_ids"]):
                expected_sample_id = dataset_sample_ids[int(global_index) % len(dataset_sample_ids)]
                if str(sample_id) != expected_sample_id:
                    raise RuntimeError(
                        f"rank {rank} data-order mismatch at index {global_index}: "
                        f"{sample_id} != {expected_sample_id}"
                    )

    snapshots = sorted(output.glob("eval_snapshots/step_*/COMPLETE"))
    expected_snapshots = expected_periodic_steps(PARENT_STEP, final_step, int(cfg["checkpointing"]["eval_snapshot_every"]))
    actual_snapshot_steps = [int(path.parent.name.split("_")[-1]) for path in snapshots]
    if actual_snapshot_steps != expected_snapshots:
        raise RuntimeError(f"snapshot steps mismatch: {actual_snapshot_steps} != {expected_snapshots}")

    resumable = sorted(output.glob("resume_checkpoints/step_*/COMPLETE"))
    expected_resumable = [PARENT_STEP] + expected_periodic_steps(
        PARENT_STEP,
        final_step,
        int(cfg["checkpointing"]["resumable_every"]),
    )
    if final_step not in expected_resumable:
        expected_resumable.append(final_step)
    expected_resumable = sorted(set(expected_resumable))
    actual_resumable_steps = [int(path.parent.name.split("_")[-1]) for path in resumable]
    if actual_resumable_steps != expected_resumable:
        raise RuntimeError(f"resumable steps mismatch: {actual_resumable_steps} != {expected_resumable}")

    final = output.joinpath("final").resolve()
    expected_final = output.joinpath(f"resume_checkpoints/step_{final_step:06d}").resolve()
    if final != expected_final:
        raise RuntimeError(f"final checkpoint mismatch: {final} != {expected_final}")
    state = json.loads(final.joinpath("trainer_state.json").read_text(encoding="utf-8"))
    if int(state["global_step"]) != final_step or int(state["world_size"]) != WORLD_SIZE:
        raise RuntimeError("final trainer state mismatch")
    expected_adam_step = 312 + optimizer_updates_added
    actual_adam_steps = optimizer_steps(final / "optimizer.pt")
    if actual_adam_steps != {expected_adam_step}:
        raise RuntimeError(f"AdamW step mismatch: {actual_adam_steps} != {{{expected_adam_step}}}")

    adapter_cfg = json.loads(final.joinpath("adapter_config.json").read_text(encoding="utf-8"))
    if float(adapter_cfg["lora_dropout"]) != 0.0:
        raise RuntimeError("saved adapter dropout is nonzero")
    with safe_open(final / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if len(keys) != 392 or any(".visual." in key or "merger" in key for key in keys):
        raise RuntimeError("saved adapter violates LLM-only scope")
    ema = torch.load(final / "ema_shadow.pt", map_location="cpu", weights_only=True)
    if not isinstance(ema, dict) or len(ema) != 392:
        raise RuntimeError("final EMA shadow mismatch")
    del ema
    adapter_changed = sha256_file(final / "adapter_model.safetensors") != sha256_file(
        PARENT_CHECKPOINT / "adapter_model.safetensors"
    )
    ema_changed = sha256_file(final / "ema_shadow.pt") != sha256_file(PARENT_CHECKPOINT / "ema_shadow.pt")
    if not adapter_changed or not ema_changed:
        raise RuntimeError("student adapter or EMA did not update after resume")

    report = {
        "status": "passed",
        "resume_mode": "state_preserving_ordered_data_extension",
        "config": str(args.config.resolve()),
        "output_dir": str(output),
        "parent_global_step": PARENT_STEP,
        "final_global_step": final_step,
        "parent_optimizer_step": 312,
        "final_optimizer_step": expected_adam_step,
        "remaining_training_samples": remaining_samples,
        "added_optimizer_updates": optimizer_updates_added,
        "full_training_log_rows": len(rows),
        "new_training_log_rows": len(new_rows),
        "parent_log_prefix_exact": True,
        "all_resumed_sample_ids_match_ordered_dataset": True,
        "dataset_shuffle": False,
        "first_resumed_log_step": int(new_rows[0]["step"]),
        "all_new_losses_finite": True,
        "fixed_learning_rate": 2.0e-5,
        "lora_dropout": 0.0,
        "trainable_adapter_tensors": len(keys),
        "visual_trainable_tensors": 0,
        "ema_shadow_tensors": 392,
        "adapter_changed": adapter_changed,
        "ema_changed": ema_changed,
        "eval_snapshot_steps": actual_snapshot_steps,
        "full_resumable_checkpoint_steps": actual_resumable_steps,
        "teacher_ground_truth_access": bool(cfg["opsd"]["teacher_ground_truth_access"]),
    }
    write_atomic(output / "post_training_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
