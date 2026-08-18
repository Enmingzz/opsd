#!/usr/bin/env python3
"""Create or validate a state-preserving resume fork for an ordered data extension."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open


OPSD_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = OPSD_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from opsd.visionzip_aokvqa.data_integrity import (  # noqa: E402
    verify_decontaminated_training_data,
)
from opsd.visionzip_aokvqa.paired_sampling import (  # noqa: E402
    paired_retention_ratio,
    paired_rollout_seed,
)
from opsd.visionzip_aokvqa.train import checkpoint_contract, checkpoint_contract_sha256  # noqa: E402


ALLOWED_PARENT_CONTRACT_DIFFERENCES = {
    "experiment.name",
    "experiment.dataset_tag",
    "dataset.name",
    "dataset.decontamination_manifest",
    "dataset.expected_rows",
    "dataset.shuffle",
    "training.max_steps",
    "checkpointing.resume_mode",
}
ALLOWED_PARENT_CONTRACT_PREFIXES = ("checkpointing.resume_fork.",)
STATE_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "ema_shadow.pt",
    "optimizer.pt",
    "README.md",
)
COPIED_LOGS = (
    "training_log.jsonl",
    "student_text_outputs.jsonl",
    "rank0_paired_sampling.jsonl",
    "rank1_paired_sampling.jsonl",
    "rank2_paired_sampling.jsonl",
    "rank3_paired_sampling.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            output.update(flatten(item, path))
        else:
            output[path] = item
    return output


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def read_raw_jsonl_rows(path: Path) -> list[bytes]:
    with path.open("rb") as handle:
        return [line.rstrip(b"\r\n") for line in handle if line.strip()]


def sample_id_from_raw_row(row: bytes) -> str:
    payload = json.loads(row)
    return str(payload.get("sample_id", payload.get("id", "")))


def observed_parent_sample_order(parent_output: Path, expected_rows: int) -> list[str]:
    observed: dict[int, str] = {}
    for rank in range(4):
        path = parent_output / f"rank{rank}_paired_sampling.jsonl"
        require(path.is_file(), f"parent paired-sampling log is missing for rank {rank}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for global_index, sample_id in zip(row["global_indices"], row["sample_ids"]):
                observed[int(global_index)] = str(sample_id)
    require(len(observed) == expected_rows, f"parent logs cover {len(observed)} rows, expected {expected_rows}")
    return [observed[index] for index in range(expected_rows)]


def state_step_values(optimizer_state: dict[str, Any]) -> set[int]:
    values: set[int] = set()
    for state in optimizer_state["state"].values():
        step = state.get("step")
        if step is not None:
            values.add(int(step.item() if hasattr(step, "item") else step))
    return values


def allowed_contract_difference(path: str) -> bool:
    return path in ALLOWED_PARENT_CONTRACT_DIFFERENCES or any(
        path.startswith(prefix) for prefix in ALLOWED_PARENT_CONTRACT_PREFIXES
    )


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def validate_plan(config_path: Path, output_dir: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resume_cfg = cfg.get("checkpointing", {}).get("resume_fork", {})
    require(
        cfg.get("checkpointing", {}).get("resume_mode") == "state_preserving_ordered_data_extension",
        "checkpointing.resume_mode is not the ordered data-extension mode",
    )
    require(all(bool(resume_cfg.get(key)) for key in ("inherit_optimizer", "inherit_rank_rng", "inherit_ema")), "all resume state must be inherited")

    parent_checkpoint = Path(resume_cfg["parent_checkpoint"]).resolve()
    require(parent_checkpoint.joinpath("COMPLETE").is_file(), "parent checkpoint is incomplete")
    parent_output = parent_checkpoint.parent.parent.resolve()
    parent_config_path = parent_checkpoint / "config_resolved.yaml"
    parent_cfg = yaml.safe_load(parent_config_path.read_text(encoding="utf-8"))
    parent_state = json.loads(parent_checkpoint.joinpath("trainer_state.json").read_text(encoding="utf-8"))
    resume_step = int(resume_cfg["resume_global_step"])
    require(int(parent_state["global_step"]) == resume_step == 9984, "parent global step mismatch")
    require(int(parent_state["world_size"]) == 4, "parent world size mismatch")
    require(
        parent_state["config_contract_sha256"] == checkpoint_contract_sha256(parent_cfg),
        "parent checkpoint contract does not match its saved config",
    )

    parent_flat = flatten(checkpoint_contract(parent_cfg))
    current_flat = flatten(checkpoint_contract(cfg))
    contract_differences = {
        key: {"parent": parent_flat.get(key), "current": current_flat.get(key)}
        for key in sorted(set(parent_flat) | set(current_flat))
        if parent_flat.get(key) != current_flat.get(key)
    }
    forbidden_differences = {
        key: value
        for key, value in contract_differences.items()
        if not allowed_contract_difference(key)
    }
    require(not forbidden_differences, f"resume changes training semantics: {forbidden_differences}")

    parent_dataset = Path(resume_cfg["parent_dataset"]).resolve()
    combined_dataset = Path(cfg["dataset"]["name"]).resolve()
    require(parent_dataset == Path(parent_cfg["dataset"]["name"]).resolve(), "parent dataset path mismatch")
    require(sha256_file(parent_dataset) == resume_cfg["parent_dataset_sha256"], "parent dataset hash mismatch")
    require(sha256_file(combined_dataset) == resume_cfg["combined_dataset_sha256"], "combined dataset hash mismatch")
    parent_rows = count_lines(parent_dataset)
    combined_rows = count_lines(combined_dataset)
    require(parent_rows == int(resume_cfg["parent_dataset_rows"]) == 10_000, "parent row count mismatch")
    require(combined_rows == int(resume_cfg["combined_dataset_rows"]) == 20_000, "combined row count mismatch")
    require(int(cfg["dataset"]["expected_rows"]) == combined_rows, "configured expected rows mismatch")
    require(bool(parent_cfg.get("dataset", {}).get("shuffle", True)), "parent dataset was not shuffled")
    require(cfg["dataset"].get("shuffle") is False, "exact-resume ordered dataset must disable loader shuffle")

    parent_raw_rows = read_raw_jsonl_rows(parent_dataset)
    expected_parent_order = parent_raw_rows[:]
    random.Random(int(training_seed := cfg["training"]["seed"])).shuffle(expected_parent_order)
    combined_raw_rows = read_raw_jsonl_rows(combined_dataset)
    require(
        combined_raw_rows[:parent_rows] == expected_parent_order,
        "combined logical prefix is not the parent's seed-shuffled 10k training order",
    )

    additional_dataset = Path(resume_cfg["additional_dataset"]).resolve()
    require(
        sha256_file(additional_dataset) == resume_cfg["additional_dataset_sha256"],
        "additional dataset hash mismatch",
    )
    additional_raw_rows = read_raw_jsonl_rows(additional_dataset)
    require(len(additional_raw_rows) == 10_000, "additional dataset row count mismatch")
    expected_additional_order = additional_raw_rows[:]
    random.Random(int(training_seed)).shuffle(expected_additional_order)
    require(
        combined_raw_rows[parent_rows:] == expected_additional_order,
        "combined logical suffix is not the expected seed-shuffled additional 10k order",
    )

    ordering_manifest_path = Path(resume_cfg["ordering_manifest"]).resolve()
    ordering_manifest = json.loads(ordering_manifest_path.read_text(encoding="utf-8"))
    require(ordering_manifest.get("status") == "passed", "ordering manifest did not pass")
    require(Path(ordering_manifest["output_jsonl"]).resolve() == combined_dataset, "ordering manifest path mismatch")
    require(ordering_manifest["output_sha256"] == sha256_file(combined_dataset), "ordering manifest hash mismatch")

    observed_parent_order = observed_parent_sample_order(parent_output, resume_step)
    logical_parent_ids = [sample_id_from_raw_row(row) for row in combined_raw_rows[:resume_step]]
    require(
        logical_parent_ids == observed_parent_order,
        "combined logical prefix does not match the sample order recorded by the parent run",
    )
    require(int(resume_cfg["first_unconsumed_dataset_index"]) == resume_step, "resume dataset index mismatch")
    require(resume_step < parent_rows, "resume step must include the unconsumed parent tail")

    integrity = verify_decontaminated_training_data(
        combined_dataset,
        Path(cfg["dataset"]["decontamination_manifest"]),
        expected_rows=combined_rows,
    )
    training = cfg["training"]
    require(training["method"] == "opsd_nogt", "method mismatch")
    require(int(training["start_step"]) in (0, resume_step), "start step conflicts with parent")
    require(int(training["max_steps"]) > resume_step, "max steps must exceed parent step")
    require((int(training["max_steps"]) - resume_step) % 32 == 0, "remaining samples are not EBS-32 aligned")
    require(float(training["lora_dropout"]) == 0.0, "LoRA dropout is nonzero")
    require(float(training["learning_rate"]) == 2.0e-5, "learning rate mismatch")
    require(float(training["weight_decay"]) == 0.0, "weight decay mismatch")
    require(int(training["gradient_accumulation_steps"]) == 8, "gradient accumulation mismatch")
    require(int(training["micro_batch_size"]) == 1, "microbatch mismatch")
    require(cfg["experiment"]["parameter_scope"] == "language_decoder_only", "scope mismatch")
    require(cfg["experiment"]["vision_encoder_lora"] is False, "vision encoder must remain frozen")
    require(cfg["generation"]["require_kv_cache"] is True, "rollout KV cache must remain enabled")
    require(float(cfg["generation"]["temperature"]) == 0.0, "rollout temperature mismatch")
    require(cfg["opsd"]["native_budget_weighting"]["enabled"] is False, "original OPSD must be unweighted")
    require(cfg["opsd"]["use_ema_teacher"] is True, "EMA teacher must remain enabled")
    require(float(cfg["opsd"]["ema_decay"]) == 0.9999, "EMA decay mismatch")

    expected_bridge = output_dir.resolve() / "resume_checkpoints" / f"step_{resume_step:06d}"
    configured_bridge = Path(cfg["checkpointing"]["resume_from"]).resolve()
    require(configured_bridge == expected_bridge, f"resume bridge path mismatch: {configured_bridge}")
    require(output_dir.resolve() != parent_output, "resume fork must preserve the parent output")

    with safe_open(parent_checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        adapter_keys = list(handle.keys())
    require(len(adapter_keys) == 392, "parent adapter tensor count mismatch")
    require(not any(".visual." in key or "merger" in key for key in adapter_keys), "parent contains visual LoRA")
    adapter_cfg = json.loads(parent_checkpoint.joinpath("adapter_config.json").read_text(encoding="utf-8"))
    require(float(adapter_cfg["lora_dropout"]) == 0.0, "parent adapter dropout is nonzero")
    ema = torch.load(parent_checkpoint / "ema_shadow.pt", map_location="cpu", weights_only=True)
    require(isinstance(ema, dict) and len(ema) == 392, "parent EMA state mismatch")
    del ema
    optimizer = torch.load(parent_checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    require(len(optimizer["state"]) == 392, "parent optimizer tensor-state count mismatch")
    require(state_step_values(optimizer) == {312}, "parent AdamW state is not at update 312")
    group = optimizer["param_groups"][0]
    require(float(group["lr"]) == 2.0e-5 and float(group["weight_decay"]) == 0.0, "parent optimizer hyperparameters mismatch")
    del optimizer
    rng_hashes: dict[str, str] = {}
    for rank in range(4):
        path = parent_checkpoint / "rank_rng_states" / f"rank_{rank:02d}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        require(int(state["global_step"]) == resume_step, f"rank {rank} RNG step mismatch")
        rng_hashes[str(rank)] = sha256_file(path)

    ratios = [float(value) for value in cfg["pruning"]["train_retention_ratios"]]
    pairing = cfg["paired_sampling"]
    combined_lines = [row.decode("utf-8") for row in combined_raw_rows]
    next_rows: list[dict[str, Any]] = []
    for global_index in range(resume_step, min(resume_step + 32, int(training["max_steps"]))):
        row = json.loads(combined_lines[global_index])
        sample_id = str(row.get("sample_id", row.get("id", "")))
        next_rows.append(
            {
                "global_index": global_index,
                "sample_id": sample_id,
                "ratio": paired_retention_ratio(
                    ratios,
                    int(pairing["ratio_seed"]),
                    global_index,
                    sample_id,
                    str(pairing["namespace"]),
                ),
                "rollout_seed": paired_rollout_seed(
                    int(pairing["rollout_seed"]),
                    global_index,
                    sample_id,
                    str(pairing["namespace"]),
                ),
            }
        )

    return {
        "status": "validated",
        "config": str(config_path.resolve()),
        "config_contract_sha256": checkpoint_contract_sha256(cfg),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_output": str(parent_output),
        "parent_global_step": resume_step,
        "parent_optimizer_updates": 312,
        "parent_adapter_sha256": sha256_file(parent_checkpoint / "adapter_model.safetensors"),
        "parent_ema_sha256": sha256_file(parent_checkpoint / "ema_shadow.pt"),
        "parent_optimizer_sha256": sha256_file(parent_checkpoint / "optimizer.pt"),
        "parent_rng_sha256": rng_hashes,
        "parent_dataset": str(parent_dataset),
        "parent_dataset_rows": parent_rows,
        "parent_dataset_sha256": sha256_file(parent_dataset),
        "combined_dataset": str(combined_dataset),
        "combined_dataset_rows": combined_rows,
        "combined_dataset_sha256": sha256_file(combined_dataset),
        "loader_shuffle_disabled": True,
        "logical_parent_10k_order_byte_exact": True,
        "parent_observed_sequence_rows_compared": resume_step,
        "parent_observed_sequence_exact": True,
        "ordering_manifest": str(ordering_manifest_path),
        "ordering_manifest_sha256": sha256_file(ordering_manifest_path),
        "additional_dataset": str(additional_dataset),
        "additional_dataset_sha256": sha256_file(additional_dataset),
        "first_unconsumed_dataset_index": resume_step,
        "first_unconsumed_sample_id": next_rows[0]["sample_id"],
        "target_global_step": int(training["max_steps"]),
        "remaining_samples": int(training["max_steps"]) - resume_step,
        "remaining_optimizer_updates": (int(training["max_steps"]) - resume_step) // 32,
        "contract_differences": contract_differences,
        "forbidden_contract_differences": forbidden_differences,
        "data_integrity_checks": integrity["checks"],
        "next_32_sampling": next_rows,
    }


def validate_prepared(output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    bridge = output_dir / "resume_checkpoints" / "step_009984"
    require(bridge.joinpath("COMPLETE").is_file(), "prepared bridge is incomplete")
    bridge_state = json.loads(bridge.joinpath("trainer_state.json").read_text(encoding="utf-8"))
    require(int(bridge_state["global_step"]) == 9984, "bridge global step mismatch")
    require(int(bridge_state["world_size"]) == 4, "bridge world size mismatch")
    require(bridge_state["config_contract_sha256"] == plan["config_contract_sha256"], "bridge contract mismatch")
    parent = Path(plan["parent_checkpoint"])
    state_hashes: dict[str, dict[str, Any]] = {}
    for relative in STATE_FILES:
        parent_path = parent / relative
        bridge_path = bridge / relative
        require(bridge_path.is_file(), f"bridge state file missing: {relative}")
        parent_sha = sha256_file(parent_path)
        bridge_sha = sha256_file(bridge_path)
        require(parent_sha == bridge_sha, f"bridge state changed: {relative}")
        state_hashes[relative] = {
            "sha256": bridge_sha,
            "same_inode": parent_path.stat().st_ino == bridge_path.stat().st_ino,
        }
    rng_hashes: dict[str, str] = {}
    for rank in range(4):
        relative = Path("rank_rng_states") / f"rank_{rank:02d}.pt"
        require(sha256_file(parent / relative) == sha256_file(bridge / relative), f"rank {rank} RNG changed")
        rng_hashes[str(rank)] = sha256_file(bridge / relative)
    parent_output = Path(plan["parent_output"])
    copied_log_hashes: dict[str, str] = {}
    for relative in COPIED_LOGS:
        source = parent_output / relative
        destination = output_dir / relative
        require(destination.is_file(), f"copied parent log missing: {relative}")
        require(sha256_file(source) == sha256_file(destination), f"copied parent log changed: {relative}")
        copied_log_hashes[relative] = sha256_file(destination)
    return {
        **plan,
        "status": "prepared_and_verified",
        "bridge_checkpoint": str(bridge.resolve()),
        "state_files_byte_identical": True,
        "state_file_hashes": state_hashes,
        "rank_rng_hashes": rng_hashes,
        "copied_parent_logs_byte_identical": True,
        "copied_parent_log_hashes": copied_log_hashes,
    }


def prepare(config_path: Path, output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    if output_dir.exists():
        manifest_path = output_dir / "resume_fork_manifest.json"
        require(manifest_path.is_file(), f"output exists without a resume-fork manifest: {output_dir}")
        return validate_prepared(output_dir, plan)

    stage = output_dir.with_name(f".{output_dir.name}.prepare.{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    bridge = stage / "resume_checkpoints" / "step_009984"
    bridge.mkdir(parents=True)
    parent = Path(plan["parent_checkpoint"])
    link_modes: dict[str, str] = {}
    try:
        for relative in STATE_FILES:
            link_modes[relative] = hardlink_or_copy(parent / relative, bridge / relative)
        for rank in range(4):
            relative = Path("rank_rng_states") / f"rank_{rank:02d}.pt"
            link_modes[str(relative)] = hardlink_or_copy(parent / relative, bridge / relative)

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        resolved_cfg = copy.deepcopy(cfg)
        resolved_cfg["output_dir"] = str(output_dir.resolve())
        atomic_write_yaml(bridge / "config_resolved.yaml", resolved_cfg)
        atomic_write_json(
            bridge / "trainer_state.json",
            {
                "checkpoint_type": "full_resumable_lora_training_checkpoint",
                "global_step": 9984,
                "world_size": 4,
                "gradient_accumulation_remainder": 0,
                "config_contract_sha256": plan["config_contract_sha256"],
                "trajectory_curriculum_state": None,
                "resume_fork_mode": "state_preserving_ordered_data_extension",
                "resume_fork_parent_checkpoint": plan["parent_checkpoint"],
                "state_files_byte_identical_to_parent": True,
            },
        )
        bridge.joinpath("COMPLETE").write_text("complete\n", encoding="utf-8")

        parent_output = Path(plan["parent_output"])
        for relative in COPIED_LOGS:
            shutil.copy2(parent_output / relative, stage / relative)
        prepared = {
            **plan,
            "status": "prepared",
            "bridge_checkpoint": str(output_dir.resolve() / "resume_checkpoints" / "step_009984"),
            "state_file_materialization": link_modes,
        }
        atomic_write_json(stage / "resume_fork_manifest.json", prepared)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, output_dir)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return validate_prepared(output_dir, plan)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    plan = validate_plan(config_path, output_dir)
    report = validate_prepared(output_dir, plan) if args.check_only else prepare(config_path, output_dir, plan)
    report_path = args.report.resolve() if args.report else output_dir / "resume_fork_validation.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
