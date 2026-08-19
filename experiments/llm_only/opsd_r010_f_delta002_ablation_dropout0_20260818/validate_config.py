#!/usr/bin/env python3
"""Fail-closed validation for the r010 -> r012 F-weighting ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("/project/6101803/enmingzz/opsd")
OUTPUT_ROOT = Path(
    "/scratch/enmingzz/outputs/llm_only/"
    "opsd_r010_f_delta002_ablation_dropout0_20260818"
)
VARIANTS = {"global_f_affine", "global_f_curriculum"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_parent_parity(
    cfg: dict[str, Any], parent: dict[str, Any], variant: str
) -> None:
    for key in (
        "base_model",
        "dataset",
        "prompt",
        "training",
        "generation",
        "pruning",
        "paired_sampling",
        "checkpointing",
    ):
        assert cfg[key] == parent[key], f"Unexpected controlled-field change: {key}"

    expected_opsd = copy.deepcopy(parent["opsd"])
    assert expected_opsd["native_budget_weighting"]["budget_delta"] == 0.075
    expected_opsd["native_budget_weighting"]["budget_delta"] = 0.02
    if variant == "global_f_affine":
        expected_opsd["trajectory_weighting"]["calibration"]["normalized_mean"] = 0.2
        expected_opsd["trajectory_weighting"]["normalization_source"] = (
            "theoretical_clipped_F_with_fixed_center_0.20"
        )
    assert cfg["opsd"] == expected_opsd, "Unexpected OPSD protocol change"


def validate(cfg: dict[str, Any]) -> dict[str, Any]:
    experiment = cfg["experiment"]
    training = cfg["training"]
    variant = str(experiment["variant"])
    assert variant in VARIANTS
    assert cfg["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert experiment["family"] == "r010_f_delta002_ablation_dropout0"
    assert experiment["parameter_scope"] == "language_decoder_only"
    assert experiment["vision_encoder_lora"] is False
    assert Path(experiment["output_root"]) == OUTPUT_ROOT
    if variant == "global_f_affine":
        assert experiment["intervention_ablation_controlled_changes"] == (
            "native_budget_delta_0.075_to_0.02_and_affine_center_0.45_to_0.20"
        )
    else:
        assert experiment["intervention_ablation_only_change"] == (
            "native_budget_delta_0.075_to_0.02"
        )

    parent_path = ROOT / experiment["intervention_ablation_parent_config"]
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    validate_parent_parity(cfg, parent, variant)

    assert training["method"] == "opsd_nogt"
    assert training["seed"] == 42
    assert training["bf16"] is True
    assert training["use_lora"] is True
    assert training["lora_dropout"] == 0.0
    assert training["learning_rate"] == 2.0e-5
    assert training["weight_decay"] == 0.0
    assert training["expected_trainable_tensors"] == 392
    assert training["expected_trainable_parameters"] == 40_370_176
    assert training["lora_layers_to_transform"] == list(range(28))
    assert training["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    assert cfg["dataset"]["shuffle"] is False
    assert cfg["pruning"]["method"] == "visionzip"
    assert cfg["pruning"]["train_retention_ratios"] == [0.10]
    assert cfg["paired_sampling"] == parent["paired_sampling"]
    assert cfg["opsd"]["teacher_strategy"] == "ema"
    assert cfg["opsd"]["ema_decay"] == 0.9999

    native = cfg["opsd"]["native_budget_weighting"]
    assert native["enabled"] is True
    assert native["mode"] == "trajectory_probe"
    assert native["budget_delta_mode"] == "absolute"
    assert native["budget_delta"] == 0.02
    student_ratio = float(cfg["pruning"]["train_retention_ratios"][0])
    probe_ratio = student_ratio + float(native["budget_delta"])
    assert math.isclose(student_ratio, 0.10)
    assert math.isclose(probe_ratio, 0.12)

    trajectory = cfg["opsd"]["trajectory_weighting"]
    if variant == "global_f_affine":
        assert trajectory["mode"] == "global_calibrated_counterfactual_teachability_batch"
        assert trajectory["calibration"] == {
            "q05": 0.0,
            "q95": 1.0,
            "normalized_mean": 0.2,
        }
        assert trajectory["coefficient"] == 1.0
        formula = "w=1+(clip(F,0,1)-0.20)"
        weight_range = [0.8, 1.8]
    else:
        assert trajectory["mode"] == "global_f_intermediate_curriculum_batch"
        assert trajectory["gamma"] == 4.0
        formula = "w=4*clip(F,0,1)*(1-clip(F,0,1))"
        weight_range = [0.0, 1.0]
    assert trajectory["batch_renormalization"] is False

    max_steps = int(training["max_steps"])
    if max_steps == 10240:
        stage = "train"
        assert training["gradient_accumulation_steps"] == 8
        assert 4 * training["micro_batch_size"] * training["gradient_accumulation_steps"] == 32
        assert cfg["generation"]["max_new_tokens"] == 512
        assert cfg["checkpointing"]["eval_snapshot_every"] == 256
        assert cfg["checkpointing"]["resumable_every"] == 1024
        effective_batch_size = 32
    elif max_steps == 4:
        stage = "smoke"
        assert training["gradient_accumulation_steps"] == 4
        assert cfg["generation"]["max_new_tokens"] == 64
        effective_batch_size = 4
    else:
        raise AssertionError(f"Unexpected max_steps={max_steps}")

    return {
        "status": "passed",
        "variant": variant,
        "stage": stage,
        "parent_config": str(parent_path),
        "parent_config_sha256": sha256(parent_path),
        "student_ratio": student_ratio,
        "probe_ratio": probe_ratio,
        "intervention": 0.02,
        "formula": formula,
        "theoretical_weight_range": weight_range,
        "effective_batch_size": effective_batch_size,
        "trainable_scope": "language_decoder_only_lora",
        "trainable_tensors": 392,
        "trainable_parameters": 40_370_176,
        "lora_dropout": 0.0,
        "controlled_parent_parity": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}; pass --overwrite")
    config_path = args.config.expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = validate(cfg)
    payload.update({"config": str(config_path), "config_sha256": sha256(config_path)})
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
