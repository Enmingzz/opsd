#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/train_10240.yaml"
PAIRED_NAMESPACE = "opsd_random_r015_r025_r035_dropout0_20260808_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-top-fraction", type=float)
    parser.add_argument("--expected-high-group-lambda", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}; pass --overwrite")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tr = cfg["training"]
    pr = cfg["pruning"]
    ps = cfg["paired_sampling"]
    opsd = cfg["opsd"]
    native = opsd["native_budget_weighting"]
    ckpt = cfg["checkpointing"]

    assert cfg["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg["experiment"]["parameter_scope"] == "language_decoder_only"
    assert cfg["experiment"]["vision_encoder_lora"] is False
    assert tr["method"] == "opsd_nogt"
    if args.expected_steps is not None:
        assert tr["max_steps"] == args.expected_steps
    assert tr["lora_dropout"] == 0.0
    assert tr["learning_rate"] == 2e-5 and tr["weight_decay"] == 0.0
    assert tr["expected_trainable_tensors"] == 392
    assert tr["expected_trainable_parameters"] == 40_370_176
    assert tr["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
    assert tr["lora_layers_to_transform"] == list(range(28))
    assert pr["method"] == "visionzip"
    assert pr["retention_ratio_schedule"] == "paired_deterministic_uniform"
    assert pr["train_retention_ratios"] == [0.10]
    assert ps["enabled"] is True and ps["allow_custom_retention_ratios"] is True
    assert ps["namespace"] == PAIRED_NAMESPACE
    assert ps["ratio_seed"] == ps["rollout_seed"] == tr["seed"] == 42
    assert opsd["teacher_strategy"] == "ema" and opsd["use_ema_teacher"] is True
    assert opsd["ema_decay"] == 0.9999
    assert opsd["teacher_ground_truth_access"] is False
    assert native["enabled"] is True
    assert native["mode"] == "token_projection_mass_grouped"
    assert native["budget_delta_mode"] == "absolute"
    assert native["budget_delta"] == 0.02
    assert 0.0 < native["top_fraction"] < 1.0
    assert 0.0 < native["high_group_lambda"] < 1.0
    if args.expected_top_fraction is not None:
        assert native["top_fraction"] == args.expected_top_fraction
    if args.expected_high_group_lambda is not None:
        assert native["high_group_lambda"] == args.expected_high_group_lambda
    assert native["preserve_loss_mass"] is False
    assert cfg["dataset"]["shuffle"] is False
    assert cfg["dataset"]["expected_rows"] == 20000
    assert cfg["generation"]["temperature"] == 0.0
    assert cfg["generation"]["require_kv_cache"] is True
    assert ckpt["save_step_zero"] is True and ckpt["save_final_full"] is True

    payload = {
        "status": "passed",
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "max_steps": tr["max_steps"],
        "effective_batch_size_4gpu": 4
        * tr["micro_batch_size"]
        * tr["gradient_accumulation_steps"],
        "trainable_tensors": tr["expected_trainable_tensors"],
        "trainable_parameters": tr["expected_trainable_parameters"],
        "retention_ratio": 0.10,
        "probe_ratio": 0.12,
        "top_fraction": native["top_fraction"],
        "high_group_lambda": native["high_group_lambda"],
        "per_token_high_low_weight_ratio": (
            native["high_group_lambda"] / native["top_fraction"]
        )
        / ((1.0 - native["high_group_lambda"]) / (1.0 - native["top_fraction"])),
        "direct_grouped_objective": not native["preserve_loss_mass"],
        "paired_namespace": ps["namespace"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
