#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


PAIRED_NAMESPACE = "opsd_random_r015_r025_r035_dropout0_20260808_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}; pass --overwrite")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = cfg["training"]
    native = cfg["opsd"]["native_budget_weighting"]
    assert cfg["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg["experiment"]["parameter_scope"] == "language_decoder_only"
    assert cfg["experiment"]["vision_encoder_lora"] is False
    assert training["method"] == "opsd_nogt"
    assert training["max_steps"] == args.expected_steps
    assert training["lora_dropout"] == 0.0
    assert training["learning_rate"] == 2e-5
    assert training["weight_decay"] == 0.0
    assert training["expected_trainable_tensors"] == 392
    assert training["expected_trainable_parameters"] == 40_370_176
    assert training["lora_layers_to_transform"] == list(range(28))
    assert cfg["pruning"]["method"] == "visionzip"
    assert cfg["pruning"]["train_retention_ratios"] == [0.10]
    assert cfg["paired_sampling"]["namespace"] == PAIRED_NAMESPACE
    assert cfg["paired_sampling"]["ratio_seed"] == 42
    assert cfg["paired_sampling"]["rollout_seed"] == 42
    assert cfg["opsd"]["teacher_strategy"] == "ema"
    assert cfg["opsd"]["use_ema_teacher"] is True
    assert cfg["opsd"]["ema_decay"] == 0.9999
    assert cfg["opsd"]["teacher_ground_truth_access"] is False
    assert native["enabled"] is True
    assert native["mode"] == "token_projection_fraction_grouped"
    assert native["selection"] == "bottom"
    assert native["budget_delta_mode"] == "absolute"
    assert native["budget_delta"] == 0.01
    assert native["top_fraction"] == 0.20
    assert native["min_teacher_kl"] == 0.0
    assert native["selected_group_lambda"] == 0.30
    assert native["preserve_loss_mass"] is False
    assert cfg["opsd"]["trajectory_weighting"]["enabled"] is False
    assert cfg["opsd"]["token_kl_floor_filter"]["enabled"] is False
    assert cfg["dataset"]["shuffle"] is False
    assert cfg["dataset"]["expected_rows"] == 20000
    assert cfg["generation"]["temperature"] == 0.0
    assert cfg["generation"]["require_kv_cache"] is True

    payload = {
        "status": "passed",
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "max_steps": training["max_steps"],
        "effective_batch_size_4gpu": 32,
        "trainable_tensors": training["expected_trainable_tensors"],
        "trainable_parameters": training["expected_trainable_parameters"],
        "student_ratio": 0.10,
        "probe_ratio": 0.11,
        "selection": "bottom",
        "ranking_eligibility_floor": None,
        "selected_fraction_of_all_valid_tokens": 0.20,
        "selected_group_lambda": 0.30,
        "complement_group_lambda": 0.70,
        "objective": "0.30*mean(KL_bottom20F)+0.70*mean(KL_complement)",
        "mean_token_weight": 1.0,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
