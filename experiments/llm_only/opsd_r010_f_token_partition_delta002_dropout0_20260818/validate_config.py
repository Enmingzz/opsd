#!/usr/bin/env python3
"""Fail closed unless a config is the intended r010-to-r012 token ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = cfg["experiment"]
    training = cfg["training"]
    native = cfg["opsd"]["native_budget_weighting"]
    variant = str(experiment["variant"])
    assert variant in {"token_top20", "token_bottom80", "token_random_drop20"}
    assert experiment["family"] == "r010_f_token_partition_delta002_dropout0"
    assert experiment["budget_intervention"] == "r010_to_r012"
    assert cfg["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg["pruning"]["method"] == "visionzip"
    assert cfg["pruning"]["train_retention_ratios"] == [0.1]
    assert training["method"] == "opsd_nogt"
    assert training["max_steps"] == 10240
    assert training["lora_dropout"] == 0.0
    assert training["expected_trainable_tensors"] == 392
    assert training["expected_trainable_parameters"] == 40_370_176
    assert training["learning_rate"] == 2.0e-5
    assert training["gradient_accumulation_steps"] == 8
    assert 4 * training["micro_batch_size"] * training["gradient_accumulation_steps"] == 32
    assert cfg["generation"]["max_new_tokens"] == 512
    assert cfg["generation"]["require_kv_cache"] is True
    assert cfg["opsd"]["teacher_strategy"] == "ema"
    assert cfg["opsd"]["ema_decay"] == 0.9999
    assert native["enabled"] is True
    expected_mode = (
        "token_random_drop20"
        if variant == "token_random_drop20"
        else f"token_projection_fraction_{variant.removeprefix('token_')}"
    )
    assert native["mode"] == expected_mode
    assert native["budget_delta_mode"] == "absolute"
    assert native["budget_delta"] == 0.02
    assert native["top_fraction"] == 0.2
    assert native["min_teacher_kl"] == 1.0e-5
    if variant == "token_random_drop20":
        assert native["random_drop_seed"] == 42
    assert cfg["opsd"]["trajectory_weighting"] == {"enabled": False}
    payload = {
        "status": "PASS",
        "variant": variant,
        "student_ratio": 0.10,
        "probe_ratio": 0.12,
        "budget_delta": native["budget_delta"],
        "selection_fraction": native["top_fraction"],
        "selection": (
            "deterministic_random_drop_from_eligible_tokens"
            if variant == "token_random_drop20"
            else "projection_fraction_partition"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
