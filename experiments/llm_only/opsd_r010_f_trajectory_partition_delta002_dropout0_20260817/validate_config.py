#!/usr/bin/env python3
"""Fail-closed preflight for the two 10%->12% trajectory partition runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


VARIANTS = {"trajectory_top20", "trajectory_bottom80"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}")
    config_path = args.config.expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp, train = cfg["experiment"], cfg["training"]
    variant = str(exp["variant"])
    assert variant in VARIANTS
    assert cfg["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert exp["parameter_scope"] == "language_decoder_only"
    assert exp["vision_encoder_lora"] is False
    assert exp["init_model"] == cfg["base_model"]
    assert exp["metric_intervention_pair"] == "native_visionzip_r010_to_r012"
    assert train["method"] == "opsd_nogt"
    assert train["max_steps"] == 10240 and train["start_step"] == 0
    assert train["seed"] == 42 and train["bf16"] is True
    assert train["use_lora"] is True and train["lora_dropout"] == 0.0
    assert train["learning_rate"] == 2.0e-5 and train["weight_decay"] == 0.0
    assert train["micro_batch_size"] == 1 and train["gradient_accumulation_steps"] == 8
    assert train["expected_trainable_tensors"] == 392
    assert train["expected_trainable_parameters"] == 40_370_176
    assert cfg["dataset"]["expected_rows"] == 20000
    assert cfg["dataset"]["shuffle"] is False
    assert cfg["pruning"]["method"] == "visionzip"
    assert cfg["pruning"]["train_retention_ratios"] == [0.10]
    assert cfg["paired_sampling"]["enabled"] is True
    assert cfg["paired_sampling"]["ratio_seed"] == 42
    assert cfg["paired_sampling"]["rollout_seed"] == 42
    assert cfg["opsd"]["teacher_strategy"] == "ema"
    assert cfg["opsd"]["use_ema_teacher"] is True
    assert cfg["opsd"]["ema_decay"] == 0.9999
    native = cfg["opsd"]["native_budget_weighting"]
    assert native["enabled"] is True and native["mode"] == "trajectory_probe"
    assert native["budget_delta_mode"] == "absolute"
    assert native["budget_delta"] == 0.02
    trajectory = cfg["opsd"]["trajectory_weighting"]
    selection = "top20" if variant.endswith("top20") else "bottom80"
    assert trajectory["enabled"] is True
    assert trajectory["mode"] == f"trajectory_projection_fraction_{selection}_batch"
    assert trajectory["top_fraction"] == 0.2
    assert trajectory["normalization"] == "probability_sum_one"
    assert trajectory["normalization_scope"] == "effective_batch"
    assert cfg["generation"]["max_new_tokens"] == 512
    assert cfg["generation"]["temperature"] == 0.0
    assert cfg["generation"]["require_kv_cache"] is True
    assert cfg["checkpointing"]["eval_snapshot_every"] == 256
    assert cfg["checkpointing"]["resumable_every"] == 1024
    payload = {
        "status": "passed",
        "variant": variant,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "student_ratio": 0.10,
        "probe_ratio": 0.12,
        "budget_delta": 0.02,
        "selection": selection,
        "trainable_scope": "language_decoder_only_lora",
        "trainable_parameters": 40_370_176,
        "lora_dropout": 0.0,
        "effective_batch_size": 32,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
