#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/train_10240.yaml"
DEFAULT_OUTPUT = HERE / "manifests/preflight.json"
PAIRED_NAMESPACE = "opsd_random_r015_r025_r035_dropout0_20260808_v1"
OLD_DATASET = Path(
    "/project/6101803/enmingzz/opsd/data/"
    "openmmreasoner_llava_cot_train10k_decontam_v1_seed42/"
    "train10k_decontam_qwentok512_imgtok1152_seed42.jsonl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [str(json.loads(line)["sample_id"]) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}; pass --overwrite")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tr = cfg["training"]
    pr = cfg["pruning"]
    ps = cfg["paired_sampling"]
    ckpt = cfg["checkpointing"]
    assert cfg["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg["experiment"]["parameter_scope"] == "language_decoder_only"
    assert cfg["experiment"]["vision_encoder_lora"] is False
    assert tr["method"] == "opsd_nogt"
    max_steps = int(tr["max_steps"])
    assert max_steps in {10240, 20000}
    assert tr["lora_dropout"] == 0.0 and tr["learning_rate"] == 2e-5
    assert tr["weight_decay"] == 0.0
    assert tr["expected_trainable_tensors"] == 392
    assert tr["expected_trainable_parameters"] == 40_370_176
    assert 4 * tr["micro_batch_size"] * tr["gradient_accumulation_steps"] == 32
    assert pr["method"] == "visionzip"
    assert pr["retention_ratio_schedule"] == "paired_deterministic_uniform"
    assert pr["train_retention_ratios"] == [0.10]
    assert ps["enabled"] is True and ps["allow_custom_retention_ratios"] is True
    assert ps["namespace"] == PAIRED_NAMESPACE
    assert ps["ratio_seed"] == ps["rollout_seed"] == 42
    assert cfg["opsd"]["teacher_strategy"] == "ema"
    assert cfg["opsd"]["ema_decay"] == 0.9999
    assert cfg["opsd"]["teacher_ground_truth_access"] is False
    assert cfg["opsd"]["native_budget_weighting"]["enabled"] is False
    assert cfg["generation"]["temperature"] == 0.0
    assert cfg["generation"]["require_kv_cache"] is True
    assert ckpt["eval_snapshot_every"] == 256 and ckpt["resumable_every"] == 1024
    stop_at_step = int(ckpt.get("stop_at_step", max_steps) or max_steps)
    resume_from = str(ckpt.get("resume_from", "") or "")
    if max_steps == 10240:
        assert stop_at_step == 10240 and not resume_from
        variant = "r010_10240"
    elif stop_at_step == 10240:
        assert not resume_from
        variant = "r010_20k_stage1"
    else:
        assert stop_at_step == 20000
        assert resume_from.endswith("/resume_checkpoints/step_010240")
        variant = "r010_20k_stage2"

    dataset = Path(cfg["dataset"]["name"])
    sample_ids = ids(dataset)
    assert len(sample_ids) == len(set(sample_ids)) == 20000
    old_ids = ids(OLD_DATASET)
    random.Random(42).shuffle(old_ids)
    assert sample_ids[:10000] == old_ids

    payload = {
        "status": "passed",
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256(dataset),
        "dataset_rows": len(sample_ids),
        "original_first_10000_sequence_exact": True,
        "variant": variant,
        "global_samples_target": max_steps,
        "segment_stop_step": stop_at_step,
        "segment_start_step": 10240 if resume_from else 0,
        "optimizer_updates_target": max_steps // 32,
        "effective_batch_size": 32,
        "ratio_counts_expected": {"0.10": max_steps},
        "paired_namespace": ps["namespace"],
        "lora_dropout": tr["lora_dropout"],
        "trainable_parameters": tr["expected_trainable_parameters"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
