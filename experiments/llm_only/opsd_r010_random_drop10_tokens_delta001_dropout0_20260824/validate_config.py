#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "opsd_random_r010_only_dropout0_20260815/configs/train_10240.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(cfg: dict) -> dict:
    normalized = deepcopy(cfg)
    for key in ("name", "family", "comparison_id", "objective"):
        normalized["experiment"].pop(key, None)
    normalized["opsd"]["native_budget_weighting"] = {"enabled": False}
    for key in ("trajectory_weighting", "token_outlier_exclusion", "token_kl_floor_filter"):
        if normalized["opsd"].get(key) == {"enabled": False}:
            normalized["opsd"].pop(key)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}; pass --overwrite")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    if int(args.expected_steps) == 10240 and normalize(cfg) != normalize(reference):
        raise AssertionError("Candidate differs from the r010-only baseline outside the random-drop objective.")

    native = cfg["opsd"]["native_budget_weighting"]
    assert native == {
        "enabled": True,
        "mode": "token_random_drop20",
        "budget_delta_mode": "absolute",
        "budget_delta": 0.01,
        "sensitivity_temperature": 1.0,
        "kl_chunk_size": 32,
        "eps": 1e-8,
        "top_fraction": 0.10,
        "min_teacher_kl": 0.0,
        "random_drop_seed": 42,
    }
    assert cfg["training"]["max_steps"] == int(args.expected_steps)
    assert cfg["training"]["lora_dropout"] == 0.0
    assert cfg["training"]["expected_trainable_tensors"] == 392
    assert cfg["training"]["expected_trainable_parameters"] == 40_370_176
    assert cfg["training"]["learning_rate"] == 2e-5
    assert cfg["training"]["weight_decay"] == 0.0
    assert cfg["pruning"]["method"] == "visionzip"
    assert cfg["pruning"]["train_retention_ratios"] == [0.1]
    assert cfg["opsd"]["teacher_strategy"] == "ema"
    assert cfg["opsd"]["ema_decay"] == 0.9999
    assert cfg["opsd"]["teacher_ground_truth_access"] is False
    assert cfg["opsd"]["trajectory_weighting"] == {"enabled": False}
    assert cfg["opsd"]["token_outlier_exclusion"] == {"enabled": False}
    assert cfg["opsd"]["token_kl_floor_filter"] == {"enabled": False}
    if int(args.expected_steps) == 10240:
        assert cfg["training"]["gradient_accumulation_steps"] == 8
        assert cfg["training"]["micro_batch_size"] == 1
        assert cfg["generation"]["max_new_tokens"] == 512
        assert cfg["checkpointing"]["eval_snapshot_every"] == 256
        assert cfg["checkpointing"]["resumable_every"] == 1024

    dataset = Path(cfg["dataset"]["name"])
    payload = {
        "status": "passed",
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "reference_config": str(REFERENCE.resolve()),
        "reference_config_sha256": sha256(REFERENCE),
        "dataset_sha256": sha256(dataset),
        "student_ratio": 0.10,
        "probe_ratio": 0.11,
        "drop_fraction": 0.10,
        "selection": "deterministic_random_over_all_valid_response_tokens",
        "kl_floor_used_for_selection": False,
        "objective": "mean(KL(q_full||p_r010) over valid non-dropped tokens)",
        "expected_steps": int(args.expected_steps),
        "trainable_tensors": 392,
        "trainable_parameters": 40_370_176,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
