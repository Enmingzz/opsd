#!/usr/bin/env python3
"""Fail-fast validation for the paired 20k SFT and progressive OPSD retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OPSD_ROOT = EXPERIMENT_ROOT.parents[2]
PROJECT_ROOT = OPSD_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from opsd.visionzip_aokvqa.paired_sampling import (  # noqa: E402
    paired_retention_ratio,
    paired_rollout_seed,
)
from opsd.visionzip_aokvqa.train import (  # noqa: E402
    checkpoint_contract_sha256,
    sample_retention_ratio,
)


CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
DATASET = Path(
    "/project/6101803/enmingzz/opsd/data/"
    "openmmreasoner_llava_cot_train20k_ordered_decontam_v1_seed42/"
    "train20k_exact_resume_order_qwentok512_imgtok1152_seed42.jsonl"
)
MANIFEST = DATASET.with_name("train20k_exact_resume_training_manifest.json")
PARENT_RANDOM_OUTPUT = Path(
    "/scratch/enmingzz/outputs/llm_only/native_budget_weighting_dropout0_pair_20260730/"
    "original_opsd_dropout0"
)
EXPECTED_DATA_SHA256 = "9f1cf9dfbff291ee0ce7f34a236820c7da907f096bdb54ec0165eed845f3516f"
EXPECTED_PHASE_COUNTS = {0.4: 4_992, 0.3: 4_992, 0.2: 4_992, 0.1: 5_024}
EXPECTED_PHASE_ENDS = [4_992, 9_984, 14_976, 20_000]
EXPECTED_LAYERS = list(range(28))
EXPECTED_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests/preflight_validation.json",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_dataset_ids() -> list[str]:
    sample_ids: list[str] = []
    with DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_ids.append(str(row.get("sample_id", row.get("id", ""))))
    return sample_ids


def assert_common_config(cfg: dict[str, Any], method: str) -> None:
    training = cfg["training"]
    require(cfg["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct", "base model mismatch")
    require(cfg["dataset"]["name"] == str(DATASET), "dataset path mismatch")
    require(cfg["dataset"]["decontamination_manifest"] == str(MANIFEST), "manifest mismatch")
    require(cfg["dataset"]["expected_rows"] == 20_000, "expected_rows mismatch")
    require(cfg["dataset"]["shuffle"] is False, "ordered 20k data must not be shuffled")
    require(training["method"] == method, f"method mismatch: {training['method']} != {method}")
    require(training["max_steps"] == 20_000 and training["start_step"] == 0, "20k horizon mismatch")
    require(training["lora_dropout"] == 0.0, "LoRA dropout must be zero")
    require(training["learning_rate"] == 2.0e-5, "learning rate must be fixed at 2e-5")
    require(training["weight_decay"] == 0.0, "weight decay mismatch")
    require(training["micro_batch_size"] == 1, "micro batch mismatch")
    require(training["gradient_accumulation_steps"] == 8, "gradient accumulation mismatch")
    require(training["lora_layers_to_transform"] == EXPECTED_LAYERS, "LLM LoRA layer scope mismatch")
    require(training["lora_layers_pattern"] == "layers", "LoRA layer pattern mismatch")
    require(training["target_modules"] == EXPECTED_TARGETS, "LoRA target modules mismatch")
    require(training["expected_trainable_tensors"] == 392, "trainable tensor contract mismatch")
    require(training["expected_trainable_parameters"] == 40_370_176, "trainable parameter contract mismatch")
    require(cfg["experiment"]["parameter_scope"] == "language_decoder_only", "scope is not LLM-only")
    require(cfg["experiment"]["vision_encoder_lora"] is False, "vision encoder LoRA must be disabled")
    require(cfg["generation"]["max_new_tokens"] == 512, "generation cap mismatch")
    require(cfg["prompt"]["enable_thinking"] is True, "reasoning prompt mismatch")
    require(cfg["pruning"]["method"] == "visionzip", "pruner mismatch")
    require(cfg["dataset"]["min_pixels"] == 1080 * 28 * 28, "min_pixels mismatch")
    require(cfg["dataset"]["max_pixels"] == 1080 * 28 * 28, "max_pixels mismatch")
    checkpointing = cfg["checkpointing"]
    require(checkpointing["eval_snapshot_every"] == 256, "snapshot interval mismatch")
    require(checkpointing["resumable_every"] == 1024, "resume interval mismatch")
    require(checkpointing["save_step_zero"] is True, "step-zero snapshot must be enabled")
    require(checkpointing["save_final_full"] is True, "full final checkpoint must be enabled")
    require(checkpointing["log_sample_assignments"] is True, "per-rank assignment audit log is disabled")


def observed_parent_ratios() -> dict[int, float]:
    observed: dict[int, float] = {}
    for rank in range(4):
        path = PARENT_RANDOM_OUTPUT / f"rank{rank}_paired_sampling.jsonl"
        require(path.is_file(), f"missing paired parent log: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for index, ratio in zip(row["global_indices"], row["retention_ratios"]):
                observed[int(index)] = float(ratio)
    return observed


def main() -> int:
    args = parse_args()
    stage1 = load_yaml(CONFIG_ROOT / "progressive_opsd_stage1.yaml")
    stage2 = load_yaml(CONFIG_ROOT / "progressive_opsd_stage2.yaml")
    sft = load_yaml(CONFIG_ROOT / "sft20k.yaml")
    assert_common_config(stage1, "opsd_nogt")
    assert_common_config(stage2, "opsd_nogt")
    assert_common_config(sft, "sft")

    require(DATASET.is_file() and MANIFEST.is_file(), "20k dataset or manifest is missing")
    require(sha256_file(DATASET) == EXPECTED_DATA_SHA256, "20k dataset hash mismatch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest["status"] == "passed", "dataset manifest is not passed")
    require(manifest["selected_rows"] == 20_000, "dataset manifest row mismatch")
    require(manifest["output_sha256"] == EXPECTED_DATA_SHA256, "manifest hash mismatch")
    require(manifest["independent_postbuild_audit"]["status"] == "passed", "data leakage audit failed")
    require(manifest["qwen_processor_token_check"]["status"] == "passed", "processor audit failed")
    require(manifest["qwen_target_token_len"]["max"] <= 512, "target token cap exceeded")
    require(manifest["qwen_prompt_image_tokens"]["max"] <= 1152, "image token cap exceeded")

    sample_ids = load_dataset_ids()
    require(len(sample_ids) == 20_000 and len(set(sample_ids)) == 20_000, "dataset IDs are not 20k unique")

    require(stage1["checkpointing"]["stop_at_step"] == 9_984, "stage-1 stop mismatch")
    require(stage2["checkpointing"]["stop_at_step"] == 20_000, "stage-2 stop mismatch")
    require(not stage1["checkpointing"]["resume_from"], "stage 1 must initialize from base")
    require(stage2["checkpointing"]["resume_from"].endswith("step_009984"), "stage-2 resume mismatch")
    require(
        checkpoint_contract_sha256(stage1) == checkpoint_contract_sha256(stage2),
        "segmented progressive configs have different checkpoint contracts",
    )
    for cfg in (stage1, stage2):
        require(cfg["paired_sampling"]["enabled"] is False, "progressive schedule must not enable paired native mode")
        require(cfg["pruning"]["retention_ratio_schedule"] == "progressive", "progressive schedule mismatch")
        require(cfg["pruning"]["progressive_phase_end_steps"] == EXPECTED_PHASE_ENDS, "phase ends mismatch")
        require(cfg["opsd"]["teacher_strategy"] == "ema", "OPSD teacher mismatch")
        require(cfg["opsd"]["ema_decay"] == 0.9999, "EMA decay mismatch")
        require(cfg["opsd"]["teacher_ground_truth_access"] is False, "teacher sees ground truth")
        require(cfg["generation"]["require_kv_cache"] is True, "rollout KV cache must be required")

    progressive_counts: Counter[float] = Counter()
    for index, sample_id in enumerate(sample_ids):
        ratio = sample_retention_ratio(
            stage1,
            __import__("random").Random(42),
            progress_step=index,
            total_steps=20_000,
            sample_id=sample_id,
        )
        progressive_counts[ratio] += 1
    require(dict(progressive_counts) == EXPECTED_PHASE_COUNTS, f"progressive counts mismatch: {progressive_counts}")

    require(sft["pruning"]["retention_ratio_schedule"] == "paired_deterministic_uniform", "SFT ratio pairing disabled")
    require(sft["paired_sampling"]["enabled"] is False, "SFT must not enable OPSD paired-native validation")
    namespace = sft["paired_sampling"]["namespace"]
    ratio_seed = int(sft["paired_sampling"]["ratio_seed"])
    rollout_seed = int(sft["paired_sampling"]["rollout_seed"])
    sft_ratios: list[float] = []
    for index, sample_id in enumerate(sample_ids):
        sft_ratios.append(
            paired_retention_ratio(
                sft["pruning"]["train_retention_ratios"],
                seed=ratio_seed,
                global_index=index,
                sample_id=sample_id,
                namespace=namespace,
            )
        )
        paired_rollout_seed(
            seed=rollout_seed,
            global_index=index,
            sample_id=sample_id,
            namespace=namespace,
        )
    parent_ratios = observed_parent_ratios()
    require(set(parent_ratios) == set(range(9_984)), "parent paired-ratio coverage mismatch")
    require(
        all(math.isclose(sft_ratios[index], parent_ratios[index]) for index in range(9_984)),
        "SFT first-9984 ratio assignments are not paired with the original OPSD run",
    )

    report = {
        "status": "passed",
        "dataset": str(DATASET),
        "dataset_sha256": EXPECTED_DATA_SHA256,
        "dataset_rows": len(sample_ids),
        "data_leakage_audit": "passed",
        "qwen_target_token_max": manifest["qwen_target_token_len"]["max"],
        "qwen_image_token_max": manifest["qwen_prompt_image_tokens"]["max"],
        "effective_batch_size_4gpu": 32,
        "lora_dropout": 0.0,
        "learning_rate": 2.0e-5,
        "learning_rate_schedule": "constant",
        "parameter_scope": "language_decoder_only",
        "trainable_tensors_expected": 392,
        "trainable_parameters_expected": 40_370_176,
        "progressive_phase_ends": EXPECTED_PHASE_ENDS,
        "progressive_phase_counts": {str(key): value for key, value in EXPECTED_PHASE_COUNTS.items()},
        "stage_checkpoint_contract_sha256": checkpoint_contract_sha256(stage1),
        "sft_ratio_counts": {str(key): value for key, value in sorted(Counter(sft_ratios).items())},
        "sft_first_9984_ratios_match_original_opsd": True,
        "progressive_segments": [
            {"start": 0, "stop": 9_984, "ratios": [0.4, 0.3]},
            {"start": 9_984, "stop": 20_000, "ratios": [0.2, 0.1]},
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
