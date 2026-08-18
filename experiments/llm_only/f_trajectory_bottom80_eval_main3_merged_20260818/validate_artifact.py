#!/usr/bin/env python3
"""Validate source LoRA adapters and provenance of merged checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("source", "finalize", "merged"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--merged", type=Path)
    parser.add_argument("--final-path", type=Path)
    return parser.parse_args()


def validate_source(args: argparse.Namespace) -> dict:
    adapter = args.adapter.resolve()
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    assert adapter.joinpath("COMPLETE").read_text().strip() == "complete"
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert int(cfg["training"]["max_steps"]) == 10240
    assert float(cfg["training"]["lora_dropout"]) == 0.0
    assert cfg["experiment"]["parameter_scope"] == "language_decoder_only"
    assert cfg["pruning"]["method"] == "visionzip"
    assert [float(value) for value in cfg["pruning"]["train_retention_ratios"]] == [0.1]

    audit = json.loads(run_root.joinpath("final_training_audit.json").read_text())
    assert audit["status"] == "passed"
    assert int(audit["final_step"]) == 10240
    assert int(audit["global_samples"]) == 10240
    assert {f"{float(k):.2f}": int(v) for k, v in audit["ratio_counts"].items()} == {
        "0.10": 10240
    }

    adapter_cfg = json.loads(adapter.joinpath("adapter_config.json").read_text())
    assert adapter_cfg["base_model_name_or_path"] == cfg["base_model"]
    assert float(adapter_cfg["lora_dropout"]) == 0.0
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    assert len(keys) == 392
    assert not any(".visual." in key or "merger" in key for key in keys)
    return {
        "status": "passed",
        "method": args.method,
        "adapter": str(adapter),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "adapter_tensors": len(keys),
        "checkpoint_step": 10240,
        "parameter_scope": "language_decoder_only",
        "lora_dropout": 0.0,
    }


def finalize(args: argparse.Namespace, source: dict) -> None:
    temporary = args.merged.resolve()
    final = args.final_path.resolve()
    metadata_path = temporary / "merge_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert Path(metadata["adapter_path"]).resolve() == args.adapter.resolve()
    metadata.update(
        {
            "output_dir": str(final),
            "adapter_model_sha256": source["adapter_sha256"],
            "source_training_job_id": str(args.training_job_id),
            "source_training_config": str(args.config.resolve()),
            "source_final_global_step": 10240,
            "training_method": args.method,
            "training_objective": args.objective,
            "train_retention_ratios": [0.1],
            "parameter_scope": "language_decoder_only",
            "lora_dropout": 0.0,
            "merge_semantics": "peft_merge_and_unload_default",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def validate_merged(args: argparse.Namespace, source: dict) -> dict:
    merged = args.merged.resolve()
    metadata = json.loads(merged.joinpath("merge_metadata.json").read_text())
    assert Path(metadata["adapter_path"]).resolve() == args.adapter.resolve()
    assert Path(metadata["output_dir"]).resolve() == merged
    assert metadata["adapter_model_sha256"] == source["adapter_sha256"]
    assert metadata["source_training_job_id"] == str(args.training_job_id)
    assert int(metadata["source_final_global_step"]) == 10240
    assert metadata["training_method"] == args.method
    assert metadata["training_objective"] == args.objective
    assert metadata["parameter_scope"] == "language_decoder_only"
    assert float(metadata["lora_dropout"]) == 0.0
    assert metadata["train_retention_ratios"] == [0.1]
    assert merged.joinpath("model.safetensors.index.json").is_file()
    shards = list(merged.glob("model-*.safetensors"))
    assert len(shards) == 4, shards
    return {**source, "merged_model": str(merged), "model_shards": len(shards)}


def main() -> None:
    args = parse_args()
    source = validate_source(args)
    if args.mode == "source":
        result = source
    elif args.mode == "finalize":
        assert args.merged is not None and args.final_path is not None
        finalize(args, source)
        result = {**source, "status": "finalized", "temporary": str(args.merged.resolve())}
    else:
        assert args.merged is not None
        result = validate_merged(args, source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
