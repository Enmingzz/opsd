#!/usr/bin/env python
"""Compare final unmerged LoRA and merged BF16 inference on one fixed MMStar sample."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OPSD_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = OPSD_ROOT.parent
DEFAULT_MERGED = Path(
    "/scratch/enmingzz/outputs/llm_only/merged_models/"
    "llm_only_random_decontam_v1_20260713/opsd_qwen25vl7b_merged_bf16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "config_max1024.json")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(
            "/scratch/enmingzz/outputs/llm_only/checkpoints/"
            "llm_only_random_decontam_v1_20260713/opsd/final"
        ),
    )
    parser.add_argument("--merged-model", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--sample-id", default="812")
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs" / "merged_equivalence_final_sample812.json",
    )
    return parser.parse_args()


def load_worker() -> Any:
    path = Path(__file__).with_name("run_checkpoint.py")
    spec = importlib.util.spec_from_file_location("opsd_checkpoint_kl_worker", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_merged(cfg: dict[str, Any], merged_path: Path) -> tuple[Any, Any]:
    from opsd.visionzip_aokvqa.qwen_wrapper import load_qwen_model_and_processor

    model, processor = load_qwen_model_and_processor(
        str(merged_path),
        bf16=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        min_pixels=int(cfg["min_pixels"]),
        max_pixels=int(cfg["max_pixels"]),
        visionzip_official=True,
    )
    model.eval()
    return model, processor


def collect(
    worker: Any,
    model: Any,
    processor: Any,
    sample: Any,
    cfg: dict[str, Any],
    fixed_ids_cpu: Any | None = None,
) -> dict[str, Any]:
    import torch
    from opsd.visionzip_aokvqa.qwen_wrapper import encode_prompt, primary_device
    from opsd.visionzip_aokvqa.train import (
        sequence_inputs_from_prompt,
        temporary_cached_rollout,
        temporary_eval,
    )

    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root="", device=device)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    with temporary_cached_rollout(model):
        generated_ids, generated_text, generation_metadata = worker.generate_official_pruned(
            model,
            processor,
            prompt_inputs,
            retention_ratio=float(cfg["rollout_retention_ratio"]),
            max_new_tokens=int(cfg["max_new_tokens"]),
            contextual_ratio=float(cfg["contextual_ratio"]),
        )
    scoring_ids = (
        fixed_ids_cpu.to(device=device, dtype=generated_ids.dtype)
        if fixed_ids_cpu is not None
        else generated_ids
    )
    sequence_inputs = sequence_inputs_from_prompt(prompt_inputs, scoring_ids)
    with temporary_eval(model):
        logits, metadata = worker.score_official_pruned(
            model,
            sequence_inputs,
            prompt_len,
            int(scoring_ids.numel()),
            float(cfg["rollout_retention_ratio"]),
            float(cfg["contextual_ratio"]),
        )
    return {
        "generated_ids": generated_ids.detach().cpu(),
        "generated_text": generated_text,
        "fixed_prefix_logits": logits.detach().float().cpu(),
        "generation_metadata": generation_metadata,
        "scoring_metadata": metadata,
    }


def clear_cuda() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> int:
    args = parse_args()
    cfg = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    clean_transformers = Path(cfg["clean_transformers_src"]).expanduser().resolve()
    os.environ["ARMEN_TRANSFORMERS_SRC"] = str(clean_transformers)
    os.environ["OPSD_PRUNING_METHOD"] = "visionzip"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(clean_transformers))
    sys.path.insert(0, str(PROJECT_ROOT))

    import torch
    from opsd.algorithm1.src.metrics import per_token_forward_kl, per_token_js
    from opsd.visionzip_aokvqa.prompting import FormattedAOKVQASample

    if not torch.cuda.is_available():
        raise RuntimeError("One CUDA GPU is required.")
    worker = load_worker()
    rows = worker.read_jsonl(Path(cfg["samples"]))
    row = next((item for item in rows if str(item["sample_id"]) == str(args.sample_id)), None)
    if row is None:
        raise KeyError(f"Unknown sample ID: {args.sample_id}")
    sample = worker.make_sample(row, FormattedAOKVQASample)

    adapter_path = args.adapter.expanduser().resolve()
    merged_path = args.merged_model.expanduser().resolve()
    unmerged_model, unmerged_processor = worker.load_model(cfg, adapter_path)
    unmerged = collect(worker, unmerged_model, unmerged_processor, sample, cfg)
    del unmerged_model
    del unmerged_processor
    clear_cuda()

    merged_model, merged_processor = load_merged(cfg, merged_path)
    merged = collect(
        worker,
        merged_model,
        merged_processor,
        sample,
        cfg,
        fixed_ids_cpu=unmerged["generated_ids"],
    )

    left = unmerged["fixed_prefix_logits"]
    right = merged["fixed_prefix_logits"]
    delta = (left - right).abs()
    forward_kl = per_token_forward_kl(left, right)
    reverse_kl = per_token_forward_kl(right, left)
    js = per_token_js(left, right)
    top1_agreement = (left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean()
    generated_equal = torch.equal(unmerged["generated_ids"], merged["generated_ids"])
    payload = {
        "sample_id": str(args.sample_id),
        "adapter": str(adapter_path),
        "merged_model": str(merged_path),
        "retention_ratio": float(cfg["rollout_retention_ratio"]),
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "unmerged_generated_token_count": int(unmerged["generated_ids"].numel()),
        "merged_generated_token_count": int(merged["generated_ids"].numel()),
        "greedy_generated_ids_exact_match": bool(generated_equal),
        "shared_prefix_token_count": int(left.shape[0]),
        "shared_prefix_top1_agreement": float(top1_agreement),
        "max_abs_logit_delta": float(delta.max()),
        "mean_abs_logit_delta": float(delta.mean()),
        "mean_kl_unmerged_to_merged": float(forward_kl.mean()),
        "mean_kl_merged_to_unmerged": float(reverse_kl.mean()),
        "mean_js": float(js.mean()),
        "unmerged_generated_text": unmerged["generated_text"],
        "merged_generated_text": merged["generated_text"],
        "interpretation": (
            "Exact greedy equivalence on this sample"
            if generated_equal
            else "Greedy outputs differ; inspect distribution metrics before treating load forms as equivalent"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.atomic_write_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    del merged_model
    del merged_processor
    clear_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
