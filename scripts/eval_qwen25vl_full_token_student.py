#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opsd.scripts.eval_qwen25vl_pruned_student import (
    encode_prompt,
    generate_full,
    record_metadata,
    score_prediction,
)
from opsd.scripts.train_qwen25vl_prune_distill import (
    bootstrap_qwen25,
    import_qwen25_modules,
    primary_device,
    read_jsonl,
    resolve_attn_implementation,
    str_to_bool,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--image_root", default="")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bf16", type=str_to_bool, default=True)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--device_map", default="auto")
    return p


def load_student(args: argparse.Namespace) -> tuple[Any, Any]:
    bootstrap_qwen25()
    AutoProcessor, model_cls = import_qwen25_modules()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    attn_impl = resolve_attn_implementation(args.attn_implementation)
    kwargs = {"torch_dtype": dtype, "attn_implementation": attn_impl}
    if args.device_map and args.device_map.lower() != "none" and torch.cuda.is_available():
        kwargs["device_map"] = args.device_map
    try:
        model = model_cls.from_pretrained(args.model_name_or_path, **kwargs)
    except Exception:
        if attn_impl == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
            model = model_cls.from_pretrained(args.model_name_or_path, **kwargs)
        else:
            raise
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, processor


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model, processor = load_student(args)
    device = primary_device(model)
    records = read_jsonl(args.eval_jsonl)
    if args.start_index and args.start_index > 0:
        records = records[int(args.start_index) :]
    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            sample_id = str(record.get("sample_id", record.get("id", idx)))
            question, prompt_inputs = encode_prompt(processor, record, args.image_root, device)
            with torch.no_grad():
                prediction, latency, generated_tokens = generate_full(model, processor, prompt_inputs, args.max_new_tokens)
            image_tokens = int((prompt_inputs["input_ids"] == model.config.image_token_id).sum().item())
            row = {
                "sample_id": sample_id,
                "mode": "full_token_distilled_student",
                "eval_mode": "full_token_distilled_student",
                "pruner": "keep_all",
                "question": question,
                "gold_answer": str(record.get("answer", "")),
                "prediction": prediction,
                "keep_ratio": 1.0,
                "num_full_visual_tokens": image_tokens,
                "num_kept_visual_tokens": image_tokens,
                "latency_seconds": latency,
                "generated_tokens": generated_tokens,
                **record_metadata(record),
                **score_prediction(record, prediction),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"sample_id": sample_id, "prediction": prediction}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
