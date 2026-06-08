#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from opsd.pruning_distill.pruners import build_pruner
from opsd.pruning_distill.qwen25_pruned_forward import (
    build_pruned_inputs_embeds,
    compute_full_position_ids,
    get_qwen25_visual_embeds,
    maybe_disable_adapter,
    validate_single_image_qwen_inputs,
)
from opsd.scripts.train_qwen25vl_prune_distill import (
    bootstrap_qwen25,
    format_question,
    image_path_for,
    import_qwen25_modules,
    messages_for,
    model_input_subset,
    move_inputs,
    parse_keep_ratios,
    primary_device,
    read_jsonl,
    resolve_attn_implementation,
    str_to_bool,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter_path", default="")
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--image_root", default="")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--keep_ratio", type=float, default=0.25)
    p.add_argument(
        "--pruner",
        choices=["random", "grid", "divprune_lite", "vscan_stage1", "existing", "keep_all"],
        default="random",
    )
    p.add_argument("--student_input_mode", choices=["drop_tokens", "mask_fill_debug"], default="drop_tokens")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--skip_full_token_base", type=str_to_bool, default=False)
    p.add_argument("--bf16", type=str_to_bool, default=True)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--divprune_grid_floor", type=str_to_bool, default=False)
    p.add_argument("--divprune_grid_size", type=int, default=4)
    p.add_argument("--divprune_chunk_size", type=int, default=8192)
    p.add_argument("--vscan_grid_size", type=int, default=4)
    p.add_argument("--vscan_score_mode", choices=["cosine_mean", "norm"], default="cosine_mean")
    p.add_argument("--vscan_global_fraction", type=float, default=0.5)
    p.add_argument("--vscan_merge_dropped", type=str_to_bool, default=False)
    return p


def load_eval_model(args: argparse.Namespace):
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

    base_model = None
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
        if not hasattr(model, "disable_adapter"):
            base_model = model_cls.from_pretrained(args.model_name_or_path, **kwargs).eval()
    model.eval()
    if base_model is not None:
        base_model.eval()
    return model, base_model, processor


@contextmanager
def base_model_context(model: Any, separate_base_model: Any | None):
    if separate_base_model is not None:
        with torch.no_grad():
            yield separate_base_model
    else:
        with maybe_disable_adapter(model), torch.no_grad():
            yield model


def encode_prompt(processor: Any, record: dict[str, Any], image_root: str, device: torch.device):
    question = format_question(record)
    image = Image.open(image_path_for(record, image_root)).convert("RGB")
    prompt_messages, add_generation_prompt = messages_for(question, None, add_generation_prompt=True)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = dict(processor(text=[prompt_text], images=[image], return_tensors="pt"))
    inputs = move_inputs(inputs, device)
    validate_single_image_qwen_inputs(inputs)
    return question, inputs


def new_token_ids(output_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    if int(output_ids.shape[1]) > prompt_len:
        return output_ids[:, prompt_len:]
    return output_ids


def decode_new_tokens(processor: Any, output_ids: torch.Tensor, prompt_len: int) -> str:
    new_ids = new_token_ids(output_ids, prompt_len)
    return processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def generate_full(model: Any, processor: Any, prompt_inputs: dict[str, torch.Tensor], max_new_tokens: int) -> tuple[str, float, int]:
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or eos_token_id
    t0 = time.perf_counter()
    output_ids = model.generate(
        **model_input_subset(prompt_inputs),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    return decode_new_tokens(processor, output_ids, prompt_len), time.perf_counter() - t0, int(new_token_ids(output_ids, prompt_len).shape[1])


def generate_pruned(
    model: Any,
    processor: Any,
    prompt_inputs: dict[str, torch.Tensor],
    pruner: Any,
    keep_ratio: float,
    mode: str,
    max_new_tokens: int,
    question: str,
    sample_id: str,
) -> tuple[str, float, int, dict[str, Any]]:
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or eos_token_id
    t0 = time.perf_counter()
    vision_embeds = get_qwen25_visual_embeds(model, prompt_inputs)
    keep_indices = pruner.select(
        vision_embeds,
        prompt_inputs.get("image_grid_thw"),
        keep_ratio,
        question=question,
        metadata={"sample_id": sample_id},
    )
    position_ids = compute_full_position_ids(
        model,
        prompt_inputs["input_ids"],
        prompt_inputs.get("image_grid_thw"),
        prompt_inputs.get("video_grid_thw"),
        prompt_inputs.get("attention_mask"),
        prompt_inputs.get("second_per_grid_ts"),
        prompt_inputs.get("mm_token_type_ids"),
    )
    pruned = build_pruned_inputs_embeds(
        model,
        prompt_inputs["input_ids"],
        prompt_inputs["attention_mask"],
        position_ids,
        vision_embeds,
        keep_indices,
        mode=mode,
        prompt_len=int(prompt_inputs["input_ids"].shape[1]),
        full_mm_token_type_ids=prompt_inputs.get("mm_token_type_ids"),
    )
    gen_kwargs = {
        "input_ids": pruned["input_ids"],
        "inputs_embeds": pruned["inputs_embeds"],
        "attention_mask": pruned["attention_mask"],
        "position_ids": pruned["position_ids"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    if "mm_token_type_ids" in pruned:
        gen_kwargs["mm_token_type_ids"] = pruned["mm_token_type_ids"]
    output_ids = model.generate(**gen_kwargs)
    prompt_len = int(pruned["input_ids"].shape[1])
    prediction = decode_new_tokens(processor, output_ids, prompt_len)
    return prediction, time.perf_counter() - t0, int(new_token_ids(output_ids, prompt_len).shape[1]), pruned["metadata"]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_yes_no(text: str) -> str | None:
    value = normalize_text(text).replace(".", "")
    if value in {"yes", "no"}:
        return value
    if len(value) == 1:
        if value == "y":
            return "yes"
        if value == "n":
            return "no"
    prefix = value[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return None


def first_choice_letter(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", text.strip().upper())
    return match.group(1) if match else None


def score_prediction(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    answer = str(record.get("answer", ""))
    out = {"exact_match": normalize_text(prediction) == normalize_text(answer)}
    answer_yes_no = parse_yes_no(answer)
    if answer_yes_no in {"yes", "no"}:
        pred_yes_no = parse_yes_no(prediction)
        out["yes_no_prediction"] = pred_yes_no or "other"
        out["yes_no_accuracy"] = pred_yes_no == answer_yes_no
    if "choices" in record:
        out["multiple_choice_accuracy"] = first_choice_letter(prediction) == first_choice_letter(answer)
    return out


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source",
        "category",
        "question_id",
        "mme_question_id",
        "mme_pair_id",
        "mme_answer_index",
    )
    return {key: record[key] for key in keys if key in record}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    model, separate_base_model, processor = load_eval_model(args)
    device = primary_device(model)
    records = read_jsonl(args.eval_jsonl)
    if args.start_index and args.start_index > 0:
        records = records[int(args.start_index) :]
    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]
    if args.num_shards > 1:
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard_index must be in [0, --num_shards).")
        records = [record for idx, record in enumerate(records) if idx % args.num_shards == args.shard_index]
    pruner = build_pruner(
        args.pruner,
        seed=args.seed,
        divprune_grid_floor=args.divprune_grid_floor,
        divprune_grid_size=args.divprune_grid_size,
        divprune_chunk_size=args.divprune_chunk_size,
        vscan_grid_size=args.vscan_grid_size,
        vscan_score_mode=args.vscan_score_mode,
        vscan_global_fraction=args.vscan_global_fraction,
        vscan_merge_dropped=args.vscan_merge_dropped,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            sample_id = str(record.get("sample_id", record.get("id", idx)))
            question, prompt_inputs = encode_prompt(processor, record, args.image_root, device)
            gold = str(record.get("answer", ""))

            with base_model_context(model, separate_base_model) as base:
                if not args.skip_full_token_base:
                    full_pred, full_latency, full_generated_tokens = generate_full(base, processor, prompt_inputs, args.max_new_tokens)
                    row = {
                        "sample_id": sample_id,
                        "mode": "full_token_base",
                        "eval_mode": "full_token_base",
                        "pruner": "keep_all",
                        "question": question,
                        "gold_answer": gold,
                        "prediction": full_pred,
                        "keep_ratio": 1.0,
                        "num_full_visual_tokens": int((prompt_inputs["input_ids"] == base.config.image_token_id).sum().item()),
                        "num_kept_visual_tokens": int((prompt_inputs["input_ids"] == base.config.image_token_id).sum().item()),
                        "latency_seconds": full_latency,
                        "generated_tokens": full_generated_tokens,
                        **record_metadata(record),
                        **score_prediction(record, full_pred),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                pruned_base_pred, pruned_base_latency, pruned_base_generated_tokens, pruned_base_meta = generate_pruned(
                    base,
                    processor,
                    prompt_inputs,
                    pruner,
                    args.keep_ratio,
                    args.student_input_mode,
                    args.max_new_tokens,
                    question,
                    sample_id,
                )
                row = {
                    "sample_id": sample_id,
                    "mode": "pruned_token_base_no_lora",
                    "eval_mode": "pruned_token_base_no_lora",
                    "pruner": args.pruner,
                    "question": question,
                    "gold_answer": gold,
                    "prediction": pruned_base_pred,
                    "keep_ratio": args.keep_ratio,
                    "num_full_visual_tokens": int(pruned_base_meta["num_full_visual_tokens"]),
                    "num_kept_visual_tokens": int(pruned_base_meta["num_kept_visual_tokens"]),
                    "latency_seconds": pruned_base_latency,
                    "generated_tokens": pruned_base_generated_tokens,
                    **record_metadata(record),
                    **score_prediction(record, pruned_base_pred),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            with torch.no_grad():
                student_pred, student_latency, student_generated_tokens, student_meta = generate_pruned(
                    model,
                    processor,
                    prompt_inputs,
                    pruner,
                    args.keep_ratio,
                    args.student_input_mode,
                    args.max_new_tokens,
                    question,
                    sample_id,
                )
            row = {
                "sample_id": sample_id,
                "mode": "pruned_token_distilled_student",
                "eval_mode": "pruned_token_distilled_student",
                "pruner": args.pruner,
                "question": question,
                "gold_answer": gold,
                "prediction": student_pred,
                "keep_ratio": args.keep_ratio,
                "num_full_visual_tokens": int(student_meta["num_full_visual_tokens"]),
                "num_kept_visual_tokens": int(student_meta["num_kept_visual_tokens"]),
                "latency_seconds": student_latency,
                "generated_tokens": student_generated_tokens,
                **record_metadata(record),
                **score_prediction(record, student_pred),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"sample_id": sample_id, "student_prediction": student_pred}, ensure_ascii=False))
            f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
