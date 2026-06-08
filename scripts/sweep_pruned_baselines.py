#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opsd.pruning_distill.pruners import build_pruner
from opsd.scripts.eval_qwen25vl_pruned_student import (
    encode_prompt,
    generate_full,
    generate_pruned,
    score_prediction,
)
from opsd.scripts.train_qwen25vl_prune_distill import (
    import_qwen25_modules,
    load_one_model,
    model_input_subset,
    parse_keep_ratios,
    primary_device,
    read_jsonl,
    resolve_attn_implementation,
    str_to_bool,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--image_root", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pruners", default="random,grid,divprune_lite,vscan_stage1")
    p.add_argument("--keep_ratios", default="1.0,0.5,0.25,0.125")
    p.add_argument("--student_input_mode", choices=["drop_tokens", "mask_fill_debug"], default="drop_tokens")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", type=str_to_bool, default=True)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--divprune_grid_floor", type=str_to_bool, default=False)
    p.add_argument("--divprune_grid_size", type=int, default=4)
    p.add_argument("--divprune_chunk_size", type=int, default=8192)
    p.add_argument("--vscan_grid_size", type=int, default=4)
    p.add_argument("--vscan_score_mode", choices=["cosine_mean", "norm"], default="cosine_mean")
    p.add_argument("--vscan_global_fraction", type=float, default=0.5)
    return p


def parse_names(value: str) -> list[str]:
    names = [x.strip() for x in value.replace(",", " ").split() if x.strip()]
    if not names:
        raise ValueError("At least one pruner is required.")
    return names


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    metrics: dict[str, Any] = {
        "num_samples": len(rows),
        "num_failures": len(failures),
        "average_latency": mean([float(r.get("latency_seconds", 0.0)) for r in ok]) if ok else None,
        "average_generated_tokens": mean([float(r.get("generated_tokens", 0.0)) for r in ok]) if ok else None,
        "average_full_visual_tokens": mean([float(r.get("num_full_visual_tokens", 0.0)) for r in ok]) if ok else None,
        "average_kept_visual_tokens": mean([float(r.get("num_kept_visual_tokens", 0.0)) for r in ok]) if ok else None,
        "average_KL_to_full_teacher": None,
    }
    exact = [r.get("exact_match") for r in ok if "exact_match" in r]
    if exact:
        metrics["exact_match"] = sum(bool(x) for x in exact) / len(exact)
    mc = [r.get("multiple_choice_accuracy") for r in ok if "multiple_choice_accuracy" in r]
    if mc:
        metrics["multiple_choice_accuracy"] = sum(bool(x) for x in mc) / len(mc)
    return metrics


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    AutoProcessor, model_cls = import_qwen25_modules()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    attn_impl = resolve_attn_implementation(args.attn_implementation)
    model = load_one_model(model_cls, args.model_name_or_path, dtype, attn_impl, args.device_map).eval()
    device = primary_device(model)

    records = read_jsonl(args.eval_jsonl)
    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]
    pruner_names = parse_names(args.pruners)
    keep_ratios = parse_keep_ratios(args.keep_ratios)

    summary_rows: list[dict[str, Any]] = []
    all_full_rows: list[dict[str, Any]] = []
    prompt_cache: list[tuple[dict[str, Any], str, Any]] = []

    full_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        sample_id = str(record.get("sample_id", record.get("id", idx)))
        try:
            question, prompt_inputs = encode_prompt(processor, record, args.image_root, device)
            prompt_cache.append((record, question, prompt_inputs))
            with torch.no_grad():
                prediction, latency, generated_tokens = generate_full(model, processor, prompt_inputs, args.max_new_tokens)
            image_token_count = int((prompt_inputs["input_ids"] == model.config.image_token_id).sum().item())
            row = {
                "sample_id": sample_id,
                "method": "full_token_base",
                "pruner": "keep_all",
                "keep_ratio": 1.0,
                "question": question,
                "gold_answer": str(record.get("answer", "")),
                "prediction": prediction,
                "num_full_visual_tokens": image_token_count,
                "num_kept_visual_tokens": image_token_count,
                "latency_seconds": latency,
                "generated_tokens": generated_tokens,
                **score_prediction(record, prediction),
            }
        except Exception as exc:
            row = {"sample_id": sample_id, "method": "full_token_base", "pruner": "keep_all", "keep_ratio": 1.0, "error": repr(exc)}
        full_rows.append(row)
    write_jsonl(out_dir / "predictions_full_token_base.jsonl", full_rows)
    full_metrics = summarize(full_rows)
    (out_dir / "metrics_full_token_base.json").write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    summary_rows.append({"method": "full_token_base", "pruner": "keep_all", "keep_ratio": 1.0, **full_metrics})
    all_full_rows.extend(full_rows)

    for pruner_name in pruner_names:
        for keep_ratio in keep_ratios:
            pruner = build_pruner(
                pruner_name,
                seed=args.seed,
                divprune_grid_floor=args.divprune_grid_floor,
                divprune_grid_size=args.divprune_grid_size,
                divprune_chunk_size=args.divprune_chunk_size,
                vscan_grid_size=args.vscan_grid_size,
                vscan_score_mode=args.vscan_score_mode,
                vscan_global_fraction=args.vscan_global_fraction,
                vscan_merge_dropped=False,
            )
            rows = []
            method = f"pruned_token_base_no_lora_{pruner_name}"
            for idx, item in enumerate(prompt_cache):
                record, question, prompt_inputs = item
                sample_id = str(record.get("sample_id", record.get("id", idx)))
                try:
                    with torch.no_grad():
                        prediction, latency, generated_tokens, meta = generate_pruned(
                            model,
                            processor,
                            prompt_inputs,
                            pruner,
                            float(keep_ratio),
                            args.student_input_mode,
                            args.max_new_tokens,
                            question,
                            sample_id,
                        )
                    row = {
                        "sample_id": sample_id,
                        "method": method,
                        "pruner": pruner_name,
                        "keep_ratio": float(keep_ratio),
                        "question": question,
                        "gold_answer": str(record.get("answer", "")),
                        "prediction": prediction,
                        "num_full_visual_tokens": int(meta["num_full_visual_tokens"]),
                        "num_kept_visual_tokens": int(meta["num_kept_visual_tokens"]),
                        "latency_seconds": latency,
                        "generated_tokens": generated_tokens,
                        **score_prediction(record, prediction),
                    }
                except Exception as exc:
                    row = {
                        "sample_id": sample_id,
                        "method": method,
                        "pruner": pruner_name,
                        "keep_ratio": float(keep_ratio),
                        "question": question,
                        "gold_answer": str(record.get("answer", "")),
                        "prediction": "",
                        "error": repr(exc),
                    }
                rows.append(row)
            suffix = f"{pruner_name}_r{str(keep_ratio).replace('.', 'p')}"
            write_jsonl(out_dir / f"predictions_{suffix}.jsonl", rows)
            metrics = summarize(rows)
            (out_dir / f"metrics_{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            summary_rows.append({"method": method, "pruner": pruner_name, "keep_ratio": float(keep_ratio), **metrics})

    summary_path = out_dir / "summary.csv"
    keys = sorted({key for row in summary_rows for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"summary_csv": str(summary_path), "num_configs": len(summary_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
