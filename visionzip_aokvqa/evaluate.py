#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opsd.visionzip_aokvqa.aokvqa import load_aokvqa_dataset, normalize_aokvqa_sample
from opsd.visionzip_aokvqa.prompting import parse_final_answer
from opsd.visionzip_aokvqa.qwen_wrapper import (
    apply_lora,
    encode_prompt,
    generate_pruned,
    load_qwen_model_and_processor,
    model_input_subset,
    primary_device,
)


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", default="")
    p.add_argument("--method", required=True)
    p.add_argument("--benchmark", default="AOKVQA")
    p.add_argument("--eval_jsonl", default="")
    p.add_argument("--retention_ratio", type=float, required=True)
    p.add_argument("--output_jsonl", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--allow_embedding_fallback", action="store_true")
    p.add_argument("--full_token", action="store_true")
    p.add_argument("--base_no_lora", action="store_true")
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_eval_samples(cfg: dict[str, Any], args: argparse.Namespace):
    if args.eval_jsonl:
        records = read_jsonl(args.eval_jsonl)
        samples = [normalize_aokvqa_sample(record, idx) for idx, record in enumerate(records)]
        if args.limit:
            samples = samples[: args.limit]
        return samples
    benchmark = str(args.benchmark).lower()
    if benchmark not in {"aokvqa", "aokvqa-smoke"}:
        raise NotImplementedError(
            "Native MME/POPE/GQA/SQA adapters are not bundled in this lightweight runner. "
            "Use VLMEvalKit for those benchmarks or pass a converted multiple-choice JSONL with --eval_jsonl."
        )
    limit = args.limit or int(get_nested(cfg, "evaluation.limit", 0) or 0)
    return load_aokvqa_dataset(
        get_nested(cfg, "dataset.name", "HuggingFaceM4/A-OKVQA"),
        splits=list(get_nested(cfg, "evaluation.aokvqa_splits", ["validation"])),
        limit=limit,
        seed=int(get_nested(cfg, "training.seed", 42)),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config)
    output_jsonl = Path(
        args.output_jsonl
        or OUTPUT_ROOT
        / "eval"
        / args.method
        / f"{args.benchmark.lower()}_r{args.retention_ratio:.1f}.jsonl"
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen_model_and_processor(
        str(get_nested(cfg, "base_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        bf16=bool(get_nested(cfg, "training.bf16", True)),
        attn_implementation=str(get_nested(cfg, "training.attn_implementation", "flash_attention_2")),
        device_map=get_nested(cfg, "training.device_map", "auto"),
    )
    if not args.full_token and not args.base_no_lora:
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required unless --full_token or --base_no_lora is set.")
        model = apply_lora(model, adapter_path=args.checkpoint_path)
    model.eval()
    samples = load_eval_samples(cfg, args)
    correct = 0
    parseable = 0
    total = 0
    latencies = []
    generated_lengths = []
    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            device = primary_device(model)
            prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
            start = time.perf_counter()
            if args.full_token:
                output_ids = model.generate(
                    **model_input_subset(prompt_inputs),
                    max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
                    do_sample=False,
                    use_cache=True,
                )
                prompt_len = int(prompt_inputs["input_ids"].shape[1])
                gen_ids = output_ids[:, prompt_len:]
                prediction = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                meta = {"num_full_visual_tokens": int((prompt_inputs["input_ids"] == model.config.image_token_id).sum().item())}
            else:
                gen_ids, prediction, meta = generate_pruned(
                    model,
                    processor,
                    prompt_inputs,
                    float(args.retention_ratio),
                    max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
                    do_sample=False,
                    allow_embedding_fallback=args.allow_embedding_fallback or bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
                )
            latency = time.perf_counter() - start
            parsed = parse_final_answer(prediction)
            is_correct = parsed == sample.correct_letter
            total += 1
            correct += int(is_correct)
            parseable += int(parsed is not None)
            latencies.append(latency)
            generated_lengths.append(int(gen_ids.numel()))
            row = {
                "sample_id": sample.sample_id,
                "method": args.method,
                "benchmark": args.benchmark,
                "retention_ratio": float(args.retention_ratio),
                "question": sample.question,
                "correct_letter": sample.correct_letter,
                "prediction": prediction,
                "parsed_final_answer": parsed,
                "correct": bool(is_correct),
                "latency_seconds": latency,
                "generated_tokens": int(gen_ids.numel()),
                **{k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))
    metrics = {
        "method": args.method,
        "benchmark": args.benchmark,
        "retention_ratio": float(args.retention_ratio),
        "num_samples": total,
        "accuracy": correct / total if total else None,
        "parseable_rate": parseable / total if total else None,
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "average_generated_tokens": sum(generated_lengths) / len(generated_lengths) if generated_lengths else None,
        "checkpoint_path": args.checkpoint_path,
    }
    with output_jsonl.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
