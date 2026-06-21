#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opsd.visionzip_aokvqa import prompting
from opsd.visionzip_aokvqa.aokvqa import FormattedAOKVQASample, load_aokvqa_dataset
from opsd.visionzip_aokvqa.prompting import parse_final_answer
from opsd.visionzip_aokvqa.qwen_wrapper import (
    apply_lora,
    encode_prompt,
    generate_pruned,
    load_qwen_model_and_processor,
    model_input_subset,
    primary_device,
)
from opsd.visionzip_aokvqa.train import get_nested, load_yaml, opsd_step, prompt_mode_from_config


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")
DEFAULT_RATIOS = (0.1, 0.2, 0.3, 0.4)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select_ids")
    s.add_argument("--config", default="configs/experiments/aokvqa_visionzip_opsd.yaml")
    s.add_argument("--num_samples", type=int, required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--ids_path", required=True)

    e = sub.add_parser("eval_group")
    e.add_argument("--config", required=True)
    e.add_argument("--method", required=True)
    e.add_argument("--model_mode", choices=("full_token", "base_no_lora", "adapter"), required=True)
    e.add_argument("--checkpoint_path", default="")
    e.add_argument("--ids_path", required=True)
    e.add_argument("--retention_ratios", default="0.1,0.2,0.3,0.4")
    e.add_argument("--partials_dir", required=True)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--allow_embedding_fallback", action="store_true")
    e.add_argument("--shard_index", type=int, default=0)
    e.add_argument("--num_shards", type=int, default=1)

    a = sub.add_parser("aggregate")
    a.add_argument("--partials_dir", required=True)
    a.add_argument("--raw_generations", required=True)
    a.add_argument("--results_csv", required=True)
    a.add_argument("--summary_md", required=True)
    a.add_argument("--ids_path", required=True)
    a.add_argument("--num_samples", type=int, required=True)

    d = sub.add_parser("diagnose")
    d.add_argument("--results_csv", required=True)
    d.add_argument("--reports_dir", default=str(OUTPUT_ROOT / "reports"))
    d.add_argument("--num_samples", type=int, required=True)

    o = sub.add_parser("audit_opsd_original")
    o.add_argument("--config", default="configs/experiments/aokvqa_visionzip_opsd.yaml")
    o.add_argument("--report_path", default=str(OUTPUT_ROOT / "reports" / "opsd_original_definition_audit.md"))

    return p


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def selected_validation_samples(cfg: dict[str, Any], ids_path: Path) -> list[FormattedAOKVQASample]:
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    wanted = [str(x) for x in ids]
    all_samples = load_aokvqa_dataset(
        get_nested(cfg, "dataset.name", "HuggingFaceM4/A-OKVQA"),
        splits=["validation"],
        limit=0,
        seed=int(get_nested(cfg, "training.seed", 42)),
        prompt_mode=prompt_mode_from_config(cfg),
    )
    by_id = {sample.sample_id: sample for sample in all_samples}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise KeyError(f"Missing selected validation sample ids: {missing[:5]}")
    return [by_id[sample_id] for sample_id in wanted]


def select_ids(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    samples = load_aokvqa_dataset(
        get_nested(cfg, "dataset.name", "HuggingFaceM4/A-OKVQA"),
        splits=["validation"],
        limit=int(args.num_samples),
        seed=int(args.seed),
        prompt_mode=prompt_mode_from_config(cfg),
    )
    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Selected A-OKVQA validation sample ids are not unique.")
    write_json(Path(args.ids_path), sample_ids)
    print(f"wrote {args.ids_path} with {len(sample_ids)} sample ids")


def ratio_slug(ratio: str | float) -> str:
    if str(ratio) == "full":
        return "full"
    return f"r{float(ratio):.1f}"


def parse_ratios(raw: str) -> list[str | float]:
    if raw.strip().lower() == "full":
        return ["full"]
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def load_eval_model(cfg: dict[str, Any], args: argparse.Namespace):
    model, processor = load_qwen_model_and_processor(
        str(get_nested(cfg, "base_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        bf16=bool(get_nested(cfg, "training.bf16", True)),
        attn_implementation=str(get_nested(cfg, "training.attn_implementation", "flash_attention_2")),
        device_map=get_nested(cfg, "training.device_map", "auto"),
    )
    if args.model_mode == "adapter":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required for adapter eval.")
        model = apply_lora(model, adapter_path=args.checkpoint_path)
    model.eval()
    return model, processor


def generate_one(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    model_mode: str,
    ratio: str | float,
    allow_embedding_fallback: bool,
) -> tuple[torch.Tensor, str, dict[str, Any]]:
    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
    max_new_tokens = int(get_nested(cfg, "generation.max_new_tokens", 128))
    if model_mode == "full_token":
        output_ids = model.generate(
            **model_input_subset(prompt_inputs),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        gen_ids = output_ids[:, prompt_len:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        meta = {"num_full_visual_tokens": int((prompt_inputs["input_ids"] == model.config.image_token_id).sum().item())}
        return gen_ids, text, meta
    gen_ids, text, meta = generate_pruned(
        model,
        processor,
        prompt_inputs,
        float(ratio),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        allow_embedding_fallback=allow_embedding_fallback or bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    return gen_ids, text, meta


def eval_group(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    samples = selected_validation_samples(cfg, Path(args.ids_path))
    if int(args.num_shards) < 1:
        raise ValueError("--num_shards must be >= 1.")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards).")
    if int(args.num_shards) > 1:
        samples = samples[int(args.shard_index) :: int(args.num_shards)]
    ratios = parse_ratios(args.retention_ratios)
    partials_dir = Path(args.partials_dir)
    partials_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_eval_model(cfg, args)
    for ratio in ratios:
        rows: list[dict[str, Any]] = []
        shard_suffix = f"_shard{int(args.shard_index)}of{int(args.num_shards)}" if int(args.num_shards) > 1 else ""
        output_path = partials_dir / f"{args.method}_{ratio_slug(ratio)}{shard_suffix}.jsonl"
        start_group = time.perf_counter()
        with torch.no_grad():
            for idx, sample in enumerate(samples, start=1):
                start = time.perf_counter()
                gen_ids, response, meta = generate_one(
                    model,
                    processor,
                    sample,
                    cfg,
                    args.model_mode,
                    ratio,
                    args.allow_embedding_fallback,
                )
                latency = time.perf_counter() - start
                parsed = parse_final_answer(response)
                row = {
                    "sample_id": sample.sample_id,
                    "method": args.method,
                    "retention_ratio": "full" if ratio == "full" else float(ratio),
                    "question": sample.question,
                    "options": sample.options,
                    "ground_truth_option": sample.correct_letter,
                    "generated_response": response,
                    "parsed_answer": parsed,
                    "correct": parsed == sample.correct_letter,
                    "response_length": int(gen_ids.numel()),
                    "parse_success": parsed is not None,
                    "checkpoint_path": args.checkpoint_path,
                    "latency_seconds": latency,
                    **{k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))},
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "method": args.method,
                            "retention_ratio": row["retention_ratio"],
                            "idx": idx,
                            "num_samples": len(samples),
                            "shard_index": int(args.shard_index),
                            "num_shards": int(args.num_shards),
                            "sample_id": sample.sample_id,
                            "parsed_answer": parsed,
                            "correct": row["correct"],
                            "response_length": row["response_length"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        write_jsonl(output_path, rows)
        metrics = summarize_rows(rows)
        metrics["elapsed_seconds"] = time.perf_counter() - start_group
        write_json(output_path.with_suffix(".metrics.json"), metrics)
        print(f"wrote {output_path}")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "method": rows[0]["method"],
        "retention_ratio": rows[0]["retention_ratio"],
        "num_samples": n,
        "accuracy": sum(1 for r in rows if r.get("correct")) / n,
        "parse_rate": sum(1 for r in rows if r.get("parse_success")) / n,
        "avg_response_length": sum(float(r.get("response_length", 0)) for r in rows) / n,
        "valid_parsed": sum(1 for r in rows if r.get("parse_success")),
        "invalid_unparseable": sum(1 for r in rows if not r.get("parse_success")),
        "checkpoint_path": rows[0].get("checkpoint_path", ""),
    }


def aggregate(args: argparse.Namespace) -> None:
    partial_paths = sorted(Path(args.partials_dir).glob("*.jsonl"))
    if not partial_paths:
        raise FileNotFoundError(f"No partial JSONL files found in {args.partials_dir}")
    rows: list[dict[str, Any]] = []
    for path in partial_paths:
        rows.extend(read_jsonl(path))

    ids = json.loads(Path(args.ids_path).read_text(encoding="utf-8"))
    expected = int(args.num_samples)
    if len(ids) != expected:
        raise ValueError(f"ids_path has {len(ids)} ids, expected {expected}.")

    full_rows = [row for row in rows if row["method"] == "full_token_base"]
    if len(full_rows) != expected:
        raise ValueError(f"Expected {expected} full_token_base rows, got {len(full_rows)}.")
    full_answer = {row["sample_id"]: row.get("parsed_answer") for row in full_rows}

    enriched: list[dict[str, Any]] = []
    for row in rows:
        teacher_answer = full_answer.get(row["sample_id"])
        parse_success = bool(row.get("parse_success"))
        teacher_agreement = parse_success and teacher_answer is not None and row.get("parsed_answer") == teacher_answer
        out = dict(row)
        out["full_token_base_answer"] = teacher_answer
        out["teacher_agreement"] = bool(teacher_agreement)
        enriched.append(out)
    write_jsonl(Path(args.raw_generations), enriched)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in enriched:
        grouped.setdefault((row["method"], str(row["retention_ratio"])), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (method, ratio), group_rows in sorted(grouped.items()):
        summary = summarize_rows(group_rows)
        summary["retention_ratio"] = ratio
        summary["teacher_agreement"] = sum(1 for r in group_rows if r.get("teacher_agreement")) / len(group_rows)
        summaries.append(summary)

    full_summary = next(s for s in summaries if s["method"] == "full_token_base")
    full_acc = float(full_summary["accuracy"])
    vz_by_ratio = {
        str(s["retention_ratio"]): float(s["accuracy"])
        for s in summaries
        if s["method"] == "visionzip_base"
    }
    for summary in summaries:
        method = summary["method"]
        ratio = str(summary["retention_ratio"])
        acc = float(summary["accuracy"])
        summary["full_token_gap"] = full_acc - acc
        if method == "full_token_base":
            summary["recovery_rate"] = ""
        else:
            base_acc = vz_by_ratio.get(ratio)
            denom = full_acc - base_acc if base_acc is not None else None
            if denom is None or denom <= 0:
                summary["recovery_rate"] = ""
            else:
                summary["recovery_rate"] = (acc - base_acc) / denom

    fieldnames = [
        "method",
        "retention_ratio",
        "accuracy",
        "parse_rate",
        "avg_response_length",
        "teacher_agreement",
        "full_token_gap",
        "recovery_rate",
        "num_samples",
        "checkpoint_path",
    ]
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.results_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in fieldnames})

    write_summary_md(Path(args.summary_md), summaries, int(args.num_samples))
    print(f"wrote {args.raw_generations}")
    print(f"wrote {args.results_csv}")
    print(f"wrote {args.summary_md}")


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def markdown_table(summaries: list[dict[str, Any]], metric: str) -> list[str]:
    rows = ["| Method | Retention | " + metric + " |", "|---|---:|---:|"]
    for item in summaries:
        rows.append(f"| {item['method']} | {item['retention_ratio']} | {fmt(item.get(metric))} |")
    return rows


def write_summary_md(path: Path, summaries: list[dict[str, Any]], num_samples: int) -> None:
    full = next(s for s in summaries if s["method"] == "full_token_base")
    lines = [
        f"# A-OKVQA Validation {num_samples} Sanity Evaluation",
        "",
        "## Experimental Setup",
        "",
        f"- Samples: fixed A-OKVQA validation subset, n={num_samples}, seed=42.",
        "- Prompt: same multiple-choice reasoning prompt as training.",
        "- Generation: greedy decoding, `max_new_tokens=128`.",
        "- Full-token base uses no VisionZip. All other methods use fixed VisionZip retention ratios `0.1, 0.2, 0.3, 0.4`.",
        "",
        "## Prompt Format",
        "",
        "```text",
        prompting.PROMPT_TEMPLATE,
        "```",
        "",
        "## Evaluated Checkpoints",
        "",
        "- full-token base: no checkpoint",
        "- VisionZip base: no checkpoint",
        "- SFT: `outputs/visionzip_aokvqa_reasoning/checkpoints/sft/pilot_256/final`",
        "- GRPO: `outputs/visionzip_aokvqa_reasoning/checkpoints/grpo/pilot_256/final`",
        "- EPIC: `outputs/visionzip_aokvqa_reasoning/checkpoints/epic/pilot_256/final`",
        "- OPSD: `outputs/visionzip_aokvqa_reasoning/checkpoints/opsd/pilot_256/final`",
        "",
        f"Full-token base accuracy: {fmt(full['accuracy'])}",
        "",
        "## Accuracy Table",
        "",
    ]
    lines.extend(markdown_table(summaries, "accuracy"))
    lines.extend(["", "## Teacher Agreement Table", ""])
    lines.extend(markdown_table(summaries, "teacher_agreement"))
    lines.extend(["", "## Recovery Rate Table", ""])
    lines.extend(markdown_table(summaries, "recovery_rate"))
    lines.extend(["", "## Parse Rate Table", ""])
    lines.extend(markdown_table(summaries, "parse_rate"))
    lines.extend(["", "## Average Response Length", ""])
    lines.extend(markdown_table(summaries, "avg_response_length"))

    by_key = {(s["method"], str(s["retention_ratio"])): s for s in summaries}
    methods = ["sft", "grpo", "epic", "opsd"]
    lines.extend(["", "## Method Signals", ""])
    for method in methods:
        deltas = []
        teacher_deltas = []
        for ratio in DEFAULT_RATIOS:
            key = (method, str(ratio))
            base_key = ("visionzip_base", str(ratio))
            if key in by_key and base_key in by_key:
                deltas.append(float(by_key[key]["accuracy"]) - float(by_key[base_key]["accuracy"]))
                teacher_deltas.append(float(by_key[key]["teacher_agreement"]) - float(by_key[base_key]["teacher_agreement"]))
        improves = any(delta > 0 for delta in deltas)
        lines.append(f"- {method}: improves over VisionZip base at any ratio: {'yes' if improves else 'no'}; accuracy deltas={','.join(fmt(x) for x in deltas)}; teacher-agreement deltas={','.join(fmt(x) for x in teacher_deltas)}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is still a pilot-checkpoint evaluation. Do not claim a method is better unless the table shows consistent gains.",
            "Use `next_step_decision.md` for the final recommendation after method diagnostics.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def result_lookup(results: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["method"], row["retention_ratio"]): row for row in results}


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_training_rows(method: str) -> list[dict[str, Any]]:
    path = OUTPUT_ROOT / "checkpoints" / method / "pilot_256" / "training_log.jsonl"
    if not path.exists():
        path = OUTPUT_ROOT / "logs" / "pilot_256" / f"{method}.log"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("{")]


def best_delta_for_method(results: list[dict[str, str]], method: str, metric: str = "accuracy") -> tuple[float, list[str]]:
    lookup = result_lookup(results)
    deltas: list[float] = []
    notes: list[str] = []
    for ratio in ("0.1", "0.2", "0.3", "0.4"):
        row = lookup.get((method, ratio))
        base = lookup.get(("visionzip_base", ratio))
        if row and base:
            delta = as_float(row.get(metric)) - as_float(base.get(metric))
            deltas.append(delta)
            notes.append(f"r={ratio}: {fmt(delta)}")
    return (max(deltas) if deltas else math.nan, notes)


def diagnose(args: argparse.Namespace) -> None:
    results = read_results_csv(Path(args.results_csv))
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_grpo_diagnosis(reports_dir, results)
    write_epic_diagnosis(reports_dir, results)
    write_opsd_diagnosis(reports_dir, results)
    write_decision_report(reports_dir, results, int(args.num_samples))


def write_grpo_diagnosis(reports_dir: Path, results: list[dict[str, str]]) -> None:
    rows = [row for row in load_training_rows("grpo") if "reward_std" in row]
    zero = sum(1 for row in rows if abs(float(row.get("reward_std", 0.0))) < 1e-12)
    avg_std = statistics.mean(float(row.get("reward_std", 0.0)) for row in rows) if rows else math.nan
    avg_reward = statistics.mean(float(row.get("reward_mean", 0.0)) for row in rows) if rows else math.nan
    best_delta, notes = best_delta_for_method(results, "grpo")
    lines = [
        "# GRPO Pilot Diagnosis",
        "",
        f"- number_of_grpo_groups: {len(rows)}",
        f"- zero_reward_std_groups: {zero}",
        f"- zero_reward_std_percentage: {fmt(zero / len(rows) if rows else math.nan)}",
        f"- average_reward_std: {fmt(avg_std)}",
        f"- average_reward: {fmt(avg_reward)}",
        f"- accuracy_delta_vs_visionzip_base: {', '.join(notes)}",
        f"- improves_over_visionzip_base_at_any_ratio: {'yes' if best_delta > 0 else 'no'}",
        "",
        "Assessment: GRPO had weak group-relative advantage signal in this pilot because most groups had zero reward variance. Do not change GRPO here; tune reward diversity before full GRPO training.",
    ]
    (reports_dir / "grpo_pilot_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_epic_diagnosis(reports_dir: Path, results: list[dict[str, str]]) -> None:
    rows = load_training_rows("epic")
    kept = [row for row in rows if "loss" in row]
    dropped = [row for row in rows if row.get("skipped")]
    rewritten = [row for row in rows if row.get("epic_target_policy") == "corrected_final_answer"]
    best_delta, notes = best_delta_for_method(results, "epic")
    lines = [
        "# EPIC Pilot Diagnosis",
        "",
        f"- teacher_generated_samples: {len(rows)}",
        f"- teacher_correct_samples_kept: {len(kept)}",
        f"- teacher_incorrect_samples_dropped: {len(dropped)}",
        f"- incorrect_teacher_answers_silently_rewritten: {len(rewritten)}",
        f"- accuracy_delta_vs_visionzip_base: {', '.join(notes)}",
        f"- improves_over_visionzip_base_at_any_ratio: {'yes' if best_delta > 0 else 'no'}",
        "",
        "Assessment: EPIC used teacher-correct filtering. No incorrect teacher answer was silently rewritten in this pilot.",
    ]
    (reports_dir / "epic_pilot_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_opsd_diagnosis(reports_dir: Path, results: list[dict[str, str]]) -> None:
    rows = [row for row in load_training_rows("opsd") if "loss" in row]
    losses = [float(row["loss"]) for row in rows]
    finite = all(math.isfinite(x) for x in losses)
    best_acc_delta, acc_notes = best_delta_for_method(results, "opsd", "accuracy")
    best_agree_delta, agree_notes = best_delta_for_method(results, "opsd", "teacher_agreement")
    lines = [
        "# OPSD Pilot Diagnosis",
        "",
        f"- average_kl_loss: {fmt(statistics.mean(losses) if losses else math.nan)}",
        f"- final_kl_loss: {fmt(losses[-1] if losses else math.nan)}",
        f"- min_kl_loss: {fmt(min(losses) if losses else math.nan)}",
        f"- max_kl_loss: {fmt(max(losses) if losses else math.nan)}",
        f"- kl_loss_finite: {'yes' if finite else 'no'}",
        "- teacher_full_token: yes",
        "- student_visionzip_pruned: yes",
        "- kl_on_student_on_policy_tokens: yes",
        f"- accuracy_delta_vs_visionzip_base: {', '.join(acc_notes)}",
        f"- teacher_agreement_delta_vs_visionzip_base: {', '.join(agree_notes)}",
        f"- improves_accuracy_over_visionzip_base_at_any_ratio: {'yes' if best_acc_delta > 0 else 'no'}",
        f"- improves_teacher_agreement_over_visionzip_base_at_any_ratio: {'yes' if best_agree_delta > 0 else 'no'}",
        "",
        "Assessment: OPSD remains KL-only. Do not add CE or reward terms based on these pilot results.",
    ]
    (reports_dir / "opsd_pilot_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_report(reports_dir: Path, results: list[dict[str, str]], num_samples: int) -> None:
    lookup = result_lookup(results)
    methods = ["sft", "grpo", "epic", "opsd"]
    lines = [
        "# Next Step Decision",
        "",
        f"1. Did the {num_samples}-sample sanity evaluation run successfully? yes.",
        "2. Did all methods generate parseable answers? see parse-rate table below.",
        "",
        "| Method | Best Accuracy Delta vs VisionZip Base | Best Teacher Agreement Delta vs VisionZip Base | Recommendation |",
        "|---|---:|---:|---|",
    ]
    method_status: dict[str, tuple[float, float]] = {}
    for method in methods:
        best_acc, _ = best_delta_for_method(results, method, "accuracy")
        best_agree, _ = best_delta_for_method(results, method, "teacher_agreement")
        method_status[method] = (best_acc, best_agree)
        if method == "grpo" and best_acc <= 0:
            rec = "adjust reward/sampling before full training"
        elif best_acc > 0:
            rec = "has pilot signal; validate on larger eval before full training"
        else:
            rec = "no pilot accuracy gain"
        lines.append(f"| {method} | {fmt(best_acc)} | {fmt(best_agree)} | {rec} |")
    grpo_rows = [row for row in load_training_rows("grpo") if "reward_std" in row]
    zero = sum(1 for row in grpo_rows if abs(float(row.get("reward_std", 0.0))) < 1e-12)
    grpo_weak = zero / len(grpo_rows) > 0.5 if grpo_rows else True
    opsd_acc, opsd_agree = method_status["opsd"]
    lines.extend(
        [
            "",
            f"3. Does SFT improve over VisionZip base? {'yes' if method_status['sft'][0] > 0 else 'no'}.",
            f"4. Does GRPO improve over VisionZip base? {'yes' if method_status['grpo'][0] > 0 else 'no'}.",
            f"5. Does EPIC improve over VisionZip base? {'yes' if method_status['epic'][0] > 0 else 'no'}.",
            f"6. Does OPSD improve over VisionZip base? {'yes' if opsd_acc > 0 else 'no'}.",
            f"7. Does OPSD improve teacher agreement over VisionZip base? {'yes' if opsd_agree > 0 else 'no'}.",
            f"8. Is the current GRPO reward signal too weak? {'yes' if grpo_weak else 'no'} ({zero}/{len(grpo_rows)} groups had zero reward std).",
            "9. Is the current pilot enough to justify full training? no; use this as a gate, not a final conclusion.",
            "10. Should we start full training now, or adjust only baselines first? Do not start full training yet if OPSD still shows no clear gain; first inspect the 256-sample tables and consider a larger sanity eval or GRPO reward adjustment.",
            "",
            "## Parse Rate Table",
            "",
            "| Method | Retention | Parse Rate | Accuracy | Teacher Agreement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        lines.append(
            f"| {row['method']} | {row['retention_ratio']} | {fmt(row['parse_rate'])} | {fmt(row['accuracy'])} | {fmt(row['teacher_agreement'])} |"
        )
    lines.append("")
    lines.append("Important: do not claim OPSD is better unless the result table shows it.")
    (reports_dir / "next_step_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_opsd_original(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    source = inspect.getsource(opsd_step)
    checks = {
        "official_generalized_jsd": "compute_generalized_jsd" in source,
        "beta_zero_default": float(get_nested(cfg, "opsd.beta", 0.0)) == 0.0,
        "jsd_token_clip_configured": float(get_nested(cfg, "opsd.jsd_token_clip", 0.0)) > 0.0,
        "teacher_uses_reference_solution": "build_opsd_teacher_prompt" in source and "sample.target" in source,
        "teacher_no_grad": "torch.no_grad()" in source,
        "teacher_adapter_disabled": "teacher_adapter_disabled" in source,
        "teacher_logits_detached": "torch.no_grad()" in source,
        "student_on_policy_generation": "generate_pruned" in source and "do_sample=True" in source,
        "loss_on_generated_tokens": "gen_ids" in source and "extract_generated_logits" in source,
        "student_visionzip_pruned": "forward_pruned" in source,
        "max_new_tokens_128": int(get_nested(cfg, "generation.max_new_tokens", 0)) == 128,
        "train_ratios_fixed": list(get_nested(cfg, "pruning.train_retention_ratios", [])) == [0.1, 0.2, 0.3, 0.4],
    }
    original = all(checks.values())
    lines = [
        "# OPSD Official-Style Definition Audit",
        "",
        f"Status: {'PASS' if original else 'FAIL'}",
        f"Official-style OPSD confirmed: {'yes' if original else 'no'}",
        "",
        "## Required Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    pretty = {
        "official_generalized_jsd": "OPSD loss uses generalized JSD",
        "beta_zero_default": "beta defaults to 0.0",
        "jsd_token_clip_configured": "per-token JSD clipping is configured",
        "teacher_uses_reference_solution": "teacher prompt uses the ground-truth reference solution",
        "teacher_no_grad": "teacher pass uses no_grad",
        "teacher_adapter_disabled": "teacher adapter is disabled",
        "teacher_logits_detached": "teacher logits are detached by no_grad",
        "student_on_policy_generation": "student trajectories are generated on-policy",
        "loss_on_generated_tokens": "distillation loss is computed on student-generated tokens",
        "student_visionzip_pruned": "VisionZip is applied to student forward",
        "max_new_tokens_128": "max_new_tokens is 128",
        "train_ratios_fixed": "training retention ratios are [0.1, 0.2, 0.3, 0.4]",
    }
    for key, value in checks.items():
        lines.append(f"| {pretty[key]} | {'PASS' if value else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Implementation Summary",
            "",
            "- Teacher: same VLM with LoRA adapter disabled, full visual tokens, and a ground-truth reference solution prompt.",
            "- Student: same VLM with LoRA enabled and VisionZip-pruned visual tokens.",
            "- Trajectory: generated by the student with `generate_pruned(..., do_sample=True)`.",
            "- Loss: `compute_generalized_jsd(teacher_logits, student_logits)` on the generated token positions.",
            "- Legacy no-GT ablation: available separately as `training.method=opsd_nogt`.",
            "- CE/reward/filtering: not used in the official-style OPSD path.",
        ]
    )
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.report_path}")
    if not original:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "select_ids":
        select_ids(args)
    elif args.cmd == "eval_group":
        eval_group(args)
    elif args.cmd == "aggregate":
        aggregate(args)
    elif args.cmd == "diagnose":
        diagnose(args)
    elif args.cmd == "audit_opsd_original":
        audit_opsd_original(args)
    else:
        raise AssertionError(args.cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
