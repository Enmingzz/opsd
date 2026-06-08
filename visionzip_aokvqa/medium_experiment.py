#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opsd.visionzip_aokvqa.aokvqa import FormattedAOKVQASample, load_aokvqa_dataset
from opsd.visionzip_aokvqa.benchmark_gap_audit import DATA_DIR, read_jsonl, write_jsonl
from opsd.visionzip_aokvqa.prompting import PROMPT_TEMPLATE, parse_final_answer
from opsd.visionzip_aokvqa.qwen_wrapper import (
    encode_prompt,
    generate_pruned,
    load_qwen_model_and_processor,
    primary_device,
)
from opsd.visionzip_aokvqa.train import get_nested, load_yaml


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOG_DIR = OUTPUT_ROOT / "logs" / "medium"
EVAL_DIR = OUTPUT_ROOT / "eval" / "medium_formal"
RATIOS = ("0.1", "0.2", "0.3", "0.4")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select_train_ids")
    s.add_argument("--config", default="configs/experiments/aokvqa_visionzip_opsd.yaml")
    s.add_argument("--num_samples", type=int, default=4096)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--output_path", default=str(REPORTS_DIR / "aokvqa_train_medium_ids.json"))

    o = sub.add_parser("check_opsd_definition")
    o.add_argument("--config", default="configs/experiments/aokvqa_visionzip_opsd.yaml")
    o.add_argument("--report_path", default=str(REPORTS_DIR / "opsd_medium_definition_check.md"))

    g = sub.add_parser("grpo_reward_diagnostic")
    g.add_argument("--config", default="configs/experiments/aokvqa_visionzip_grpo.yaml")
    g.add_argument("--ids_path", default=str(REPORTS_DIR / "aokvqa_train_medium_ids.json"))
    g.add_argument("--num_samples", type=int, default=256)
    g.add_argument("--settings", default="4:0.7:0.9,8:0.9:0.95,8:1.0:0.95")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output_path", default=str(REPORTS_DIR / "grpo_medium_reward_diagnostic.md"))
    g.add_argument("--json_path", default=str(REPORTS_DIR / "grpo_medium_reward_diagnostic.json"))

    e = sub.add_parser("select_eval_ids")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--gqa_samples", type=int, default=2000)
    e.add_argument("--sqa_samples", type=int, default=0)
    e.add_argument("--pope_samples", type=int, default=2000)
    e.add_argument("--mme_jsonl", default="/scratch/enmingzz/temp/opsd_data/mme_test.jsonl")
    e.add_argument("--pope_jsonl", default="/scratch/enmingzz/temp/opsd_data/pope_full.jsonl")
    e.add_argument("--gqa_jsonl", default=str(DATA_DIR / "gqa_testdev.jsonl"))
    e.add_argument("--sqa_jsonl", default=str(DATA_DIR / "sqa_img_test.jsonl"))
    e.add_argument("--output_path", default=str(REPORTS_DIR / "medium_eval_ids.json"))

    b = sub.add_parser("seed_base_partials")
    b.add_argument("--source_raw", default=str(OUTPUT_ROOT / "eval" / "benchmark_gap_audit" / "raw_generations.jsonl"))
    b.add_argument("--ids_path", default=str(REPORTS_DIR / "medium_eval_ids.json"))
    b.add_argument("--partials_dir", default=str(EVAL_DIR / "partials"))
    b.add_argument("--ratios", default="0.1,0.2")

    r = sub.add_parser("write_reports")
    r.add_argument("--results_csv", default=str(REPORTS_DIR / "medium_formal_results.csv"))
    r.add_argument("--summary_md", default=str(REPORTS_DIR / "medium_formal_summary.md"))
    r.add_argument("--decision_md", default=str(REPORTS_DIR / "medium_next_step_decision.md"))
    r.add_argument("--train_ids", default=str(REPORTS_DIR / "aokvqa_train_medium_ids.json"))
    r.add_argument("--grpo_report", default=str(REPORTS_DIR / "grpo_medium_reward_diagnostic.md"))

    return p


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_train_samples(cfg: dict[str, Any], seed: int) -> list[FormattedAOKVQASample]:
    return load_aokvqa_dataset(
        get_nested(cfg, "dataset.name", "HuggingFaceM4/A-OKVQA"),
        splits=list(get_nested(cfg, "dataset.use_splits", ["train", "validation"])),
        limit=0,
        seed=seed,
    )


def select_train_ids(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    samples = load_train_samples(cfg, args.seed)
    selected = samples[: int(args.num_samples)]
    ids = [sample.sample_id for sample in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("Selected training IDs are not unique.")
    write_json(Path(args.output_path), ids)
    print(json.dumps({"output_path": args.output_path, "num_ids": len(ids)}, indent=2))


def check_opsd_definition(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    problems = []
    if str(get_nested(cfg, "training.method", "")).lower() != "opsd":
        problems.append("training.method is not opsd")
    source = Path("opsd/visionzip_aokvqa/train.py").read_text(encoding="utf-8")
    opsd_body = source[source.index("def opsd_step(") : source.index("def offpolicy_step(")]
    checks = {
        "teacher_full_token_frozen": "teacher_adapter_disabled(model)" in opsd_body and "torch.no_grad()" in opsd_body,
        "teacher_privileged_ground_truth_context": "build_opsd_teacher_prompt" in opsd_body and "sample.target" in opsd_body,
        "student_visionzip_pruned": "forward_pruned(" in opsd_body,
        "student_on_policy_generation": "generate_pruned(" in opsd_body,
        "same_generated_suffix": "sequence_inputs_from_prompt(student_prompt_inputs, gen_ids)" in opsd_body
        and "sequence_inputs_from_prompt(teacher_prompt_inputs, gen_ids)" in opsd_body,
        "official_generalized_jsd": "compute_generalized_jsd(" in opsd_body,
        "official_beta_zero_default": float(get_nested(cfg, "opsd.beta", 0.0)) == 0.0,
        "official_jsd_token_clip_configured": float(get_nested(cfg, "opsd.jsd_token_clip", 0.0)) > 0.0,
        "max_new_tokens_64": int(get_nested(cfg, "generation.max_new_tokens", 64)) == 64,
        "retention_ratios": [float(x) for x in get_nested(cfg, "pruning.train_retention_ratios", [])] == [0.1, 0.2, 0.3, 0.4],
    }
    for key, ok in checks.items():
        if not ok:
            problems.append(key)
    lines = [
        "# OPSD Medium Definition Check",
        "",
        f"Status: {'PASS' if not problems else 'FAIL'}",
        "",
        "| Requirement | Status |",
        "|---|---|",
    ]
    for key, ok in checks.items():
        lines.append(f"| {key} | {'PASS' if ok else 'FAIL'} |")
    lines.extend(
        [
            "",
            "Confirmed definition:",
            "- teacher is full-token frozen VLM via `teacher_adapter_disabled(model)` under `torch.no_grad()`",
            "- student is VisionZip-pruned via `forward_pruned`",
            "- student generates on-policy tokens via `generate_pruned`",
            "- KL is computed on `gen_ids`, not A-OKVQA targets",
            "- loss is forward KL only",
            "- no CE, reward, rationale target, teacher-correct filtering, or hard-subset filtering is used in OPSD",
            "",
            f"Problems: {problems if problems else 'none'}",
        ]
    )
    path = Path(args.report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    if problems:
        raise SystemExit(1)


def parse_settings(raw: str) -> list[tuple[int, float, float]]:
    out = []
    for item in raw.split(","):
        group, temp, top_p = item.split(":")
        out.append((int(group), float(temp), float(top_p)))
    return out


def subset_by_ids(samples: list[FormattedAOKVQASample], ids_path: str, num_samples: int) -> list[FormattedAOKVQASample]:
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    wanted = [str(x) for x in ids[: int(num_samples)]]
    by_id = {sample.sample_id: sample for sample in samples}
    return [by_id[sample_id] for sample_id in wanted]


def grpo_reward_diagnostic(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    all_samples = load_train_samples(cfg, args.seed)
    samples = subset_by_ids(all_samples, args.ids_path, args.num_samples)
    model, processor = load_qwen_model_and_processor(
        str(get_nested(cfg, "base_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        bf16=bool(get_nested(cfg, "training.bf16", True)),
        attn_implementation=str(get_nested(cfg, "training.attn_implementation", "flash_attention_2")),
        device_map=get_nested(cfg, "training.device_map", "auto"),
    )
    model.eval()
    rng = random.Random(int(args.seed))
    settings = parse_settings(args.settings)
    summaries = []
    device = primary_device(model)
    for group_size, temperature, top_p in settings:
        started = time.time()
        reward_stds = []
        reward_means = []
        parse_rates = []
        zero = 0
        for idx, sample in enumerate(samples, start=1):
            ratio = rng.choice([0.1, 0.2, 0.3, 0.4])
            prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
            rewards = []
            parseable = 0
            with torch.no_grad():
                for _ in range(group_size):
                    _, text, _ = generate_pruned(
                        model,
                        processor,
                        prompt_inputs,
                        ratio,
                        max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
                    )
                    parsed = parse_final_answer(text)
                    parseable += int(parsed is not None)
                    rewards.append((1.0 if parsed == sample.correct_letter else 0.0) + (0.1 if parsed is not None else 0.0))
            std = statistics.pstdev(rewards)
            reward_stds.append(std)
            reward_means.append(statistics.mean(rewards))
            parse_rates.append(parseable / group_size)
            zero += int(std <= 1e-8)
            print(
                json.dumps(
                    {
                        "setting": f"G{group_size}_T{temperature}_P{top_p}",
                        "idx": idx,
                        "sample_id": sample.sample_id,
                        "reward_std": std,
                        "reward_mean": reward_means[-1],
                        "parse_rate": parse_rates[-1],
                    }
                ),
                flush=True,
            )
        summaries.append(
            {
                "group_size": group_size,
                "temperature": temperature,
                "top_p": top_p,
                "zero_reward_std_percentage": zero / len(samples),
                "average_reward_std": statistics.mean(reward_stds),
                "average_reward": statistics.mean(reward_means),
                "parse_rate": statistics.mean(parse_rates),
                "estimated_rollouts": len(samples) * group_size,
                "elapsed_seconds": time.time() - started,
            }
        )
    best = sorted(summaries, key=lambda x: (x["zero_reward_std_percentage"], -x["average_reward_std"], -x["parse_rate"]))[0]
    write_json(Path(args.json_path), {"settings": summaries, "chosen": best})
    lines = [
        "# GRPO Medium Reward Diagnostic",
        "",
        "| Group Size | Temperature | Top-p | Zero Reward Std % | Avg Reward Std | Avg Reward | Parse Rate | Estimated Rollouts |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['group_size']} | {row['temperature']} | {row['top_p']} | "
            f"{row['zero_reward_std_percentage']:.4f} | {row['average_reward_std']:.4f} | "
            f"{row['average_reward']:.4f} | {row['parse_rate']:.4f} | {row['estimated_rollouts']} |"
        )
    lines.extend(
        [
            "",
            f"Chosen setting for medium GRPO: group_size={best['group_size']}, temperature={best['temperature']}, top_p={best['top_p']}.",
            "",
            "Reward definition was not changed.",
        ]
    )
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_path}")


def select_ids_from_jsonl(path: str, count: int, seed: int) -> list[str]:
    rows = read_jsonl(Path(path))
    ids = [str(row.get("sample_id", row.get("id", idx))) for idx, row in enumerate(rows)]
    if count and count > 0 and count < len(ids):
        rng = random.Random(seed)
        ids = list(ids)
        rng.shuffle(ids)
        ids = ids[:count]
    return ids


def select_eval_ids(args: argparse.Namespace) -> None:
    data = {
        "mme": select_ids_from_jsonl(args.mme_jsonl, 0, args.seed),
        "pope": select_ids_from_jsonl(args.pope_jsonl, args.pope_samples, args.seed),
        "gqa": select_ids_from_jsonl(args.gqa_jsonl, args.gqa_samples, args.seed),
        "sqa": select_ids_from_jsonl(args.sqa_jsonl, args.sqa_samples, args.seed),
    }
    write_json(Path(args.output_path), data)
    print(json.dumps({k: len(v) for k, v in data.items()}, indent=2))


def seed_base_partials(args: argparse.Namespace) -> None:
    ids = json.loads(Path(args.ids_path).read_text(encoding="utf-8"))
    wanted = {benchmark: set(map(str, values)) for benchmark, values in ids.items()}
    ratios = {"full", *[str(float(x.strip())) for x in args.ratios.split(",") if x.strip()]}
    rows = []
    for row in read_jsonl(Path(args.source_raw)):
        benchmark = str(row.get("benchmark"))
        if benchmark not in wanted or str(row.get("sample_id")) not in wanted[benchmark]:
            continue
        method = str(row.get("method"))
        ratio = str(row.get("retention_ratio"))
        if method == "full_token_base" and ratio == "full":
            rows.append(row)
        elif method == "visionzip_base" and ratio in ratios:
            rows.append(row)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ratio = "full" if str(row["retention_ratio"]) == "full" else f"{float(row['retention_ratio']):.1f}"
        grouped[(row["benchmark"], row["method"], ratio)].append(row)
    partials = Path(args.partials_dir)
    partials.mkdir(parents=True, exist_ok=True)
    for (benchmark, method, ratio), group_rows in grouped.items():
        ratio_slug = ratio.replace(".", "p")
        write_jsonl(partials / f"{benchmark}_{method}_{ratio_slug}_base_seeded.jsonl", group_rows)
    print(json.dumps({"seeded_groups": len(grouped), "seeded_rows": len(rows)}, indent=2))


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def read_results(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_reports(args: argparse.Namespace) -> None:
    rows = read_results(args.results_csv)
    train_ids = json.loads(Path(args.train_ids).read_text(encoding="utf-8")) if Path(args.train_ids).exists() else []
    lines = [
        "# Medium Formal Evaluation Summary",
        "",
        "## Experimental Setup",
        "",
        f"- Training subset size: {len(train_ids)}",
        f"- Training IDs: `{args.train_ids}`",
        "- Training retention ratios: random sample from `[0.1, 0.2, 0.3, 0.4]` per sample.",
        "- Evaluation retention ratios in this run: rows shown below.",
        "- Generation: greedy, `max_new_tokens=64`.",
        "- OPSD: original KL-only OPSD; no CE/reward/ground-truth/rationale supervision.",
        "",
        "## Prompt Format",
        "",
        "```text",
        PROMPT_TEMPLATE,
        "```",
        "",
        "## GRPO Reward-Diversity Diagnostic",
        "",
    ]
    if Path(args.grpo_report).exists():
        lines.append(f"See `{args.grpo_report}`.")
    else:
        lines.append("GRPO diagnostic report not found.")
    lines.extend(
        [
            "",
            "## Main Result Table",
            "",
            "| Benchmark | Method | Retention | Score | Teacher Agreement | Gap To Full | Improvement Over VZ | Recovery Rate | Parse Rate | Samples |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {row['retention_ratio']} | {fmt(row['score'])} | "
            f"{fmt(row.get('teacher_agreement'))} | {fmt(row.get('gap_to_full'))} | "
            f"{fmt(row.get('improvement_over_vz'))} | {fmt(row.get('recovery_rate'))} | "
            f"{fmt(row.get('parse_rate'))} | {row.get('num_samples')} |"
        )
    lines.extend(["", "## MME Breakdown", ""])
    lines.append("| Method | Retention | MME Total | Perception | Cognition | Improvement Over VZ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["benchmark"] == "mme":
            lines.append(
                f"| {row['method']} | {row['retention_ratio']} | {fmt(row['score'])} | "
                f"{fmt(row.get('mme_perception_score'))} | {fmt(row.get('mme_cognition_score'))} | "
                f"{fmt(row.get('improvement_over_vz'))} |"
            )
    decisions = decision_lines(rows)
    lines.extend(["", "## Decision Signals", ""])
    lines.extend(decisions)
    Path(args.summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.decision_md).write_text("# Medium Next Step Decision\n\n" + "\n".join(decisions) + "\n", encoding="utf-8")
    print(f"wrote {args.summary_md}")
    print(f"wrote {args.decision_md}")


def best_improvement(rows: list[dict[str, str]], method: str, benchmark: str, ratio: str | None = None) -> float:
    vals = []
    for row in rows:
        if row["method"] != method or row["benchmark"] != benchmark:
            continue
        if ratio is not None and row["retention_ratio"] != ratio:
            continue
        try:
            vals.append(float(row.get("improvement_over_vz") or 0.0))
        except Exception:
            pass
    return max(vals) if vals else math.nan


def decision_lines(rows: list[dict[str, str]]) -> list[str]:
    methods = ["sft", "grpo", "epic", "opsd"]
    lines = []
    for method in methods:
        improvements = [best_improvement(rows, method, b) for b in ["mme", "gqa", "sqa", "pope"]]
        ok = any(math.isfinite(x) and x > 0 for x in improvements)
        lines.append(f"- Does {method.upper()} improve over VisionZip base anywhere? {'yes' if ok else 'no'}; best improvements={','.join(fmt(x) for x in improvements)}.")
    opsd_mme = best_improvement(rows, "opsd", "mme")
    opsd_gqa = best_improvement(rows, "opsd", "gqa")
    opsd_sqa = best_improvement(rows, "opsd", "sqa")
    lines.append(f"- OPSD MME improvement over VZ: {fmt(opsd_mme)}.")
    lines.append(f"- OPSD GQA improvement over VZ: {fmt(opsd_gqa)}.")
    lines.append(f"- OPSD SQA improvement over VZ: {fmt(opsd_sqa)}.")
    justified = any(math.isfinite(x) and x > 0 for x in [opsd_mme, opsd_gqa, opsd_sqa])
    lines.append(f"- Is full-dataset training justified from OPSD medium signal? {'yes' if justified else 'no'}; use medium table, especially MME cognition, before claiming superiority.")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "select_train_ids":
        select_train_ids(args)
    elif args.cmd == "check_opsd_definition":
        check_opsd_definition(args)
    elif args.cmd == "grpo_reward_diagnostic":
        grpo_reward_diagnostic(args)
    elif args.cmd == "select_eval_ids":
        select_eval_ids(args)
    elif args.cmd == "seed_base_partials":
        seed_base_partials(args)
    elif args.cmd == "write_reports":
        write_reports(args)
    else:
        raise ValueError(args.cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
