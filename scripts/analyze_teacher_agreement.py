#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_jsonl", action="append", required=True, help="Checkpoint eval JSONL. May be repeated.")
    p.add_argument("--checkpoint_name", action="append", default=[], help="Name for each --eval_jsonl.")
    p.add_argument("--baseline_summary_csv", default="")
    p.add_argument("--training_log_jsonl", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--title", default="OPSD Teacher Agreement Analysis")
    return p


def normalize_prediction(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_sample_and_mode(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row.get("sample_id"))][str(row.get("mode"))] = row
    return grouped


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def is_degenerate(text: str) -> bool:
    text = normalize_prediction(text)
    if not text:
        return True
    toks = text.split()
    if len(toks) <= 2:
        return True
    if len(set(toks)) <= 2 and len(toks) >= 6:
        return True
    return False


def summarize_eval(name: str, path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    grouped = by_sample_and_mode(rows)
    full_mode = "full_token_base"
    pruned_mode = "pruned_token_base_no_lora"
    student_mode = "pruned_token_distilled_student"

    sample_ids = sorted(grouped)
    usable = [sid for sid in sample_ids if full_mode in grouped[sid] and pruned_mode in grouped[sid] and student_mode in grouped[sid]]
    changed: list[dict[str, Any]] = []
    raw_agree = 0
    student_agree = 0
    student_equals_raw = 0
    fixed = 0
    broken = 0
    full_lengths: list[float] = []
    raw_lengths: list[float] = []
    student_lengths: list[float] = []
    raw_degenerate = 0
    student_degenerate = 0
    full_degenerate = 0
    raw_latencies: list[float] = []
    student_latencies: list[float] = []
    full_latencies: list[float] = []
    full_tokens: list[float] = []
    kept_tokens: list[float] = []

    for sid in usable:
        full = grouped[sid][full_mode]
        raw = grouped[sid][pruned_mode]
        student = grouped[sid][student_mode]
        full_pred = normalize_prediction(full.get("prediction"))
        raw_pred = normalize_prediction(raw.get("prediction"))
        student_pred = normalize_prediction(student.get("prediction"))
        raw_matches = raw_pred == full_pred
        student_matches = student_pred == full_pred
        if raw_matches:
            raw_agree += 1
        if student_matches:
            student_agree += 1
        if student_pred == raw_pred:
            student_equals_raw += 1
        if (not raw_matches) and student_matches:
            fixed += 1
        if raw_matches and (not student_matches):
            broken += 1
        if raw_pred != student_pred:
            changed.append(
                {
                    "checkpoint": name,
                    "sample_id": sid,
                    "question": student.get("question", raw.get("question", full.get("question", ""))),
                    "gold_answer": student.get("gold_answer", raw.get("gold_answer", full.get("gold_answer", ""))),
                    "full_prediction": full_pred,
                    "raw_pruned_prediction": raw_pred,
                    "distilled_prediction": student_pred,
                    "raw_matches_full": raw_matches,
                    "distilled_matches_full": student_matches,
                    "fixed_by_distillation": (not raw_matches) and student_matches,
                    "broken_by_distillation": raw_matches and (not student_matches),
                }
            )
        full_lengths.append(float(full.get("generated_tokens", 0)))
        raw_lengths.append(float(raw.get("generated_tokens", 0)))
        student_lengths.append(float(student.get("generated_tokens", 0)))
        full_degenerate += int(is_degenerate(full_pred))
        raw_degenerate += int(is_degenerate(raw_pred))
        student_degenerate += int(is_degenerate(student_pred))
        full_latencies.append(float(full.get("latency_seconds", 0)))
        raw_latencies.append(float(raw.get("latency_seconds", 0)))
        student_latencies.append(float(student.get("latency_seconds", 0)))
        full_tokens.append(float(raw.get("num_full_visual_tokens", 0)))
        kept_tokens.append(float(raw.get("num_kept_visual_tokens", 0)))

    n = len(usable)
    raw_rate = raw_agree / n if n else 0.0
    student_rate = student_agree / n if n else 0.0
    gap_recovery = None
    if n and (1.0 - raw_rate) > 1e-12:
        gap_recovery = (student_rate - raw_rate) / (1.0 - raw_rate)

    summary = {
        "checkpoint": name,
        "path": path,
        "num_samples": n,
        "raw_pruned_full_agreement": raw_agree,
        "raw_pruned_full_agreement_rate": raw_rate,
        "distilled_full_agreement": student_agree,
        "distilled_full_agreement_rate": student_rate,
        "delta_agreement": student_rate - raw_rate,
        "gap_recovery": gap_recovery,
        "student_equals_raw_pruned": student_equals_raw,
        "changed_vs_raw_pruned": len(changed),
        "fixed_by_distillation": fixed,
        "broken_by_distillation": broken,
        "avg_full_generated_tokens": mean(full_lengths),
        "avg_raw_pruned_generated_tokens": mean(raw_lengths),
        "avg_distilled_generated_tokens": mean(student_lengths),
        "full_degenerate_generations": full_degenerate,
        "raw_pruned_degenerate_generations": raw_degenerate,
        "distilled_degenerate_generations": student_degenerate,
        "avg_full_latency_seconds": mean(full_latencies),
        "avg_raw_pruned_latency_seconds": mean(raw_latencies),
        "avg_distilled_latency_seconds": mean(student_latencies),
        "avg_full_visual_tokens": mean(full_tokens),
        "avg_kept_visual_tokens": mean(kept_tokens),
    }
    return summary, changed


def summarize_training_log(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    rows = read_jsonl(str(p))
    if not rows:
        return None
    losses = [float(row.get("total_loss", float("nan"))) for row in rows]
    finite_losses = [x for x in losses if math.isfinite(x)]
    return {
        "num_rows": len(rows),
        "first_loss": finite_losses[0] if finite_losses else None,
        "last_loss": finite_losses[-1] if finite_losses else None,
        "avg_loss": mean(finite_losses),
        "nan_losses": len(losses) - len(finite_losses),
        "loss_types": sorted({str(row.get("loss_type")) for row in rows}),
        "step_modes": {k: sum(1 for row in rows if str(row.get("step_mode")) == k) for k in sorted({str(row.get("step_mode")) for row in rows})},
        "distill_modes": sorted({str(row.get("distill_mode")) for row in rows}),
        "ce_loss_values": sorted({float(row.get("ce_loss", 0.0)) for row in rows if row.get("ce_loss", 0.0) is not None}),
        "keep_ratios": {str(k): sum(1 for row in rows if row.get("keep_ratio") == k) for k in sorted({row.get("keep_ratio") for row in rows})},
        "avg_teacher_entropy": mean([float(row.get("teacher_entropy", 0.0)) for row in rows]),
        "avg_generated_tokens": mean([float(row.get("generated_tokens", 0.0)) for row in rows]),
        "avg_full_visual_tokens": mean([float(row.get("num_full_visual_tokens", 0.0)) for row in rows]),
        "avg_kept_visual_tokens": mean([float(row.get("num_kept_visual_tokens", 0.0)) for row in rows]),
    }


def read_baseline_rows(path: str) -> list[dict[str, str]]:
    if not path or not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.checkpoint_name and len(args.checkpoint_name) != len(args.eval_jsonl):
        raise ValueError("--checkpoint_name must be repeated once per --eval_jsonl")
    names = args.checkpoint_name or [Path(p).stem for p in args.eval_jsonl]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    changed_all: list[dict[str, Any]] = []
    for name, path in zip(names, args.eval_jsonl):
        summary, changed = summarize_eval(name, path)
        summaries.append(summary)
        changed_all.extend(changed)

    best = max(summaries, key=lambda row: (float(row["distilled_full_agreement_rate"]), -float(row["broken_by_distillation"]))) if summaries else None
    training_summary = summarize_training_log(args.training_log_jsonl)
    baseline_rows = read_baseline_rows(args.baseline_summary_csv)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_checkpoint": best,
                "checkpoints": summaries,
                "training_summary": training_summary,
                "baseline_summary_csv": args.baseline_summary_csv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "checkpoint_summary.csv").open("w", encoding="utf-8", newline="") as f:
        if summaries:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    with (out_dir / "changed_samples.jsonl").open("w", encoding="utf-8") as f:
        for row in changed_all:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report: list[str] = [f"# {args.title}", ""]
    if baseline_rows:
        report += ["## Baseline Sweep", ""]
        report.append("| Method | Pruner | Ratio | Samples | Failures | Exact Match | Avg Latency | Avg Full Tokens | Avg Kept Tokens |")
        report.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in baseline_rows:
            report.append(
                "| {method} | {pruner} | {keep_ratio} | {num_samples} | {num_failures} | {exact_match} | {average_latency} | {average_full_visual_tokens} | {average_kept_visual_tokens} |".format(
                    **row
                )
            )
        report.append("")
    else:
        report += ["## Baseline Sweep", "", "Baseline summary CSV was not found when analysis ran.", ""]

    report += ["## Training", ""]
    if training_summary:
        report += [
            f"- Rows: {training_summary['num_rows']}",
            f"- Loss types: {training_summary['loss_types']}",
            f"- Step modes: {training_summary['step_modes']}",
            f"- Distill modes: {training_summary['distill_modes']}",
            f"- CE loss values: {training_summary['ce_loss_values']}",
            f"- First/last/avg loss: {training_summary['first_loss']} / {training_summary['last_loss']} / {training_summary['avg_loss']}",
            f"- NaN losses: {training_summary['nan_losses']}",
            f"- Avg generated tokens: {training_summary['avg_generated_tokens']}",
            f"- Avg visual tokens full/kept: {training_summary['avg_full_visual_tokens']} / {training_summary['avg_kept_visual_tokens']}",
            "",
        ]
    else:
        report += ["Training log was not found when analysis ran.", ""]

    report += ["## Teacher Agreement", ""]
    report.append("| Checkpoint | Samples | Raw Agree | Student Agree | Delta | Gap Recovery | Fixed | Broken | Changed | Avg Student Tokens | Student Degenerate |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        report.append(
            f"| {row['checkpoint']} | {row['num_samples']} | {row['raw_pruned_full_agreement_rate']:.4f} | {row['distilled_full_agreement_rate']:.4f} | {row['delta_agreement']:.4f} | {row['gap_recovery'] if row['gap_recovery'] is not None else 'null'} | {row['fixed_by_distillation']} | {row['broken_by_distillation']} | {row['changed_vs_raw_pruned']} | {row['avg_distilled_generated_tokens']} | {row['distilled_degenerate_generations']} |"
        )
    report.append("")
    if best:
        report += [
            "## Decision",
            "",
            f"- Best checkpoint by teacher agreement: `{best['checkpoint']}`.",
            f"- Distilled agreement delta over raw pruned: {best['delta_agreement']:.4f}.",
            f"- Gap recovery: {best['gap_recovery']}.",
            "- Run 1000-sample eval if the best checkpoint has positive agreement delta on this 200-sample run.",
            "- Run multi-ratio next only if this single-ratio run clearly improves over raw pruning.",
            "- Keep on-policy deferred unless teacher-rollout improvement is confirmed at larger eval scale.",
            "",
        ]
    report += ["## Outputs", "", "- `summary.json`", "- `checkpoint_summary.csv`", "- `changed_samples.jsonl`", ""]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"best_checkpoint": best, "num_checkpoints": len(summaries), "output_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
