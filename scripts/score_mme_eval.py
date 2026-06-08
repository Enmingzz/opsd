#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PERCEPTION = {
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
}
COGNITION = {
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    return p


def parse_yes_no(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip().lower()).replace(".", "")
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
    return "other"


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_pairs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    total = 0
    correct = 0
    other = 0
    latencies: list[float] = []
    generated_tokens: list[float] = []
    full_tokens: list[float] = []
    kept_tokens: list[float] = []

    for row in rows:
        answer = parse_yes_no(row.get("gold_answer"))
        pred = parse_yes_no(row.get("prediction"))
        score = float(pred == answer)
        total += 1
        correct += int(score)
        other += int(pred == "other")
        category = str(row.get("category", "unknown"))
        pair_id = str(row.get("mme_pair_id") or row.get("mme_question_id") or row.get("question_id") or row.get("sample_id"))
        category_pairs[category][pair_id].append(score)
        latencies.append(float(row.get("latency_seconds", 0.0)))
        generated_tokens.append(float(row.get("generated_tokens", 0.0)))
        full_tokens.append(float(row.get("num_full_visual_tokens", 0.0)))
        kept_tokens.append(float(row.get("num_kept_visual_tokens", 0.0)))

    category_scores: dict[str, float] = {}
    category_acc: dict[str, float] = {}
    category_acc_plus: dict[str, float] = {}
    invalid_pair_count = 0
    for category, pairs in category_pairs.items():
        pair_scores = []
        pair_acc = []
        pair_acc_plus = []
        for scores in pairs.values():
            if len(scores) != 2:
                invalid_pair_count += 1
            acc = sum(scores) / len(scores) * 100.0 if scores else 0.0
            acc_plus = 100.0 if len(scores) == 2 and sum(scores) == 2 else 0.0
            pair_scores.append(acc + acc_plus)
            pair_acc.append(acc)
            pair_acc_plus.append(acc_plus)
        category_scores[category] = mean(pair_scores) if pair_scores else 0.0
        category_acc[category] = mean(pair_acc) if pair_acc else 0.0
        category_acc_plus[category] = mean(pair_acc_plus) if pair_acc_plus else 0.0

    perception_score = sum(score for cat, score in category_scores.items() if cat in PERCEPTION)
    cognition_score = sum(score for cat, score in category_scores.items() if cat in COGNITION)
    total_score = perception_score + cognition_score
    return {
        "num_rows": total,
        "yes_no_accuracy": correct / total if total else 0.0,
        "other_predictions": other,
        "mme_perception_score": perception_score,
        "mme_cognition_score": cognition_score,
        "mme_total_score": total_score,
        "invalid_pair_count": invalid_pair_count,
        "avg_latency_seconds": mean(latencies) if latencies else None,
        "avg_generated_tokens": mean(generated_tokens) if generated_tokens else None,
        "avg_full_visual_tokens": mean(full_tokens) if full_tokens else None,
        "avg_kept_visual_tokens": mean(kept_tokens) if kept_tokens else None,
        "category_scores": category_scores,
        "category_accuracy": category_acc,
        "category_acc_plus": category_acc_plus,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.eval_jsonl)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("mode", row.get("eval_mode", "unknown")))].append(row)

    summaries = {mode: summarize_mode(mode_rows) for mode, mode_rows in sorted(by_mode.items())}
    (out_dir / "mme_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "mme_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "mode",
            "num_rows",
            "yes_no_accuracy",
            "other_predictions",
            "mme_perception_score",
            "mme_cognition_score",
            "mme_total_score",
            "invalid_pair_count",
            "avg_latency_seconds",
            "avg_generated_tokens",
            "avg_full_visual_tokens",
            "avg_kept_visual_tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode, summary in summaries.items():
            writer.writerow({key: (mode if key == "mode" else summary.get(key)) for key in fieldnames})

    lines = ["# MME Evaluation", ""]
    lines.append("| Mode | Rows | Acc | MME Total | Perception | Cognition | Other | Avg Kept Tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode, summary in summaries.items():
        lines.append(
            f"| {mode} | {summary['num_rows']} | {summary['yes_no_accuracy']:.4f} | "
            f"{summary['mme_total_score']:.2f} | {summary['mme_perception_score']:.2f} | "
            f"{summary['mme_cognition_score']:.2f} | {summary['other_predictions']} | "
            f"{summary['avg_kept_visual_tokens']:.2f} |"
        )
    lines.append("")
    lines.append("MME total follows the pairwise acc + acc_plus aggregation used by lmms-eval.")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "modes": list(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
