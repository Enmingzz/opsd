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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    return p


def parse_yes_no(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip().lower()).replace(".", "")
    if value in {"yes", "no"}:
        return value
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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    tp = fp = fn = tn = 0
    other = 0
    latencies: list[float] = []
    generated_tokens: list[float] = []
    full_tokens: list[float] = []
    kept_tokens: list[float] = []
    for row in rows:
        gt = parse_yes_no(row.get("gold_answer"))
        pred = parse_yes_no(row.get("prediction"))
        total += 1
        correct += int(pred == gt)
        other += int(pred == "other")
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "yes" and pred == "no":
            fn += 1
        elif gt == "no" and pred == "no":
            tn += 1
        latencies.append(float(row.get("latency_seconds", 0.0)))
        generated_tokens.append(float(row.get("generated_tokens", 0.0)))
        full_tokens.append(float(row.get("num_full_visual_tokens", 0.0)))
        kept_tokens.append(float(row.get("num_kept_visual_tokens", 0.0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pred_yes_ratio = (tp + fp) / total if total else 0.0
    gt_yes_ratio = (tp + fn) / total if total else 0.0
    return {
        "num_rows": total,
        "accuracy": correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_yes_ratio": pred_yes_ratio,
        "gt_yes_ratio": gt_yes_ratio,
        "other_predictions": other,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "avg_latency_seconds": mean(latencies) if latencies else None,
        "avg_generated_tokens": mean(generated_tokens) if generated_tokens else None,
        "avg_full_visual_tokens": mean(full_tokens) if full_tokens else None,
        "avg_kept_visual_tokens": mean(kept_tokens) if kept_tokens else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.eval_jsonl)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("mode", row.get("eval_mode", "unknown")))
        category = str(row.get("category", "unknown"))
        grouped[(mode, category)].append(row)
        overall[mode].append(row)

    summary: dict[str, Any] = {"overall": {}, "by_category": {}}
    for mode, mode_rows in sorted(overall.items()):
        summary["overall"][mode] = summarize(mode_rows)
    for (mode, category), mode_rows in sorted(grouped.items()):
        summary["by_category"].setdefault(mode, {})[category] = summarize(mode_rows)

    (out_dir / "pope_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "scope",
        "mode",
        "category",
        "num_rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pred_yes_ratio",
        "other_predictions",
        "avg_full_visual_tokens",
        "avg_kept_visual_tokens",
        "avg_latency_seconds",
        "avg_generated_tokens",
    ]
    with (out_dir / "pope_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for mode, vals in summary["overall"].items():
            writer.writerow({"scope": "overall", "mode": mode, "category": "all", **{k: vals.get(k) for k in fields if k not in {"scope", "mode", "category"}}})
        for mode, cats in summary["by_category"].items():
            for category, vals in cats.items():
                writer.writerow({"scope": "category", "mode": mode, "category": category, **{k: vals.get(k) for k in fields if k not in {"scope", "mode", "category"}}})

    lines = ["# POPE Evaluation", ""]
    lines.append("| Scope | Mode | Category | Rows | Acc | Precision | Recall | F1 | Yes Ratio | Avg Kept Tokens |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode, vals in summary["overall"].items():
        lines.append(
            f"| overall | {mode} | all | {vals['num_rows']} | {vals['accuracy']:.4f} | "
            f"{vals['precision']:.4f} | {vals['recall']:.4f} | {vals['f1']:.4f} | "
            f"{vals['pred_yes_ratio']:.4f} | {vals['avg_kept_visual_tokens']:.2f} |"
        )
    for mode, cats in summary["by_category"].items():
        for category, vals in cats.items():
            lines.append(
                f"| category | {mode} | {category} | {vals['num_rows']} | {vals['accuracy']:.4f} | "
                f"{vals['precision']:.4f} | {vals['recall']:.4f} | {vals['f1']:.4f} | "
                f"{vals['pred_yes_ratio']:.4f} | {vals['avg_kept_visual_tokens']:.2f} |"
            )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "modes": list(summary["overall"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
