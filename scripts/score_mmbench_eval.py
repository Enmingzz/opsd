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


def first_choice_letter(text: Any) -> str | None:
    match = re.search(r"\b([A-D])\b", str(text or "").strip().upper())
    return match.group(1) if match else None


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
    invalid = 0
    latencies: list[float] = []
    generated_tokens: list[float] = []
    full_tokens: list[float] = []
    kept_tokens: list[float] = []
    for row in rows:
        gt = first_choice_letter(row.get("gold_answer"))
        pred = first_choice_letter(row.get("prediction"))
        total += 1
        invalid += int(pred is None)
        correct += int(pred == gt and gt is not None)
        latencies.append(float(row.get("latency_seconds", 0.0)))
        generated_tokens.append(float(row.get("generated_tokens", 0.0)))
        full_tokens.append(float(row.get("num_full_visual_tokens", 0.0)))
        kept_tokens.append(float(row.get("num_kept_visual_tokens", 0.0)))
    return {
        "num_rows": total,
        "accuracy": correct / total if total else 0.0,
        "invalid_predictions": invalid,
        "invalid_rate": invalid / total if total else 0.0,
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

    overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("mode", row.get("eval_mode", "unknown")))
        category = str(row.get("category", "unknown"))
        overall[mode].append(row)
        by_category[(mode, category)].append(row)

    summary: dict[str, Any] = {"overall": {}, "by_category": {}}
    for mode, mode_rows in sorted(overall.items()):
        summary["overall"][mode] = summarize(mode_rows)
    for (mode, category), mode_rows in sorted(by_category.items()):
        summary["by_category"].setdefault(mode, {})[category] = summarize(mode_rows)

    (out_dir / "mmbench_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = [
        "scope",
        "mode",
        "category",
        "num_rows",
        "accuracy",
        "invalid_predictions",
        "invalid_rate",
        "avg_full_visual_tokens",
        "avg_kept_visual_tokens",
        "avg_latency_seconds",
        "avg_generated_tokens",
    ]
    with (out_dir / "mmbench_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for mode, vals in summary["overall"].items():
            writer.writerow({"scope": "overall", "mode": mode, "category": "all", **{k: vals.get(k) for k in fields if k not in {"scope", "mode", "category"}}})
        for mode, cats in summary["by_category"].items():
            for category, vals in cats.items():
                writer.writerow({"scope": "category", "mode": mode, "category": category, **{k: vals.get(k) for k in fields if k not in {"scope", "mode", "category"}}})

    lines = ["# MMBench Evaluation", ""]
    lines.append("This is simple option-letter accuracy, not GPT-based official MMBench matching.")
    lines.append("")
    lines.append("| Scope | Mode | Category | Rows | Acc | Invalid | Avg Kept Tokens |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for mode, vals in summary["overall"].items():
        lines.append(
            f"| overall | {mode} | all | {vals['num_rows']} | {vals['accuracy']:.4f} | "
            f"{vals['invalid_predictions']} | {vals['avg_kept_visual_tokens']:.2f} |"
        )
    for mode, cats in summary["by_category"].items():
        for category, vals in cats.items():
            lines.append(
                f"| category | {mode} | {category} | {vals['num_rows']} | {vals['accuracy']:.4f} | "
                f"{vals['invalid_predictions']} | {vals['avg_kept_visual_tokens']:.2f} |"
            )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "modes": list(summary["overall"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
