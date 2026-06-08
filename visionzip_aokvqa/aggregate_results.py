#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default=str(OUTPUT_ROOT / "eval"))
    p.add_argument("--reports_dir", default=str(OUTPUT_ROOT / "reports"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_dir = Path(args.eval_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(eval_dir.rglob("*.metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "method": metrics.get("method", path.parent.name),
                "retention_ratio": metrics.get("retention_ratio", ""),
                "benchmark": metrics.get("benchmark", ""),
                "metric": "accuracy",
                "score": metrics.get("accuracy", ""),
                "checkpoint_path": metrics.get("checkpoint_path", ""),
            }
        )
        rows.append(
            {
                "method": metrics.get("method", path.parent.name),
                "retention_ratio": metrics.get("retention_ratio", ""),
                "benchmark": metrics.get("benchmark", ""),
                "metric": "parseable_rate",
                "score": metrics.get("parseable_rate", ""),
                "checkpoint_path": metrics.get("checkpoint_path", ""),
            }
        )
    table_path = reports_dir / "results_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "retention_ratio", "benchmark", "metric", "score", "checkpoint_path"])
        writer.writeheader()
        writer.writerows(rows)
    summary = reports_dir / "results_summary.md"
    lines = [
        "# VisionZip A-OKVQA Reasoning Results Summary",
        "",
        "This file is generated from metric JSON files under the evaluation directory.",
        "No scores are fabricated; missing cells mean that evaluation has not been run yet.",
        "",
        "## Result Rows",
        "",
    ]
    if rows:
        lines.append("| method | retention_ratio | benchmark | metric | score | checkpoint |")
        lines.append("|---|---:|---|---|---:|---|")
        for row in rows:
            lines.append(
                f"| {row['method']} | {row['retention_ratio']} | {row['benchmark']} | "
                f"{row['metric']} | {row['score']} | {row['checkpoint_path']} |"
            )
    else:
        lines.append("No evaluation metrics found yet.")
    lines.extend(
        [
            "",
            "## Reproducibility Notes",
            "",
            "- Base model, dataset, prompt, VisionZip pruning, retention ratios, and max_new_tokens are fixed by the configs.",
            "- The intended comparison varies only the training objective: SFT, GRPO, EPIC, or OPSD.",
            "- Full MME/POPE/GQA/SQA scores are not populated by the smoke test. The lightweight evaluator requires converted benchmark JSONL files via `MME_JSONL`, `POPE_JSONL`, `GQA_JSONL`, and `SQA_JSONL`, or a separate VLMEvalKit run with the same VisionZip model wrapper.",
        ]
    )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {table_path}")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
