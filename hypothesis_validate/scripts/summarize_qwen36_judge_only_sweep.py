#!/usr/bin/env python3
"""Summarize the Qwen3.6-27B judge-only open-ended rollout sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BENCHMARKS = (
    ("MathVista_MINI", "mathvista_mini"),
    ("MMStar_OpenEnded", "mmstar_open_ended"),
)
RATIOS = ("r010", "r020", "r030", "r040", "r100")
K_VALUES = (1, 2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-dir",
        default="qwen36_27b_judge_only",
        help="Judge result subdirectory under each benchmark/ratio condition.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Default: SWEEP_ROOT/qwen36_27b_judge_only_summary",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def make_row(benchmark: str, ratio: str, summary: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "retention_ratio": ratio,
        "samples": int(summary["samples"]),
        "judged_records": int(summary["total_judged_records"]),
        "judge_invocations": int(summary["judge_invocations"]),
        "judge_parseable_rate": float(summary["judge_parseable_rate"]),
        "closed_answer_tag_rate": float(summary["closed_answer_tag_rate"]),
        "raw_response_fallback_rate": float(summary["raw_response_fallback_rate"]),
        "greedy_accuracy": float(summary["greedy_accuracy"]),
        **{f"pass_at_{k}": float(summary["pass_at_k"][str(k)]) for k in K_VALUES},
        "completed_only_pass_at_64": float(summary["completed_only_pass_at_64"]),
        "judge_model": str(summary["judge_model"]),
        "protocol_version": str(summary["protocol_version"]),
        "summary_path": str(path),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "Benchmark",
        "Ratio",
        "Greedy",
        "pass@1",
        "pass@2",
        "pass@4",
        "pass@8",
        "pass@16",
        "pass@32",
        "pass@64",
        "Closed answer",
        "Raw fallback",
    ]
    lines = [
        "# Qwen3.6-27B Judge-Only Scores",
        "",
        "Every rollout is judged by `Qwen/Qwen3.6-27B-FP8`. The candidate is the verbatim",
        "content of closed `<answer>...</answer>` tags, or the complete raw response when no",
        "closed tag exists. No numeric extractor, benchmark answer parser, exact-match shortcut,",
        "or local correctness fallback is used.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [
            row["benchmark"],
            row["retention_ratio"],
            f"{100 * row['greedy_accuracy']:.1f}",
            *[f"{100 * row[f'pass_at_{k}']:.1f}" for k in K_VALUES],
            f"{100 * row['closed_answer_tag_rate']:.1f}",
            f"{100 * row['raw_response_fallback_rate']:.1f}",
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines.extend(
        [
            "",
            "All displayed values other than sample counts are percentages.",
            "",
            "The earlier `qwen7_score_summary.*` is retained only as a legacy artifact because",
            "its MathVista path used numeric answer extraction before semantic judging.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.sweep_root.expanduser().resolve()
    prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix
        else root / "qwen36_27b_judge_only_summary"
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for benchmark, directory in BENCHMARKS:
        for ratio in RATIOS:
            path = root / directory / ratio / args.protocol_dir / "summary.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            for key in ("local_answer_parser_used", "numeric_extractor_used", "exact_match_shortcut_used"):
                if bool(summary.get(key)):
                    raise RuntimeError(f"Forbidden shortcut {key}=true in {path}")
            if int(summary["judge_invocations"]) != int(summary["total_judged_records"]):
                raise RuntimeError(f"Not every record was judged in {path}")
            rows.append(make_row(benchmark, ratio, summary, path))
    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing condition summaries:\n" + "\n".join(missing))
    if not rows:
        raise RuntimeError("No completed judge-only conditions found.")

    csv_path = prefix.with_suffix(".csv")
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    atomic_text(prefix.with_suffix(".md"), markdown(rows))
    print(json.dumps({"conditions": len(rows), "missing": missing, "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
