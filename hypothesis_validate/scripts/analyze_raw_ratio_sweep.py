#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RATIOS = ("r010", "r020", "r030")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired sample analysis for the Raw VisionZip KL ratio sweep.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-samples", type=int, default=100)
    return parser.parse_args()


def read_ratio(path: Path, expected_samples: int, window: int) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(rows) != expected_samples or len(by_id) != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} unique rows in {path}, got {len(rows)}/{len(by_id)}.")
    too_short = [sample_id for sample_id, row in by_id.items() if len(row["kl_full_to_method"]) < window]
    if too_short:
        raise RuntimeError(f"{path}: {len(too_short)} samples have fewer than {window} KL positions.")
    return by_id


def bootstrap_mean_ci(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    means.sort()
    return means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    by_ratio = {
        ratio: read_ratio(
            args.input_root / ratio / "raw_pruned" / "position_divergence.jsonl",
            args.expected_samples,
            args.window,
        )
        for ratio in RATIOS
    }
    ids = set(by_ratio[RATIOS[0]])
    if any(set(by_ratio[ratio]) != ids for ratio in RATIOS[1:]):
        raise RuntimeError("Sample IDs differ across retention ratios.")

    sample_rows: list[dict[str, Any]] = []
    for sample_id in sorted(ids):
        means = {
            ratio: statistics.fmean(by_ratio[ratio][sample_id]["kl_full_to_method"][: args.window])
            for ratio in RATIOS
        }
        sample_rows.append(
            {
                "benchmark": args.benchmark,
                "sample_id": sample_id,
                "window": args.window,
                "mean_kl_r010": means["r010"],
                "mean_kl_r020": means["r020"],
                "mean_kl_r030": means["r030"],
                "strict_monotonic_r010_gt_r020_gt_r030": means["r010"] > means["r020"] > means["r030"],
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for ratio in RATIOS:
        values = [float(row[f"mean_kl_{ratio}"]) for row in sample_rows]
        low, high = bootstrap_mean_ci(values, args.bootstrap_resamples, args.seed + int(ratio[1:]))
        summary_rows.append(
            {
                "benchmark": args.benchmark,
                "comparison": ratio,
                "n": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "fraction_positive": "",
            }
        )
    for left, right in (("r010", "r020"), ("r020", "r030"), ("r010", "r030")):
        differences = [float(row[f"mean_kl_{left}"]) - float(row[f"mean_kl_{right}"]) for row in sample_rows]
        low, high = bootstrap_mean_ci(differences, args.bootstrap_resamples, args.seed + int(left[1:]) + int(right[1:]))
        summary_rows.append(
            {
                "benchmark": args.benchmark,
                "comparison": f"{left}-{right}",
                "n": len(differences),
                "mean": statistics.fmean(differences),
                "median": statistics.median(differences),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "fraction_positive": sum(value > 0 for value in differences) / len(differences),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw_ratio_paired_sample_level.csv", sample_rows)
    write_csv(args.output_dir / "raw_ratio_paired_summary.csv", summary_rows)
    monotonic = sum(row["strict_monotonic_r010_gt_r020_gt_r030"] for row in sample_rows)
    report = [
        f"# {args.benchmark} Raw VisionZip Paired KL Analysis",
        "",
        f"All statistics use the first {args.window} generated-token positions from the same {len(sample_rows)} samples.",
        "",
        f"- Strict per-sample ordering `r010 > r020 > r030`: {monotonic}/{len(sample_rows)} ({100 * monotonic / len(sample_rows):.1f}%).",
        "- Confidence intervals are deterministic nonparametric bootstrap intervals over matched samples.",
        "",
        "| Comparison | Mean | Median | 95% bootstrap CI | Fraction positive |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        fraction = "-" if row["fraction_positive"] == "" else f"{100 * float(row['fraction_positive']):.1f}%"
        report.append(
            f"| {row['comparison']} | {row['mean']:.4f} | {row['median']:.4f} | "
            f"[{row['bootstrap_ci95_low']:.4f}, {row['bootstrap_ci95_high']:.4f}] | {fraction} |"
        )
    (args.output_dir / "raw_ratio_paired_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"benchmark": args.benchmark, "samples": len(sample_rows), "strict_monotonic": monotonic}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
