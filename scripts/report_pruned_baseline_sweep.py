#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True)
    return p


def load_rows(summary_csv: Path) -> list[dict[str, Any]]:
    with summary_csv.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None, "None"):
            return None
        return float(value)
    except Exception:
        return None


def choose_metric(rows: list[dict[str, Any]]) -> str | None:
    for key in ("multiple_choice_accuracy", "exact_match"):
        if any(as_float(row.get(key)) is not None for row in rows):
            return key
    return None


def make_plot(rows: list[dict[str, Any]], metric: str, output: Path, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("pruner") == "keep_all":
            continue
        x = as_float(row.get("keep_ratio"))
        y = as_float(row.get(metric))
        if x is None or y is None:
            continue
        grouped.setdefault(str(row.get("pruner")), []).append((x, y))
    if not grouped:
        return
    plt.figure()
    for pruner, points in grouped.items():
        points = sorted(points)
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=pruner)
    plt.xlabel("keep_ratio")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("pruner") != "keep_all" and as_float(row.get(metric)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: as_float(r.get(metric)) or -1e9)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    sweep_dir = Path(args.sweep_dir)
    rows = load_rows(sweep_dir / "summary.csv")
    metric = choose_metric(rows)
    latency_key = "average_latency"
    token_key = "average_kept_visual_tokens"

    if metric:
        make_plot(rows, metric, sweep_dir / "accuracy_vs_keep_ratio.png", metric)
    make_plot(rows, latency_key, sweep_dir / "latency_vs_keep_ratio.png", "average latency seconds")
    make_plot(rows, token_key, sweep_dir / "tokens_vs_keep_ratio.png", "average kept visual tokens")

    full = next((r for r in rows if r.get("method") == "full_token_base"), None)
    best = best_row(rows, metric) if metric else None
    r025 = [r for r in rows if as_float(r.get("keep_ratio")) == 0.25 and r.get("pruner") != "keep_all"]
    r0125 = [r for r in rows if as_float(r.get("keep_ratio")) == 0.125 and r.get("pruner") != "keep_all"]

    report = ["# OPSD Pruned Baseline Sweep", ""]
    report.append("## Summary")
    if full:
        report.append(f"- Full-token baseline {metric or 'accuracy'}: {full.get(metric, 'not computed') if metric else 'not computed'}")
    if best and metric:
        report.append(
            f"- Best pruned baseline: {best.get('pruner')} at keep_ratio={best.get('keep_ratio')} "
            f"with {metric}={best.get(metric)}"
        )
    report.append("")
    report.append("## Required Questions")
    report.append("- How much does pruning hurt before training? Compare each pruned row in `summary.csv` against `full_token_base`.")
    if best:
        report.append(f"- Which pruner is reasonable but not too strong? Current best by available metric is `{best.get('pruner')}`.")
    else:
        report.append("- Which pruner is reasonable but not too strong? Accuracy was not computable; use latency/tokens and qualitative predictions.")
    if r025:
        vals = [as_float(r.get(metric)) for r in r025] if metric else []
        vals = [v for v in vals if v is not None]
        report.append(f"- Is keep_ratio=0.25 a good main setting? Rows exist for {len(r025)} pruners; metric range is {min(vals):.4f}-{max(vals):.4f}." if vals else "- Is keep_ratio=0.25 a good main setting? Inspect `summary.csv`; no accuracy metric was computed.")
    if r0125:
        vals = [as_float(r.get(metric)) for r in r0125] if metric else []
        vals = [v for v in vals if v is not None]
        report.append(f"- Is keep_ratio=0.125 too destructive? Rows exist for {len(r0125)} pruners; metric range is {min(vals):.4f}-{max(vals):.4f}." if vals else "- Is keep_ratio=0.125 too destructive? Inspect examples; no accuracy metric was computed.")
    report.append("- Does latency decrease with token count? Check `latency_vs_keep_ratio.png` and `tokens_vs_keep_ratio.png`; raw values are in `summary.csv`.")
    report.append("")
    report.append("## Files")
    report.append("- `summary.csv`")
    report.append("- `predictions_*.jsonl`")
    report.append("- `metrics_*.json`")
    (sweep_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(sweep_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
