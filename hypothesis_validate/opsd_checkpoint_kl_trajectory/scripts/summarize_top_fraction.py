#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_METRICS = (
    "kl_r025_official_to_r020",
    "kl_base_full_to_r020",
    "kl_base_full_to_r025_official",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the highest-divergence token fraction within each rollout."
    )
    parser.add_argument("--token-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = values[rng.integers(0, values.size, size=values.size)].mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def summarize_rollouts(tokens: pd.DataFrame, fraction: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["checkpoint_label", "checkpoint_step", "sample_id"]
    for (label, step, sample_id), frame in tokens.groupby(keys, sort=False):
        selected_count = max(1, math.ceil(fraction * len(frame)))
        row: dict[str, object] = {
            "checkpoint_label": label,
            "checkpoint_step": int(step),
            "sample_id": str(sample_id),
            "response_token_count": int(len(frame)),
            "selected_token_count": int(selected_count),
            "selected_fraction": float(selected_count / len(frame)),
        }
        for metric in PRIMARY_METRICS:
            # Each metric independently selects its own highest-divergence positions.
            row[metric] = float(frame[metric].nlargest(selected_count).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_checkpoints(
    rollouts: pd.DataFrame,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (label, step), frame in rollouts.groupby(
        ["checkpoint_label", "checkpoint_step"], sort=False
    ):
        row: dict[str, object] = {
            "checkpoint_label": label,
            "checkpoint_step": int(step),
            "n_samples": int(frame["sample_id"].nunique()),
            "mean_response_token_count": float(frame["response_token_count"].mean()),
            "mean_selected_token_count": float(frame["selected_token_count"].mean()),
        }
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            values = frame[metric].to_numpy(dtype=np.float64)
            low, high = bootstrap_ci(values, resamples, seed + int(step) + metric_index)
            row[metric] = float(values.mean())
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("checkpoint_step").reset_index(drop=True)


def build_trends(summary: pd.DataFrame, fraction: float) -> dict[str, object]:
    trends: dict[str, object] = {}
    for metric in PRIMARY_METRICS:
        initial = float(summary.iloc[0][metric])
        final = float(summary.iloc[-1][metric])
        trends[metric] = {
            "initial": initial,
            "final": final,
            "absolute_change": final - initial,
            "relative_reduction_percent": 100.0 * (initial - final) / initial,
            "adjacent_decreases": int((np.diff(summary[metric]) < 0).sum()),
            "adjacent_comparisons": int(len(summary) - 1),
        }
    return {
        "selection_fraction": fraction,
        "selection_scope": "independently within each rollout and metric",
        "aggregation": "selected-token mean within rollout, then equal-weight mean across rollouts",
        "trends": trends,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    tokens = pd.read_parquet(args.token_metrics.expanduser().resolve())
    missing = sorted(set(PRIMARY_METRICS) - set(tokens.columns))
    if missing:
        raise ValueError(f"Missing metrics: {missing}")
    rollouts = summarize_rollouts(tokens, args.fraction)
    summary = summarize_checkpoints(
        rollouts,
        args.bootstrap_resamples,
        args.seed,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"top{int(round(100 * args.fraction)):02d}"
    rollouts.to_csv(output_dir / f"per_sample_{suffix}_token_kl.csv", index=False)
    summary.to_csv(output_dir / f"checkpoint_summary_{suffix}_token_kl.csv", index=False)
    atomic_write_text(
        output_dir / f"{suffix}_token_kl_trends.json",
        json.dumps(build_trends(summary, args.fraction), indent=2, sort_keys=True) + "\n",
    )
    print(summary[["checkpoint_step", *PRIMARY_METRICS]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
