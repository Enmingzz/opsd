#!/usr/bin/env python3
"""Summarize concentration of generated-token native Budget Advantage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mass_fraction(top_mass: float, total_mass: float) -> float:
    return top_mass / total_mass if total_mass > 0.0 else float("nan")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1].")
    rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {args.scores}")

    records: list[dict[str, float | int | str]] = []
    pooled_values: list[np.ndarray] = []
    for row in rows:
        values = np.asarray(row["metrics"]["student_distribution_budget_advantage"], dtype=np.float64)
        deltas = np.asarray(row["metrics"]["student_distribution_action_logprob_delta"], dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or values.shape != deltas.shape:
            raise ValueError(f"Invalid BA arrays for {row['condition_id']}")
        if np.any(values < 0.0) or not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite or negative BA for {row['condition_id']}")
        k = max(1, math.ceil(args.top_fraction * len(values)))
        order = np.argsort(values, kind="stable")
        selected = values[order[-k:]]
        total_mass = float(values.sum())
        top_mass = float(selected.sum())
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "condition_id": str(row["condition_id"]),
                "retention_ratio": float(row["retention_ratio"]),
                "b_plus_ratio": float(row["b_plus_ratio"]),
                "token_count": int(len(values)),
                "top_token_count": int(k),
                "positive_ba_token_count": int(np.count_nonzero(values > 0.0)),
                "negative_delta_token_count": int(np.count_nonzero(deltas < 0.0)),
                "ba_mass": total_mass,
                "top_ba_mass": top_mass,
                "top_ba_mass_fraction": mass_fraction(top_mass, total_mass),
            }
        )
        pooled_values.append(values)

    frame = pd.DataFrame.from_records(records)
    valid = frame[frame["ba_mass"] > 0.0].reset_index(drop=True)
    all_values = np.concatenate(pooled_values)
    global_k = max(1, math.ceil(args.top_fraction * len(all_values)))
    global_top_mass = float(np.partition(all_values, len(all_values) - global_k)[-global_k:].sum())

    rng = np.random.default_rng(args.seed)
    bootstrap = []
    for _ in range(args.bootstrap_resamples):
        sampled = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
        bootstrap.append(
            mass_fraction(float(sampled["top_ba_mass"].sum()), float(sampled["ba_mass"].sum()))
        )
    bootstrap_values = np.asarray([x for x in bootstrap if np.isfinite(x)])

    summary = {
        "definition": "BA_t=max(log p_bplus(y_t)-log p_b(y_t),0)",
        "selection_scope": "top fraction selected separately within each trajectory",
        "top_fraction_requested": args.top_fraction,
        "trajectory_count": int(len(frame)),
        "trajectory_count_with_positive_ba": int(len(valid)),
        "all_zero_ba_trajectory_count": int(len(frame) - len(valid)),
        "token_count": int(frame["token_count"].sum()),
        "selected_token_count": int(frame["top_token_count"].sum()),
        "selected_token_fraction_actual": float(frame["top_token_count"].sum() / frame["token_count"].sum()),
        "positive_ba_token_count": int(frame["positive_ba_token_count"].sum()),
        "positive_ba_token_fraction": float(frame["positive_ba_token_count"].sum() / frame["token_count"].sum()),
        "negative_budget_effect_token_fraction": float(frame["negative_delta_token_count"].sum() / frame["token_count"].sum()),
        "within_trajectory_top_ba_mass_fraction_pooled": mass_fraction(
            float(frame["top_ba_mass"].sum()), float(frame["ba_mass"].sum())
        ),
        "within_trajectory_top_ba_mass_fraction_bootstrap_ci95": [
            float(np.quantile(bootstrap_values, 0.025)),
            float(np.quantile(bootstrap_values, 0.975)),
        ],
        "per_trajectory_top_ba_mass_fraction_mean": float(valid["top_ba_mass_fraction"].mean()),
        "per_trajectory_top_ba_mass_fraction_median": float(valid["top_ba_mass_fraction"].median()),
        "global_top_ba_mass_fraction": mass_fraction(global_top_mass, float(all_values.sum())),
        "retention_ratios": sorted(float(x) for x in frame["retention_ratio"].unique()),
        "b_plus_ratios": sorted(float(x) for x in frame["b_plus_ratio"].unique()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_trajectory.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
