#!/usr/bin/env python3
"""Analyze the 1% fixed-teacher projection-fraction trend."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
INPUT = HERE / "outputs/consolidated/per_token_metrics_all_steps.parquet"
OUTPUT = HERE / "outputs/analysis/f_trend_d01"
EPS = 1e-12
BOOTSTRAP_DRAWS = 2000
SEED = 42


def bootstrap(samples: pd.DataFrame, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    a = samples["A_sum"].to_numpy(dtype=np.float64)
    p = samples["P_sum"].to_numpy(dtype=np.float64)
    f = samples["F_trajectory"].to_numpy(dtype=np.float64)
    n = len(samples)
    pooled = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    mean = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, n, size=n)
        pooled[draw] = p[indices].sum() / max(a[indices].sum(), EPS)
        mean[draw] = f[indices].mean()
    return {
        "pooled": (float(np.quantile(pooled, 0.025)), float(np.quantile(pooled, 0.975))),
        "trajectory_mean": (
            float(np.quantile(mean, 0.025)),
            float(np.quantile(mean, 0.975)),
        ),
    }


def paired_endpoint_bootstrap(
    trajectories: pd.DataFrame, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    first = trajectories[trajectories["checkpoint_step"] == trajectories["checkpoint_step"].min()]
    final = trajectories[trajectories["checkpoint_step"] == trajectories["checkpoint_step"].max()]
    merged = first.merge(final, on="sample_id", suffixes=("_first", "_final"), validate="one_to_one")
    draws: dict[str, np.ndarray] = {
        "pooled_absolute": np.empty(BOOTSTRAP_DRAWS),
        "pooled_relative_percent": np.empty(BOOTSTRAP_DRAWS),
        "trajectory_mean_absolute": np.empty(BOOTSTRAP_DRAWS),
        "trajectory_mean_relative_percent": np.empty(BOOTSTRAP_DRAWS),
    }
    n = len(merged)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = merged.iloc[rng.integers(0, n, size=n)]
        pooled_first = sampled["P_sum_first"].sum() / sampled["A_sum_first"].sum()
        pooled_final = sampled["P_sum_final"].sum() / sampled["A_sum_final"].sum()
        mean_first = sampled["F_trajectory_first"].mean()
        mean_final = sampled["F_trajectory_final"].mean()
        draws["pooled_absolute"][draw] = pooled_final - pooled_first
        draws["pooled_relative_percent"][draw] = 100.0 * (pooled_final / pooled_first - 1.0)
        draws["trajectory_mean_absolute"][draw] = mean_final - mean_first
        draws["trajectory_mean_relative_percent"][draw] = 100.0 * (
            mean_final / mean_first - 1.0
        )
    return {
        name: {
            "estimate": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in draws.items()
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "checkpoint_step",
        "sample_id",
        "d01_A_jsd",
        "d01_B_jsd",
        "d01_C_jsd",
    ]
    frame = pq.read_table(INPUT, columns=columns).to_pandas()
    frame["P"] = 0.5 * (
        frame["d01_A_jsd"] + frame["d01_B_jsd"] - frame["d01_C_jsd"]
    )
    trajectories = (
        frame.groupby(["checkpoint_step", "sample_id"], sort=True, observed=True)
        .agg(A_sum=("d01_A_jsd", "sum"), P_sum=("P", "sum"), token_count=("P", "size"))
        .reset_index()
    )
    if bool((trajectories["A_sum"] <= EPS).any()):
        raise ValueError("At least one trajectory has zero teacher JSD mass")
    trajectories["F_trajectory"] = trajectories["P_sum"] / trajectories["A_sum"]
    trajectories["F_clipped"] = trajectories["F_trajectory"].clip(0.0, 1.0)
    trajectories.to_parquet(OUTPUT / "per_trajectory_f_d01.parquet", index=False)

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int]] = []
    for step, part in trajectories.groupby("checkpoint_step", sort=True):
        ci = bootstrap(part, rng)
        pooled = float(part["P_sum"].sum() / part["A_sum"].sum())
        values = part["F_trajectory"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "checkpoint_step": int(step),
                "sample_count": int(len(part)),
                "token_count": int(part["token_count"].sum()),
                "F_pooled": pooled,
                "F_pooled_ci_low": ci["pooled"][0],
                "F_pooled_ci_high": ci["pooled"][1],
                "F_trajectory_mean": float(values.mean()),
                "F_trajectory_mean_ci_low": ci["trajectory_mean"][0],
                "F_trajectory_mean_ci_high": ci["trajectory_mean"][1],
                "F_trajectory_median": float(np.median(values)),
                "F_trajectory_q25": float(np.quantile(values, 0.25)),
                "F_trajectory_q75": float(np.quantile(values, 0.75)),
                "F_clipped_mean": float(part["F_clipped"].mean()),
                "fraction_F_negative": float((values < 0.0).mean()),
                "fraction_F_above_one": float((values > 1.0).mean()),
                "A_sum": float(part["A_sum"].sum()),
                "P_sum": float(part["P_sum"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    for column in ("F_pooled", "F_trajectory_mean", "F_clipped_mean"):
        summary[f"{column}_percent_of_step0"] = 100.0 * summary[column] / summary[column].iloc[0]
    summary.to_csv(OUTPUT / "f_trend_d01.csv", index=False)

    steps = summary["checkpoint_step"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    styles = [
        ("F_pooled", "F_pooled_ci_low", "F_pooled_ci_high", "Pooled $\\sum P / \\sum A$", "#176B87"),
        (
            "F_trajectory_mean",
            "F_trajectory_mean_ci_low",
            "F_trajectory_mean_ci_high",
            "Mean trajectory $F_i$",
            "#B54A35",
        ),
    ]
    for value, low, high, label, color in styles:
        y = summary[value].to_numpy()
        axes[0].plot(steps, y, marker="o", linewidth=2.0, label=label, color=color)
        axes[0].fill_between(
            steps,
            summary[low].to_numpy(),
            summary[high].to_numpy(),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axes[1].plot(
            steps,
            100.0 * y / y[0],
            marker="o",
            linewidth=2.0,
            label=label,
            color=color,
        )
    axes[0].set_title("1% intervention: fixed-base-teacher projection fraction")
    axes[0].set_ylabel("Raw projection fraction $F$")
    axes[1].set_title("Relative to step 0")
    axes[1].set_ylabel("Step-0 percentage (%)")
    for axis in axes:
        axis.set_xlabel("R010-only OPSD training step")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        axis.ticklabel_format(style="plain", axis="x")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"f_trend_d01.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    endpoint = {
        "definition": {
            "A": "JSD(q_fixed_full, p_student_r010)",
            "B": "JSD(p_student_r010, p_student_r011)",
            "C": "JSD(q_fixed_full, p_student_r011)",
            "P": "(A+B-C)/2",
            "F_trajectory": "sum_t(P_t)/sum_t(A_t)",
            "F_pooled": "sum_i,t(P_it)/sum_i,t(A_it)",
        },
        "bootstrap": {
            "unit": "sample_id within checkpoint",
            "draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
        },
        "step0": summary.iloc[0].to_dict(),
        "final": summary.iloc[-1].to_dict(),
        "pooled_endpoint_change_percent": float(
            100.0 * (summary["F_pooled"].iloc[-1] / summary["F_pooled"].iloc[0] - 1.0)
        ),
        "trajectory_mean_endpoint_change_percent": float(
            100.0
            * (summary["F_trajectory_mean"].iloc[-1] / summary["F_trajectory_mean"].iloc[0] - 1.0)
        ),
        "paired_endpoint_bootstrap": paired_endpoint_bootstrap(
            trajectories, np.random.default_rng(SEED + 1)
        ),
        "step_trend_spearman": {
            "F_pooled": float(
                spearmanr(summary["checkpoint_step"], summary["F_pooled"]).statistic
            ),
            "F_trajectory_mean": float(
                spearmanr(
                    summary["checkpoint_step"], summary["F_trajectory_mean"]
                ).statistic
            ),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(endpoint, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(endpoint, indent=2))


if __name__ == "__main__":
    main()
