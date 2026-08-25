#!/usr/bin/env python3
"""Validate and summarize the four-checkpoint fixed-teacher F trend."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs" / "analysis"
STEPS = (0, 1024, 2048, 3072)
BOOTSTRAP_DRAWS = 2000
SEED = 42
EPS = 1e-12


def bootstrap(part: pd.DataFrame, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    a = part["A_sum"].to_numpy(dtype=np.float64)
    p = part["P_sum"].to_numpy(dtype=np.float64)
    f = part["F_trajectory"].to_numpy(dtype=np.float64)
    n = len(part)
    pooled = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    trajectory_mean = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, n, size=n)
        pooled[draw] = p[indices].sum() / max(a[indices].sum(), EPS)
        trajectory_mean[draw] = f[indices].mean()
    return {
        "pooled": tuple(float(x) for x in np.quantile(pooled, (0.025, 0.975))),
        "trajectory_mean": tuple(
            float(x) for x in np.quantile(trajectory_mean, (0.025, 0.975))
        ),
    }


def load_and_validate() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    expected_ids: set[str] | None = None
    for step in STEPS:
        root = HERE / "outputs" / f"step_{step:06d}"
        manifest_path = root / "manifest.json"
        parquet_path = root / "per_token_metrics.parquet"
        if not manifest_path.is_file() or not parquet_path.is_file():
            raise FileNotFoundError(f"Incomplete step {step}: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "status": "complete",
            "checkpoint_step": step,
            "teacher_mode": "fixed_base",
            "pruning_method": "random",
            "sample_count_expected": 200,
            "offset": 0,
        }
        for key, value in required.items():
            if manifest.get(key) != value:
                raise ValueError(f"Step {step}: manifest {key}={manifest.get(key)!r}, expected {value!r}")
        if manifest.get("interventions") != [0.01]:
            raise ValueError(f"Step {step}: unexpected interventions {manifest.get('interventions')}")
        columns = [
            "checkpoint_step",
            "sample_id",
            "d01_A_jsd",
            "d01_B_jsd",
            "d01_C_jsd",
        ]
        frame = pq.read_table(parquet_path, columns=columns).to_pandas()
        ids = set(frame["sample_id"].astype(str).unique())
        if len(ids) != 200:
            raise ValueError(f"Step {step}: expected 200 sample IDs, found {len(ids)}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"Step {step}: sample IDs do not match step 0")
        if not np.isfinite(frame[["d01_A_jsd", "d01_B_jsd", "d01_C_jsd"]].to_numpy()).all():
            raise ValueError(f"Step {step}: non-finite divergence")
        frames.append(frame)
        manifests.append(manifest)
    return pd.concat(frames, ignore_index=True), manifests


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, manifests = load_and_validate()
    frame["P"] = 0.5 * (frame["d01_A_jsd"] + frame["d01_B_jsd"] - frame["d01_C_jsd"])
    trajectories = (
        frame.groupby(["checkpoint_step", "sample_id"], sort=True, observed=True)
        .agg(
            A_sum=("d01_A_jsd", "sum"),
            B_sum=("d01_B_jsd", "sum"),
            C_sum=("d01_C_jsd", "sum"),
            P_sum=("P", "sum"),
            token_count=("P", "size"),
        )
        .reset_index()
    )
    if bool((trajectories["A_sum"] <= EPS).any()):
        raise ValueError("At least one trajectory has zero fixed-teacher JSD mass")
    trajectories["F_trajectory"] = trajectories["P_sum"] / trajectories["A_sum"]
    trajectories.to_parquet(OUTPUT / "per_trajectory_f_d01.parquet", index=False)

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int]] = []
    for step, part in trajectories.groupby("checkpoint_step", sort=True):
        ci = bootstrap(part, rng)
        rows.append(
            {
                "checkpoint_step": int(step),
                "sample_count": int(len(part)),
                "token_count": int(part["token_count"].sum()),
                "F_pooled": float(part["P_sum"].sum() / part["A_sum"].sum()),
                "F_pooled_ci_low": ci["pooled"][0],
                "F_pooled_ci_high": ci["pooled"][1],
                "F_trajectory_mean": float(part["F_trajectory"].mean()),
                "F_trajectory_mean_ci_low": ci["trajectory_mean"][0],
                "F_trajectory_mean_ci_high": ci["trajectory_mean"][1],
                "F_trajectory_median": float(part["F_trajectory"].median()),
                "A_token_mean": float(part["A_sum"].sum() / part["token_count"].sum()),
                "B_token_mean": float(part["B_sum"].sum() / part["token_count"].sum()),
                "C_token_mean": float(part["C_sum"].sum() / part["token_count"].sum()),
                "fraction_F_negative": float((part["F_trajectory"] < 0).mean()),
                "fraction_F_above_one": float((part["F_trajectory"] > 1).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    for column in ("F_pooled", "F_trajectory_mean"):
        summary[f"{column}_percent_of_step0"] = 100.0 * summary[column] / summary[column].iloc[0]
    summary.to_csv(OUTPUT / "f_trend_d01.csv", index=False)

    steps = summary["checkpoint_step"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), constrained_layout=True)
    styles = (
        ("F_pooled", "F_pooled_ci_low", "F_pooled_ci_high", "Pooled sum(P) / sum(A)", "#176B87"),
        (
            "F_trajectory_mean",
            "F_trajectory_mean_ci_low",
            "F_trajectory_mean_ci_high",
            "Mean trajectory F",
            "#B54A35",
        ),
    )
    for value, low, high, label, color in styles:
        y = summary[value].to_numpy(dtype=np.float64)
        axes[0].plot(steps, y, marker="o", linewidth=2.1, label=label, color=color)
        axes[0].fill_between(
            steps,
            summary[low].to_numpy(dtype=np.float64),
            summary[high].to_numpy(dtype=np.float64),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axes[1].plot(steps, 100.0 * y / y[0], marker="o", linewidth=2.1, label=label, color=color)
    axes[0].set_title("RandomPruner r010 to r011, fixed teacher")
    axes[0].set_ylabel("Projection fraction F")
    axes[1].set_title("Relative to step 0")
    axes[1].set_ylabel("Step-0 percentage (%)")
    for axis in axes:
        axis.set_xlabel("Original OPSD training step")
        axis.set_xticks(steps)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"f_trend_d01.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    endpoint = {
        "definitions": {
            "A": "JSD(q_fixed_base_full, p_checkpoint_random_r010)",
            "B": "JSD(p_checkpoint_random_r010, p_checkpoint_random_r011)",
            "C": "JSD(q_fixed_base_full, p_checkpoint_random_r011)",
            "P": "(A+B-C)/2",
            "F_trajectory": "sum_t(P_it)/sum_t(A_it)",
            "F_pooled": "sum_i,t(P_it)/sum_i,t(A_it)",
        },
        "protocol": {
            "samples": "first 200 LCOT-1k holdout samples",
            "steps": list(STEPS),
            "checkpoint_specific_greedy_prefix": True,
            "same_prefix_within_checkpoint_branches": True,
            "fixed_teacher": True,
            "random_masks_nested": True,
            "intervention": 0.01,
        },
        "bootstrap": {"unit": "sample_id", "draws": BOOTSTRAP_DRAWS, "seed": SEED},
        "pooled_step0_to_step3072_percent_change": float(
            100.0 * (summary["F_pooled"].iloc[-1] / summary["F_pooled"].iloc[0] - 1.0)
        ),
        "trajectory_mean_step0_to_step3072_percent_change": float(
            100.0
            * (summary["F_trajectory_mean"].iloc[-1] / summary["F_trajectory_mean"].iloc[0] - 1.0)
        ),
        "manifests": [
            {
                "step": manifest["checkpoint_step"],
                "adapter_path": manifest["adapter_path"],
                "adapter_sha256": manifest["adapter_sha256"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "slurm_job_id": manifest.get("slurm_job_id"),
            }
            for manifest in manifests
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(endpoint, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(endpoint, indent=2))


if __name__ == "__main__":
    main()
