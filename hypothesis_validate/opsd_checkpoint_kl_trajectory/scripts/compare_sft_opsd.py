#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    "kl_r025_official_to_r020",
    "kl_base_full_to_r020",
    "kl_base_full_to_r025_official",
)
def ratio_percent_label(ratio: float) -> str:
    return f"{100.0 * float(ratio):g}"


def metric_labels(low_ratio: float, high_ratio: float) -> dict[str, str]:
    low = ratio_percent_label(low_ratio)
    high = ratio_percent_label(high_ratio)
    return {
        "kl_r025_official_to_r020": rf"$KL(p_{{{high}}}\,\Vert\,p_{{{low}}})$",
        "kl_base_full_to_r020": rf"$KL(p_{{base,full}}\,\Vert\,p_{{{low}}})$",
        "kl_base_full_to_r025_official": rf"$KL(p_{{base,full}}\,\Vert\,p_{{{high}}})$",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="Compare matched SFT and OPSD KL trajectories.")
    parser.add_argument(
        "--opsd-summary",
        type=Path,
        default=root
        / "llm_only_opsd_mmstar_metric_noocr_clean100_max1024"
        / "analysis"
        / "checkpoint_summary.csv",
    )
    parser.add_argument(
        "--sft-summary",
        type=Path,
        default=root
        / "llm_only_sft_mmstar_metric_noocr_clean100_max1024"
        / "analysis"
        / "checkpoint_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "sft_vs_opsd_mmstar_metric_noocr_clean100_max1024",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-ratio", type=float, default=0.20)
    parser.add_argument("--high-ratio", type=float, default=0.25)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path.expanduser().resolve())
    required = {"checkpoint_step", "n_samples", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["checkpoint_step"].duplicated().any():
        raise ValueError(f"Duplicate checkpoint steps in {path}")
    frame = frame.sort_values("checkpoint_step").reset_index(drop=True)
    frame.insert(0, "method", method)
    return frame


def endpoint_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, method_frame in frame.groupby("method", sort=False):
        method_frame = method_frame.sort_values("checkpoint_step")
        for metric in METRICS:
            initial = float(method_frame.iloc[0][metric])
            final = float(method_frame.iloc[-1][metric])
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "initial": initial,
                    "final": final,
                    "absolute_change": final - initial,
                    "relative_reduction_percent": 100.0 * (initial - final) / initial,
                }
            )
    return rows


def gap_trajectory(opsd: pd.DataFrame, sft: pd.DataFrame) -> pd.DataFrame:
    metric = "kl_r025_official_to_r020"
    result = opsd[["checkpoint_step", metric]].rename(columns={metric: "opsd_gap"}).merge(
        sft[["checkpoint_step", metric]].rename(columns={metric: "sft_gap"}),
        on="checkpoint_step",
        validate="one_to_one",
    )
    initial = float(result.iloc[0]["opsd_gap"])
    result["opsd_reduction_from_step0_percent"] = 100.0 * (initial - result["opsd_gap"]) / initial
    result["sft_reduction_from_step0_percent"] = 100.0 * (initial - result["sft_gap"]) / initial
    result["sft_minus_opsd_gap"] = result["sft_gap"] - result["opsd_gap"]
    return result


def paired_final_bootstrap(
    opsd_summary_path: Path,
    sft_summary_path: Path,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    opsd_samples = pd.read_csv(opsd_summary_path.resolve().parent / "per_sample_metrics.csv")
    sft_samples = pd.read_csv(sft_summary_path.resolve().parent / "per_sample_metrics.csv")
    final_step = int(min(opsd_samples["checkpoint_step"].max(), sft_samples["checkpoint_step"].max()))
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        left = opsd_samples[opsd_samples["checkpoint_step"] == final_step][
            ["sample_id", metric]
        ].rename(columns={metric: "opsd"})
        right = sft_samples[sft_samples["checkpoint_step"] == final_step][
            ["sample_id", metric]
        ].rename(columns={metric: "sft"})
        paired = left.merge(right, on="sample_id", validate="one_to_one")
        if len(paired) != 100:
            raise ValueError(f"Expected 100 paired final samples for {metric}, found {len(paired)}")
        differences = (paired["sft"] - paired["opsd"]).to_numpy(dtype=np.float64)
        bootstrap_means = np.empty(int(resamples), dtype=np.float64)
        for index in range(int(resamples)):
            bootstrap_means[index] = differences[
                rng.integers(0, differences.size, size=differences.size)
            ].mean()
        low, high = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append(
            {
                "checkpoint_step": final_step,
                "metric": metric,
                "n_paired_samples": int(differences.size),
                "mean_sft_minus_opsd": float(differences.mean()),
                "bootstrap_95_ci_low": float(low),
                "bootstrap_95_ci_high": float(high),
                "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
            }
        )
    return pd.DataFrame(rows)


def make_plot(
    combined: pd.DataFrame,
    output_dir: Path,
    low_ratio: float,
    high_ratio: float,
) -> None:
    labels = metric_labels(low_ratio, high_ratio)
    low = ratio_percent_label(low_ratio)
    high = ratio_percent_label(high_ratio)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for axis, metric in zip(axes, METRICS, strict=True):
        for method, method_frame in combined.groupby("method", sort=False):
            axis.plot(
                method_frame["checkpoint_step"],
                method_frame[metric],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=method,
            )
        axis.set_title(labels[metric])
        axis.set_xlabel("Training examples seen")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Exact KL (sample-balanced mean)")
    axes[-1].legend(frameon=False)
    figure.suptitle("Matched LLM-only SFT vs OPSD checkpoint trajectories", y=1.02)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"sft_vs_opsd_kl_trajectory.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    metric = "kl_r025_official_to_r020"
    for method, method_frame in combined.groupby("method", sort=False):
        axis.plot(
            method_frame["checkpoint_step"],
            method_frame[metric],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=method,
        )
    axis.set_xlabel("Training examples seen")
    axis.set_ylabel("Exact KL (ordinary sample-balanced mean)")
    axis.set_title(rf"VisionZip budget gap: $KL(p_{{{high}}}\,\Vert\,p_{{{low}}})$")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"sft_vs_opsd_budget_gap_mean.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for axis, metric in zip(axes, METRICS[1:], strict=True):
        for method, method_frame in combined.groupby("method", sort=False):
            axis.plot(
                method_frame["checkpoint_step"],
                method_frame[metric],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=method,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Training examples seen")
        axis.set_title(labels[metric])
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Exact KL (ordinary sample-balanced mean, log scale)")
    axes[-1].legend(frameon=False)
    figure.suptitle("Alignment to the fixed full-token base teacher", y=1.02)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"sft_vs_opsd_full_teacher_alignment_mean.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    opsd = load(args.opsd_summary, "OPSD")
    sft = load(args.sft_summary, "SFT")
    if opsd["checkpoint_step"].tolist() != sft["checkpoint_step"].tolist():
        raise ValueError("SFT and OPSD checkpoint steps do not match")
    if not (opsd["n_samples"].eq(100).all() and sft["n_samples"].eq(100).all()):
        raise ValueError("Expected 100 complete samples at every checkpoint")

    combined = pd.concat([opsd, sft], ignore_index=True)
    combined.to_csv(output_dir / "sft_vs_opsd_checkpoint_summary.csv", index=False)
    endpoints = pd.DataFrame(endpoint_rows(combined))
    endpoints.to_csv(output_dir / "sft_vs_opsd_endpoint_summary.csv", index=False)
    gap = gap_trajectory(opsd, sft)
    gap.to_csv(output_dir / "budget_gap_all_steps.csv", index=False)
    paired_final = paired_final_bootstrap(
        args.opsd_summary,
        args.sft_summary,
        args.bootstrap_resamples,
        args.seed,
    )
    paired_final.to_csv(output_dir / "paired_final_aggregate_difference.csv", index=False)

    step0_deltas = {
        metric: float(sft.iloc[0][metric] - opsd.iloc[0][metric]) for metric in METRICS
    }
    step0_equal = all(np.isclose(value, 0.0, rtol=0.0, atol=1e-12) for value in step0_deltas.values())
    make_plot(combined, output_dir, args.low_ratio, args.high_ratio)
    low = ratio_percent_label(args.low_ratio)
    high = ratio_percent_label(args.high_ratio)

    endpoint_table = endpoints.copy()
    for column in ("initial", "final", "absolute_change"):
        endpoint_table[column] = endpoint_table[column].map(lambda value: f"{value:.6f}")
    endpoint_table["relative_reduction_percent"] = endpoint_table[
        "relative_reduction_percent"
    ].map(lambda value: f"{value:.2f}")
    gap_table = gap.copy()
    for column in ("opsd_gap", "sft_gap", "sft_minus_opsd_gap"):
        gap_table[column] = gap_table[column].map(lambda value: f"{value:.6f}")
    for column in ("opsd_reduction_from_step0_percent", "sft_reduction_from_step0_percent"):
        gap_table[column] = gap_table[column].map(lambda value: f"{value:.2f}")
    paired_table = paired_final.copy()
    for column in ("mean_sft_minus_opsd", "bootstrap_95_ci_low", "bootstrap_95_ci_high"):
        paired_table[column] = paired_table[column].map(lambda value: f"{value:.6f}")
    gap_test = paired_final[
        paired_final["metric"] == "kl_r025_official_to_r020"
    ].iloc[0]
    gap_direction_favors_opsd = bool(gap_test["mean_sft_minus_opsd"] > 0.0)
    gap_supported = bool(gap_direction_favors_opsd and gap_test["bootstrap_95_ci_low"] > 0.0)
    gap_verdict = (
        "SUPPORTED"
        if gap_supported
        else "DIRECTIONALLY_FAVORS_OPSD_BUT_INCONCLUSIVE"
        if gap_direction_favors_opsd
        else "NOT_SUPPORTED"
    )
    if gap_supported:
        gap_description = (
            "OPSD has the lower final aggregate budget-gap mean and the matched-sample "
            "bootstrap confidence interval excludes zero in its favor."
        )
    elif gap_direction_favors_opsd:
        gap_description = (
            "OPSD has the lower final aggregate budget-gap mean, but the matched-sample "
            "bootstrap confidence interval crosses zero."
        )
    else:
        gap_description = (
            "OPSD does not have a lower final aggregate budget-gap mean than SFT; the proposed "
            "budget-gap advantage is not supported for this pair."
        )
    teacher_tests = paired_final[
        paired_final["metric"].isin(METRICS[1:])
    ]
    teacher_alignment_supported = bool(
        len(teacher_tests) == 2
        and (teacher_tests["mean_sft_minus_opsd"] > 0.0).all()
        and (teacher_tests["bootstrap_95_ci_low"] > 0.0).all()
    )
    teacher_alignment_description = (
        "SUPPORTED for this cohort: both final SFT-minus-OPSD full-teacher KL confidence intervals are positive."
        if teacher_alignment_supported
        else "NOT jointly supported: at least one final full-teacher KL comparison does not significantly favor OPSD."
    )
    report = [
        f"# SFT vs OPSD {low}%/{high}% KL Trajectory",
        "",
        "Both methods use the same manually audited MMStar Clean-100 cohort, clean-Armen environment,",
        f"VisionZip {low}% rollout, {high}% comparison, fixed full-token base teacher, greedy decoding, and",
        f"`max_new_tokens=1024`. Each checkpoint is evaluated on its own on-policy {low}% prefix, so this",
        "comparison measures deployment behavior and does not isolate parameters under one cross-method prefix.",
        "",
        f"Step-0 aggregate means identical within 1e-12: **{step0_equal}**.",
        "",
        "## Result",
        "",
        f"- {low}%-to-{high}% gap claim: **{gap_verdict}**. {gap_description}",
        f"- Full-teacher alignment: **{teacher_alignment_description}**",
        "- These conclusions concern distributional alignment on on-policy prefixes, not official",
        "  MMStar accuracy.",
        "",
        "## Endpoint summary",
        "",
        endpoint_table.to_markdown(index=False),
        "",
        f"## {low}%-to-{high}% gap at every checkpoint",
        "",
        gap_table.to_markdown(index=False),
        "",
        "## Paired final aggregate differences",
        "",
        "Differences are SFT minus OPSD. Positive values favor OPSD. Confidence intervals bootstrap",
        "matched sample IDs and estimate uncertainty of the aggregate mean, not the fraction of samples.",
        "",
        paired_table.to_markdown(index=False),
        "",
        "## Interpretation rule",
        "",
        f"The claim that OPSD recovers the {low}%-to-{high}% gap better than SFT is supported only if",
        f"`KL(p{high} || p{low})` falls more from the shared step 0 under OPSD than under SFT. The two",
        "full-teacher KL curves diagnose alignment to the fixed base teacher but are distinct claims.",
        "",
    ]
    atomic_write(output_dir / "REPORT.md", "\n".join(report))
    atomic_write(
        output_dir / "comparison_summary.json",
        json.dumps(
            {
                "step0_equal_atol_1e-12": step0_equal,
                "step0_sft_minus_opsd": step0_deltas,
                "methods": ["SFT", "OPSD"],
                "ratio_pair": {
                    "low_retention_ratio": float(args.low_ratio),
                    "high_retention_ratio": float(args.high_ratio),
                },
                "primary_aggregation": "sample-balanced mean of per-sample token means",
                "prefix_policy": f"each method/checkpoint generates its own on-policy {low}% prefix",
                "paired_final_aggregate_difference": paired_final.to_dict(orient="records"),
                "bootstrap_resamples": int(args.bootstrap_resamples),
                "bootstrap_seed": int(args.seed),
                "budget_gap_verdict": gap_verdict,
                "teacher_alignment_supported": teacher_alignment_supported,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(endpoint_table.to_string(index=False))
    print(f"step0_equal={step0_equal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
