#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUDGET_METRIC = "kl_r025_official_to_r020"
TEACHER_LOW_METRIC = "kl_base_full_to_r020"
TEACHER_HIGH_METRIC = "kl_base_full_to_r025_official"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all OPSD/SFT budget-gap ratio pairs.")
    parser.add_argument(
        "--config-manifest",
        type=Path,
        default=ROOT / "configs" / "ratio_pairs" / "manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ratio_pairs_mmstar_clean100_max1024" / "analysis",
    )
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def percent(value: float) -> str:
    return f"{100.0 * float(value):g}%"


def pair_label(low: float, high: float) -> str:
    return f"{percent(low)} to {percent(high)}"


def load_trajectories(manifest: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entry in manifest:
        summary_path = Path(entry["output_root"]) / "analysis" / "checkpoint_summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        frame = pd.read_csv(summary_path)
        if len(frame) != 11 or not frame["n_samples"].eq(100).all():
            raise ValueError(f"Incomplete checkpoint summary: {summary_path}")
        frame.insert(0, "method", str(entry["method"]).upper())
        frame.insert(0, "pair_label", pair_label(entry["low_retention_ratio"], entry["high_retention_ratio"]))
        frame.insert(0, "high_retention_ratio", float(entry["high_retention_ratio"]))
        frame.insert(0, "low_retention_ratio", float(entry["low_retention_ratio"]))
        frame.insert(0, "pair", str(entry["pair"]))
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 88:
        raise ValueError(f"Expected 88 checkpoint rows, found {len(result)}")
    return result


def endpoint_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (pair, method), frame in trajectories.groupby(["pair", "method"], sort=False):
        frame = frame.sort_values("checkpoint_step")
        low = float(frame.iloc[0]["low_retention_ratio"])
        high = float(frame.iloc[0]["high_retention_ratio"])
        for metric in (BUDGET_METRIC, TEACHER_LOW_METRIC, TEACHER_HIGH_METRIC):
            initial = float(frame.iloc[0][metric])
            final = float(frame.iloc[-1][metric])
            rows.append(
                {
                    "pair": pair,
                    "pair_label": pair_label(low, high),
                    "method": method,
                    "metric": metric,
                    "initial": initial,
                    "final": final,
                    "absolute_change": final - initial,
                    "relative_reduction_percent": 100.0 * (initial - final) / initial,
                }
            )
    return pd.DataFrame(rows)


def load_paired_results(manifest: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    entries_by_pair: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest:
        entries_by_pair.setdefault(str(entry["pair"]), []).append(entry)
    for pair, entries in entries_by_pair.items():
        first = entries[0]
        pair_root = Path(first["output_root"]).parent
        path = pair_root / "sft_vs_opsd" / "paired_final_aggregate_difference.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "pair_label", pair_label(first["low_retention_ratio"], first["high_retention_ratio"]))
        frame.insert(0, "high_retention_ratio", float(first["high_retention_ratio"]))
        frame.insert(0, "low_retention_ratio", float(first["low_retention_ratio"]))
        frame.insert(0, "pair", pair)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def aggregate_paired_budget_effect(
    manifest: list[dict[str, Any]],
    seed: int = 42,
    resamples: int = 10000,
    require_format_complete: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    entries_by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in manifest:
        entries_by_pair.setdefault(str(entry["pair"]), {})[str(entry["method"])] = entry
    merged: pd.DataFrame | None = None
    pair_columns: list[str] = []
    pair_effects: list[dict[str, Any]] = []
    for pair_index, (pair, methods) in enumerate(entries_by_pair.items()):
        frames: dict[str, pd.DataFrame] = {}
        for method in ("opsd", "sft"):
            path = Path(methods[method]["output_root"]) / "analysis" / "per_sample_metrics.csv"
            frame = pd.read_csv(path)
            final_step = int(frame["checkpoint_step"].max())
            frames[method] = frame[frame["checkpoint_step"] == final_step][
                ["sample_id", BUDGET_METRIC, "format_complete", "hit_max_new_tokens"]
            ].rename(
                columns={
                    BUDGET_METRIC: method,
                    "format_complete": f"{method}_format_complete",
                    "hit_max_new_tokens": f"{method}_truncated",
                }
            )
        paired = frames["opsd"].merge(frames["sft"], on="sample_id", validate="one_to_one")
        if len(paired) != 100:
            raise ValueError(f"Expected 100 final paired samples for {pair}, found {len(paired)}")
        if require_format_complete:
            paired = paired[
                paired["opsd_format_complete"]
                & paired["sft_format_complete"]
                & ~paired["opsd_truncated"]
                & ~paired["sft_truncated"]
            ].copy()
        column = f"{pair}_sft_minus_opsd"
        paired[column] = paired["sft"] - paired["opsd"]
        pair_columns.append(column)
        pair_values = paired[column].to_numpy(dtype=np.float64)
        pair_rng = np.random.default_rng(int(seed) + 1000 + pair_index)
        pair_bootstrap = np.empty(int(resamples), dtype=np.float64)
        for index in range(int(resamples)):
            pair_bootstrap[index] = pair_values[
                pair_rng.integers(0, pair_values.size, size=pair_values.size)
            ].mean()
        pair_low, pair_high = np.quantile(pair_bootstrap, [0.025, 0.975])
        pair_effects.append(
            {
                "pair": pair,
                "matched_sample_count": int(len(pair_values)),
                "mean_sft_minus_opsd": float(pair_values.mean()),
                "bootstrap_95_ci_low": float(pair_low),
                "bootstrap_95_ci_high": float(pair_high),
                "ci_excludes_zero_in_opsd_direction": bool(pair_low > 0.0),
            }
        )
        current = paired[["sample_id", column]]
        merged = current if merged is None else merged.merge(current, on="sample_id", validate="one_to_one")
    if merged is None or merged.empty:
        raise ValueError("Could not form a non-empty cross-pair sample matrix")
    if not require_format_complete and len(merged) != 100:
        raise ValueError(f"Expected 100 all-sample clusters, found {len(merged)}")
    merged["equal_pair_mean_sft_minus_opsd"] = merged[pair_columns].mean(axis=1)
    values = merged["equal_pair_mean_sft_minus_opsd"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        bootstrap[index] = values[rng.integers(0, values.size, size=values.size)].mean()
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    summary = {
        "definition": (
            "for each sample, average final SFT-minus-OPSD budget KL across the four predefined ratio pairs; bootstrap sample-ID clusters"
        ),
        "filter": (
            "both methods have complete think/answer tags and neither is truncated at the final checkpoint"
            if require_format_complete
            else "all predefined final-checkpoint samples"
        ),
        "positive_favors": "OPSD",
        "pair_count": len(pair_columns),
        "sample_cluster_count": len(values),
        "mean_sft_minus_opsd": float(values.mean()),
        "median_sample_cluster_effect": float(np.median(values)),
        "fraction_sample_clusters_positive": float((values > 0.0).mean()),
        "bootstrap_95_ci_low": float(low),
        "bootstrap_95_ci_high": float(high),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "ci_excludes_zero_in_opsd_direction": bool(low > 0.0),
        "pair_effects": pair_effects,
    }
    return summary, merged


def save_plots(trajectories: pd.DataFrame, output_dir: Path) -> None:
    pair_order = list(dict.fromkeys(trajectories["pair"].tolist()))
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True)
    for axis, pair in zip(axes.flat, pair_order, strict=True):
        pair_frame = trajectories[trajectories["pair"] == pair]
        title = str(pair_frame.iloc[0]["pair_label"])
        for method, frame in pair_frame.groupby("method", sort=False):
            frame = frame.sort_values("checkpoint_step")
            line = axis.plot(
                frame["checkpoint_step"],
                frame[BUDGET_METRIC],
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                label=method,
            )[0]
            axis.fill_between(
                frame["checkpoint_step"].to_numpy(),
                frame[f"{BUDGET_METRIC}_ci_low"].to_numpy(),
                frame[f"{BUDGET_METRIC}_ci_high"].to_numpy(),
                color=line.get_color(),
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(title)
        axis.set_xlabel("Training examples seen")
        axis.set_ylabel("KL(high-retention || low-retention)")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("VisionZip budget gap across LLM-only SFT and OPSD checkpoints")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"all_ratio_pairs_budget_gap.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True)
    for axis, pair in zip(axes.flat, pair_order, strict=True):
        pair_frame = trajectories[trajectories["pair"] == pair]
        title = str(pair_frame.iloc[0]["pair_label"])
        for method, frame in pair_frame.groupby("method", sort=False):
            frame = frame.sort_values("checkpoint_step")
            axis.plot(
                frame["checkpoint_step"],
                frame[TEACHER_LOW_METRIC],
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                label=method,
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Training examples seen")
        axis.set_ylabel("KL(full base || low-retention student)")
        axis.grid(alpha=0.25, which="both")
    axes.flat[0].legend(frameon=False)
    figure.suptitle("Alignment to the fixed full-token base teacher")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"all_ratio_pairs_full_teacher_alignment.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.config_manifest.expanduser().resolve().read_text(encoding="utf-8"))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectories(manifest)
    endpoints = endpoint_summary(trajectories)
    paired = load_paired_results(manifest)
    aggregate_effect, aggregate_effect_samples = aggregate_paired_budget_effect(manifest)
    format_effect, format_effect_samples = aggregate_paired_budget_effect(
        manifest, require_format_complete=True
    )
    trajectories.to_csv(output_dir / "all_ratio_pairs_checkpoint_summary.csv", index=False)
    endpoints.to_csv(output_dir / "all_ratio_pairs_endpoint_summary.csv", index=False)
    paired.to_csv(output_dir / "all_ratio_pairs_paired_final_summary.csv", index=False)
    aggregate_effect_samples.to_csv(
        output_dir / "aggregate_budget_gap_effect_by_sample.csv", index=False
    )
    atomic_write(
        output_dir / "aggregate_budget_gap_effect.json",
        json.dumps(aggregate_effect, indent=2, sort_keys=True) + "\n",
    )
    format_effect_samples.to_csv(
        output_dir / "aggregate_budget_gap_effect_format_complete_by_sample.csv", index=False
    )
    atomic_write(
        output_dir / "aggregate_budget_gap_effect_format_complete.json",
        json.dumps(format_effect, indent=2, sort_keys=True) + "\n",
    )
    save_plots(trajectories, output_dir)

    budget_endpoints = endpoints[endpoints["metric"] == BUDGET_METRIC].copy()
    budget_endpoints["initial"] = budget_endpoints["initial"].map(lambda value: f"{value:.6f}")
    budget_endpoints["final"] = budget_endpoints["final"].map(lambda value: f"{value:.6f}")
    budget_endpoints["relative_reduction_percent"] = budget_endpoints[
        "relative_reduction_percent"
    ].map(lambda value: f"{value:.2f}")
    paired_budget = paired[paired["metric"] == BUDGET_METRIC].copy()
    for column in ("mean_sft_minus_opsd", "bootstrap_95_ci_low", "bootstrap_95_ci_high"):
        paired_budget[column] = paired_budget[column].map(lambda value: f"{value:.6f}")
    report = [
        "# OPSD/SFT VisionZip Budget-Gap Ratio Sweep",
        "",
        "All four ratio pairs use the same manually audited MMStar Clean-100 cohort, greedy",
        "checkpoint-specific low-retention prefixes, `max_new_tokens=1024`, exact full-vocabulary KL,",
        "and the fixed adapter-disabled full-token base teacher. Means are sample-balanced: token KL is",
        "averaged within sample, then the 100 samples receive equal weight.",
        "",
        "## Budget-gap endpoints",
        "",
        budget_endpoints[
            ["pair_label", "method", "initial", "final", "relative_reduction_percent"]
        ].to_markdown(index=False),
        "",
        "## Final matched SFT-minus-OPSD comparison",
        "",
        "Positive differences favor OPSD. Confidence intervals use matched-sample bootstrap.",
        "",
        paired_budget[
            [
                "pair_label",
                "mean_sft_minus_opsd",
                "bootstrap_95_ci_low",
                "bootstrap_95_ci_high",
                "ci_excludes_zero",
            ]
        ].to_markdown(index=False),
        "",
        "## Joint four-pair test",
        "",
        "The four ratio-pair effects are averaged within each sample ID before bootstrapping the 100",
        "sample clusters. Positive SFT-minus-OPSD values favor OPSD.",
        "",
        f"- Mean SFT-minus-OPSD: {aggregate_effect['mean_sft_minus_opsd']:.6f}",
        f"- Clustered-bootstrap 95% CI: [{aggregate_effect['bootstrap_95_ci_low']:.6f}, "
        f"{aggregate_effect['bootstrap_95_ci_high']:.6f}]",
        f"- Fraction of sample clusters positive: {aggregate_effect['fraction_sample_clusters_positive']:.3f}",
        "",
        "## Format-complete sensitivity",
        "",
        "This sensitivity keeps only sample IDs for which both methods have complete `<think>` and",
        "`<answer>` tags and neither rollout is truncated at the final checkpoint for every ratio pair.",
        "",
        f"- Matched sample-ID intersection: {format_effect['sample_cluster_count']}",
        f"- Mean SFT-minus-OPSD: {format_effect['mean_sft_minus_opsd']:.6f}",
        f"- Clustered-bootstrap 95% CI: [{format_effect['bootstrap_95_ci_low']:.6f}, "
        f"{format_effect['bootstrap_95_ci_high']:.6f}]",
        "",
        "## Interpretation boundary",
        "",
        "Each method/checkpoint generates its own on-policy low-retention prefix. These curves therefore",
        "measure deployment behavior and combine changes in visited prefixes with conditional distributions;",
        "they are not a cross-method fixed-prefix causal comparison.",
        "",
    ]
    atomic_write(output_dir / "REPORT.md", "\n".join(report))
    print(budget_endpoints.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
