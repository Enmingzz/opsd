#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config.json"
PRIMARY_METRICS = (
    "kl_r025_official_to_r020",
    "kl_base_full_to_r020",
    "kl_base_full_to_r025_official",
)
ALL_METRICS = PRIMARY_METRICS + (
    "js_r020_r025_official",
    "kl_r025_nested_to_r020",
    "js_r020_r025_nested",
    "kl_base_full_to_r025_nested",
)
TRIM_FRACTION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate checkpoint-wise KL trajectory outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def save_token_table(tokens: pd.DataFrame, analysis: Path) -> dict[str, str]:
    parquet_path = analysis / "per_token_metrics.parquet"
    try:
        tokens.to_parquet(parquet_path, index=False)
        return {"format": "parquet", "path": str(parquet_path)}
    except ImportError as error:
        csv_path = analysis / "per_token_metrics.csv.gz"
        tokens.to_csv(csv_path, index=False, compression="gzip")
        metadata = {
            "format": "csv.gz",
            "path": str(csv_path),
            "parquet_unavailable_reason": str(error),
        }
        atomic_write_json(analysis / "per_token_metrics_format.json", metadata)
        return metadata


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        value = float(values.mean()) if values.size else math.nan
        return value, value
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        means[index] = values[rng.integers(0, values.size, size=values.size)].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def symmetric_trimmed_mean(values: np.ndarray, fraction: float) -> tuple[float, int]:
    """Return a sample-level mean after removing `fraction` from each tail."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    trim_count = int(math.floor(values.size * float(fraction)))
    if trim_count == 0:
        return float(values.mean()), 0
    if 2 * trim_count >= values.size:
        raise ValueError(
            f"Cannot trim {trim_count} observations from each tail of {values.size} values"
        )
    return float(values[trim_count:-trim_count].mean()), trim_count


def load_rows(root: Path, cfg: dict[str, Any], allow_incomplete: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sample_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    expected_ids = {
        str(json.loads(line)["sample_id"])
        for line in Path(cfg["samples"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for checkpoint in cfg["checkpoint_steps"]:
        label = str(checkpoint["label"])
        step = int(checkpoint["step"])
        directory = root / label / "samples"
        files = sorted(directory.glob("*.json")) if directory.is_dir() else []
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        found_ids = {str(payload["sample_id"]) for payload in payloads}
        missing = sorted(expected_ids - found_ids)
        extra = sorted(found_ids - expected_ids)
        if missing:
            problems.append(f"{label}: missing {len(missing)} samples")
        if extra:
            problems.append(f"{label}: has {len(extra)} unexpected samples")
        for payload in payloads:
            generated_text = str(payload.get("generated_text", ""))
            generated_token_count = int(payload["generated_token_count"])
            sample_row: dict[str, Any] = {
                "checkpoint_label": label,
                "checkpoint_step": step,
                "sample_id": str(payload["sample_id"]),
                "generated_token_count": generated_token_count,
                "token_trimmed_each_tail_count": int(
                    math.floor(generated_token_count * TRIM_FRACTION)
                ),
                "hit_max_new_tokens": bool(payload["hit_max_new_tokens"]),
                "r020_visual_tokens": int(payload["visual_tokens"]["r020_official"]),
                "r025_visual_tokens": int(payload["visual_tokens"]["r025_official"]),
                "low_retention_ratio": float(
                    payload.get("ratio_pair", {}).get(
                        "low_retention_ratio", cfg["rollout_retention_ratio"]
                    )
                ),
                "high_retention_ratio": float(
                    payload.get("ratio_pair", {}).get(
                        "high_retention_ratio", cfg["comparison_retention_ratio"]
                    )
                ),
                "low_visual_tokens": int(
                    payload["visual_tokens"].get(
                        "low_official", payload["visual_tokens"]["r020_official"]
                    )
                ),
                "high_visual_tokens": int(
                    payload["visual_tokens"].get(
                        "high_official", payload["visual_tokens"]["r025_official"]
                    )
                ),
                "fixed_context_equivalent": bool(payload["official_fixed_r020_equivalence"]["allclose"]),
                "elapsed_seconds": float(payload["timings"]["total_seconds"]),
                "peak_gpu_allocated_gib": float(payload.get("peak_gpu_allocated_gib", math.nan)),
                "format_complete": all(
                    tag in generated_text
                    for tag in ("<think>", "</think>", "<answer>", "</answer>")
                ),
            }
            for metric in ALL_METRICS:
                values = np.asarray(payload["metrics"][metric], dtype=np.float64)
                sample_row[metric] = float(np.mean(values))
                sample_row[f"{metric}_token_trimmed_mean_5pct"] = symmetric_trimmed_mean(
                    values, TRIM_FRACTION
                )[0]
            sample_rows.append(sample_row)
            for position, token_id in enumerate(payload["generated_token_ids"]):
                token_row: dict[str, Any] = {
                    "checkpoint_label": label,
                    "checkpoint_step": step,
                    "sample_id": str(payload["sample_id"]),
                    "token_position": int(position),
                    "token_id": int(token_id),
                    "token_text": payload["generated_token_text"][position],
                }
                for metric in ALL_METRICS:
                    token_row[metric] = float(payload["metrics"][metric][position])
                token_rows.append(token_row)
    if problems and not allow_incomplete:
        raise RuntimeError("Incomplete experiment: " + "; ".join(problems))
    return pd.DataFrame(sample_rows), pd.DataFrame(token_rows), problems


def checkpoint_summary(samples: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    resamples = int(cfg["bootstrap_resamples"])
    seed = int(cfg["seed"])
    for (label, step), frame in samples.groupby(["checkpoint_label", "checkpoint_step"], sort=False):
        row: dict[str, Any] = {
            "checkpoint_label": label,
            "checkpoint_step": int(step),
            "n_samples": int(frame["sample_id"].nunique()),
            "mean_generated_tokens": float(frame["generated_token_count"].mean()),
            "truncation_rate": float(frame["hit_max_new_tokens"].mean()),
            "mean_elapsed_seconds_per_sample": float(frame["elapsed_seconds"].mean()),
            "peak_gpu_allocated_gib": float(frame["peak_gpu_allocated_gib"].max()),
            "fixed_context_equivalence_rate": float(frame["fixed_context_equivalent"].mean()),
            "format_complete_rate": float(frame["format_complete"].mean()),
            "mean_token_trimmed_each_tail_count": float(
                frame["token_trimmed_each_tail_count"].mean()
            ),
            "min_token_trimmed_each_tail_count": int(
                frame["token_trimmed_each_tail_count"].min()
            ),
            "max_token_trimmed_each_tail_count": int(
                frame["token_trimmed_each_tail_count"].max()
            ),
        }
        for metric_index, metric in enumerate(ALL_METRICS):
            values = frame[metric].to_numpy(dtype=np.float64)
            token_counts = frame["generated_token_count"].to_numpy(dtype=np.float64)
            low, high = bootstrap_ci(values, resamples, seed + int(step) + metric_index)
            token_trimmed_values = frame[
                f"{metric}_token_trimmed_mean_5pct"
            ].to_numpy(dtype=np.float64)
            row[metric] = float(values.mean())
            row[f"{metric}_sample_balanced"] = float(values.mean())
            row[f"{metric}_token_pooled"] = float(np.average(values, weights=token_counts))
            row[f"{metric}_token_trimmed_mean_5pct"] = float(token_trimmed_values.mean())
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("checkpoint_step").reset_index(drop=True)


def ratio_percent_label(ratio: float) -> str:
    return f"{100.0 * float(ratio):g}"


def metric_display_labels(low_ratio: float, high_ratio: float) -> dict[str, str]:
    low_label = ratio_percent_label(low_ratio)
    high_label = ratio_percent_label(high_ratio)
    return {
        "kl_r025_official_to_r020": rf"$KL(p_{{{high_label}}}\,\Vert\,p_{{{low_label}}})$",
        "kl_base_full_to_r020": rf"$KL(p_{{base,full}}\,\Vert\,p_{{{low_label}}})$",
        "kl_base_full_to_r025_official": rf"$KL(p_{{base,full}}\,\Vert\,p_{{{high_label}}})$",
    }


def save_main_plot(
    summary: pd.DataFrame,
    figures: Path,
    method_display_name: str,
    low_ratio: float,
    high_ratio: float,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    labels = metric_display_labels(low_ratio, high_ratio)
    low_label = ratio_percent_label(low_ratio)
    high_label = ratio_percent_label(high_ratio)
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    x = summary["checkpoint_step"].to_numpy()
    for metric in PRIMARY_METRICS:
        y = summary[metric].to_numpy()
        ci_low = summary[f"{metric}_ci_low"].to_numpy()
        ci_high = summary[f"{metric}_ci_high"].to_numpy()
        line = axis.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=labels[metric])[0]
        axis.fill_between(x, ci_low, ci_high, color=line.get_color(), alpha=0.14, linewidth=0)
    axis.set_xlabel("Training examples seen")
    axis.set_ylabel("Exact token-level KL (sample-balanced mean)")
    axis.set_title(
        f"LLM-only {method_display_name}: {low_label}% to {high_label}% budget divergence"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"primary_kl_trajectory.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    x = summary["checkpoint_step"].to_numpy()
    for metric in PRIMARY_METRICS:
        axis.plot(
            x,
            summary[f"{metric}_token_trimmed_mean_5pct"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=labels[metric],
        )
    axis.set_xlabel("Training examples seen")
    axis.set_ylabel("Exact KL (mean of within-sample 5% token-trimmed means)")
    axis.set_title(f"Diagnostic only: {method_display_name} token-trimmed KL trajectory")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"primary_kl_trajectory_trimmed_5pct.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for metric, label in (
        ("kl_r025_official_to_r020", f"Official {high_label}%"),
        ("kl_r025_nested_to_r020", f"Nested {high_label}% add-back"),
        ("js_r020_r025_official", f"Official {low_label}%/{high_label}% JS"),
        ("js_r020_r025_nested", f"Nested {low_label}%/{high_label}% JS"),
    ):
        axis.plot(x, summary[metric], marker="o", linewidth=1.6, markersize=4, label=label)
    axis.set_xlabel("Training examples seen")
    axis.set_ylabel("Divergence (sample-balanced mean)")
    axis.set_title(f"Official versus nested {high_label}% budget comparison")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"official_vs_nested_high_retention.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def trend_payload(summary: pd.DataFrame, metric: str) -> dict[str, Any]:
    values = summary[metric].to_numpy(dtype=np.float64)
    steps = summary["checkpoint_step"].to_numpy(dtype=np.float64)
    adjacent = np.diff(values)
    correlation = spearmanr(steps, values).statistic if len(values) > 1 else math.nan
    initial = float(values[0])
    final = float(values[-1])
    return {
        "initial": initial,
        "final": final,
        "absolute_change": final - initial,
        "relative_reduction_percent": float(100.0 * (initial - final) / initial) if initial > 0 else None,
        "adjacent_decreases": int((adjacent < 0).sum()),
        "adjacent_comparisons": int(adjacent.size),
        "strictly_monotonic_decrease": bool(np.all(adjacent < 0)),
        "spearman_step_vs_metric": float(correlation) if math.isfinite(float(correlation)) else None,
    }


def paired_endpoint_change(samples: pd.DataFrame, metric: str, cfg: dict[str, Any]) -> dict[str, Any]:
    initial_step = min(int(item["step"]) for item in cfg["checkpoint_steps"])
    final_step = max(int(item["step"]) for item in cfg["checkpoint_steps"])
    endpoint = samples[samples["checkpoint_step"].isin((initial_step, final_step))]
    pivot = endpoint.pivot(index="sample_id", columns="checkpoint_step", values=metric).dropna()
    if initial_step not in pivot or final_step not in pivot:
        return {"n_matched": 0}
    differences = (pivot[final_step] - pivot[initial_step]).to_numpy(dtype=np.float64)
    low, high = bootstrap_ci(
        differences,
        int(cfg["bootstrap_resamples"]),
        int(cfg["seed"]) + 1701,
    )
    statistic, p_value = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return {
        "n_matched": int(differences.size),
        "mean_final_minus_initial": float(differences.mean()),
        "median_final_minus_initial": float(np.median(differences)),
        "mean_change_ci_low": low,
        "mean_change_ci_high": high,
        "fraction_samples_decreased": float((differences < 0).mean()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_two_sided_p": float(p_value),
    }


def build_report(
    method_display_name: str,
    low_ratio: float,
    high_ratio: float,
    summary: pd.DataFrame,
    format_complete_summary: pd.DataFrame,
    trends: dict[str, Any],
    trimmed_trends: dict[str, Any],
    format_complete_trends: dict[str, Any],
    paired_changes: dict[str, Any],
    quality_checks: dict[str, Any],
    problems: list[str],
) -> str:
    low = ratio_percent_label(low_ratio)
    high = ratio_percent_label(high_ratio)
    display_names = {
        "kl_r025_official_to_r020": f"KL(p{high} || p{low})",
        "kl_base_full_to_r020": f"KL(pbase,full || p{low})",
        "kl_base_full_to_r025_official": f"KL(pbase,full || p{high})",
    }
    columns = ["checkpoint_step", *PRIMARY_METRICS, "n_samples", "mean_generated_tokens"]
    table = summary[columns].copy()
    for metric in PRIMARY_METRICS:
        table[metric] = table[metric].map(lambda value: f"{value:.6f}")
    table = table.rename(columns=display_names)
    trimmed_columns = [
        "checkpoint_step",
        "n_samples",
        "mean_token_trimmed_each_tail_count",
    ]
    trimmed_table = summary[trimmed_columns].copy()
    for metric in PRIMARY_METRICS:
        trimmed_table[metric] = summary[f"{metric}_token_trimmed_mean_5pct"].map(
            lambda value: f"{value:.6f}"
        )
    trimmed_table = trimmed_table[
        [
            "checkpoint_step",
            *PRIMARY_METRICS,
            "n_samples",
            "mean_token_trimmed_each_tail_count",
        ]
    ]
    trimmed_table = trimmed_table.rename(columns=display_names)
    lines = [
        f"# {method_display_name} LLM-only Checkpoint KL Trajectory",
        "",
        "## Protocol",
        "",
        f"Each checkpoint generates one greedy prefix under official VisionZip {low}% retention. The same token IDs are then",
        f"teacher-forced under {low}%, official {high}%, nested {high}%, and the fixed adapter-disabled base model with",
        "full visual tokens. The base teacher is identical at every checkpoint; EMA checkpoints are not used.",
        "KL is exact over the full vocabulary and averaged first within sample, then across the 100 samples.",
        "The checkpoint CSV also records a token-pooled mean for each metric; the plots and inferential tests use",
        "the sample-balanced mean so long rollouts do not receive disproportionate weight.",
        "",
        "## Primary results",
        "",
        table.to_markdown(index=False),
        "",
        "## Optional token-trimmed sensitivity diagnostic",
        "",
        "Within every sample and metric, the lowest and highest 5% of token-level KL values are removed",
        "independently. The remaining tokens are averaged within that sample, then the resulting 100 sample",
        "means are averaged with equal sample weight. No samples are removed. This diagnostic is retained for",
        "auditability but is not used for the primary trend or scientific conclusion.",
        "",
        trimmed_table.to_markdown(index=False),
        "",
        "## Primary aggregate-mean trend checks",
        "",
    ]
    for metric in PRIMARY_METRICS:
        item = trends[metric]
        lines.append(
            f"- `{metric}`: {item['initial']:.6f} -> {item['final']:.6f}; "
            f"relative reduction={item['relative_reduction_percent']:.2f}% if defined; "
            f"adjacent decreases={item['adjacent_decreases']}/{item['adjacent_comparisons']}; "
            f"Spearman(step, KL)={item['spearman_step_vs_metric']}."
        )
    if not format_complete_summary.empty:
        format_columns = ["checkpoint_step", *PRIMARY_METRICS, "n_samples"]
        format_table = format_complete_summary[format_columns].copy()
        for metric in PRIMARY_METRICS:
            format_table[metric] = format_table[metric].map(lambda value: f"{value:.6f}")
        format_table = format_table.rename(columns=display_names)
        lines.extend([
            "",
            "## Format-complete matched-sample sensitivity",
            "",
            "This sensitivity analysis keeps only sample IDs whose rollout contains complete `<think>` and",
            "`<answer>` tags at every checkpoint. Missing tags were not generation-length truncations.",
            "",
            format_table.to_markdown(index=False),
            "",
        ])
        for metric in PRIMARY_METRICS:
            item = format_complete_trends[metric]
            lines.append(
                f"- `{metric}`: {item['initial']:.6f} -> {item['final']:.6f}; "
                f"relative reduction={item['relative_reduction_percent']:.2f}%."
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Checkpoint-specific prefixes are intentionally on-policy, so a decline combines changes in the visited",
            "prefix distribution with changes in conditional distributions. The nested high-retention control isolates token",
            "add-back more cleanly than independently recomputed official high-retention contextual centers.",
            "",
            "## Completeness",
            "",
            "- " + ("All configured checkpoints and samples are complete." if not problems else "; ".join(problems)),
            f"- Records: {quality_checks['record_count']}/{quality_checks['expected_record_count']}.",
            f"- Generation truncations: {quality_checks['truncation_count']}.",
            f"- Fixed-context equivalence failures: {quality_checks['fixed_context_equivalence_failures']}.",
            f"- Complete `<think>/<answer>` format: {quality_checks['format_complete_count']}/"
            f"{quality_checks['record_count']} rollouts.",
            f"- Generated length: mean {quality_checks['mean_generated_tokens']:.2f}, "
            f"maximum {quality_checks['max_generated_tokens']} tokens.",
            f"- Maximum allocated GPU memory: {quality_checks['max_peak_gpu_allocated_gib']:.2f} GiB.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cfg = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    method_display_name = str(cfg.get("method_display_name", "OPSD"))
    low_ratio = float(cfg["rollout_retention_ratio"])
    high_ratio = float(cfg["comparison_retention_ratio"])
    root = (args.output_root or Path(cfg["output_root"])).expanduser().resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    samples, tokens, problems = load_rows(root, cfg, args.allow_incomplete)
    if samples.empty:
        raise RuntimeError(f"No completed sample outputs under {root}")
    summary = checkpoint_summary(samples, cfg)
    checkpoint_count = len(cfg["checkpoint_steps"])
    format_by_sample = samples.groupby("sample_id")["format_complete"].agg(["count", "all"])
    format_complete_ids = set(
        format_by_sample[
            (format_by_sample["count"] == checkpoint_count) & format_by_sample["all"]
        ].index.astype(str)
    )
    format_complete_samples = samples[samples["sample_id"].isin(format_complete_ids)].copy()
    format_complete_summary = (
        checkpoint_summary(format_complete_samples, cfg)
        if not format_complete_samples.empty
        else pd.DataFrame()
    )
    samples.to_csv(analysis / "per_sample_metrics.csv", index=False)
    token_table = save_token_table(tokens, analysis)
    summary.to_csv(analysis / "checkpoint_summary.csv", index=False)
    trimmed_export_columns = [
        "checkpoint_label",
        "checkpoint_step",
        "n_samples",
        "mean_token_trimmed_each_tail_count",
        "min_token_trimmed_each_tail_count",
        "max_token_trimmed_each_tail_count",
    ]
    for metric in PRIMARY_METRICS:
        trimmed_export_columns.extend(
            [metric, f"{metric}_token_trimmed_mean_5pct"]
        )
    summary[trimmed_export_columns].to_csv(
        analysis / "checkpoint_summary_trimmed_mean_5pct.csv", index=False
    )
    format_complete_summary.to_csv(
        analysis / "checkpoint_summary_format_complete_intersection.csv", index=False
    )
    save_main_plot(summary, analysis / "figures", method_display_name, low_ratio, high_ratio)
    trends = {metric: trend_payload(summary, metric) for metric in PRIMARY_METRICS}
    trimmed_trends = {
        metric: trend_payload(summary, f"{metric}_token_trimmed_mean_5pct")
        for metric in PRIMARY_METRICS
    }
    format_complete_trends = (
        {metric: trend_payload(format_complete_summary, metric) for metric in PRIMARY_METRICS}
        if not format_complete_summary.empty
        else {}
    )
    paired_changes = {metric: paired_endpoint_change(samples, metric, cfg) for metric in PRIMARY_METRICS}
    expected_record_count = len(cfg["checkpoint_steps"]) * len(
        {
            str(json.loads(line)["sample_id"])
            for line in Path(cfg["samples"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    quality_checks = {
        "record_count": int(len(samples)),
        "expected_record_count": int(expected_record_count),
        "truncation_count": int(samples["hit_max_new_tokens"].sum()),
        "fixed_context_equivalence_failures": int((~samples["fixed_context_equivalent"]).sum()),
        "format_complete_count": int(samples["format_complete"].sum()),
        "format_incomplete_count": int((~samples["format_complete"]).sum()),
        "mean_generated_tokens": float(samples["generated_token_count"].mean()),
        "median_generated_tokens": float(samples["generated_token_count"].median()),
        "max_generated_tokens": int(samples["generated_token_count"].max()),
        "max_peak_gpu_allocated_gib": float(samples["peak_gpu_allocated_gib"].max()),
    }
    payload = {
        "experiment_id": cfg["experiment_id"],
        "ratio_pair": {
            "low_retention_ratio": low_ratio,
            "high_retention_ratio": high_ratio,
        },
        "complete": not problems,
        "problems": problems,
        "checkpoints_completed": int(summary["checkpoint_label"].nunique()),
        "configured_checkpoints": len(cfg["checkpoint_steps"]),
        "primary_metrics": list(PRIMARY_METRICS),
        "token_table": token_table,
        "trends": trends,
        "trimmed_mean_5pct": {
            "definition": "within each sample and metric, remove the lowest and highest 5% of token-level values, average remaining tokens, then equally average the 100 sample means",
            "role": "diagnostic_only_not_used_for_primary_conclusion",
            "trends": trimmed_trends,
        },
        "paired_step0_to_final": paired_changes,
        "quality_checks": quality_checks,
        "format_complete_sensitivity": {
            "matched_sample_count": len(format_complete_ids),
            "definition": "complete think/answer tags at every configured checkpoint",
            "trends": format_complete_trends,
        },
    }
    atomic_write_json(analysis / "trend_summary.json", payload)
    atomic_write_text(
        analysis / "REPORT.md",
        build_report(
            method_display_name,
            low_ratio,
            high_ratio,
            summary,
            format_complete_summary,
            trends,
            trimmed_trends,
            format_complete_trends,
            paired_changes,
            quality_checks,
            problems,
        ),
    )
    print(
        summary[
            [
                "checkpoint_step",
                *PRIMARY_METRICS,
                *[f"{metric}_token_trimmed_mean_5pct" for metric in PRIMARY_METRICS],
                "n_samples",
            ]
        ].to_string(index=False)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
