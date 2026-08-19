#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC = "kl_r025_official_to_r020"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    outputs = EXPERIMENT_ROOT / "outputs"
    parser = argparse.ArgumentParser(description="Plot cumulative r020/r025 KL for SFT and OPSD.")
    parser.add_argument(
        "--opsd-token-metrics",
        type=Path,
        default=outputs
        / "llm_only_opsd_mmstar_metric_noocr_clean100_max1024"
        / "analysis"
        / "per_token_metrics.parquet",
    )
    parser.add_argument(
        "--sft-token-metrics",
        type=Path,
        default=outputs
        / "llm_only_sft_mmstar_metric_noocr_clean100_max1024"
        / "analysis"
        / "per_token_metrics.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=outputs / "sft_vs_opsd_mmstar_metric_noocr_clean100_max1024",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def summarize(method: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokens = pd.read_parquet(path.expanduser().resolve())
    required = {"checkpoint_step", "sample_id", "token_position", METRIC}
    missing = required - set(tokens.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    samples = (
        tokens.groupby(["checkpoint_step", "sample_id"], as_index=False)
        .agg(
            cumulative_budget_gap=(METRIC, "sum"),
            response_tokens=(METRIC, "size"),
            ordinary_token_mean=(METRIC, "mean"),
        )
    )
    checkpoints = (
        samples.groupby("checkpoint_step", as_index=False)
        .agg(
            mean_cumulative_budget_gap=("cumulative_budget_gap", "mean"),
            median_cumulative_budget_gap=("cumulative_budget_gap", "median"),
            mean_response_tokens=("response_tokens", "mean"),
            ordinary_sample_balanced_mean=("ordinary_token_mean", "mean"),
            n_samples=("sample_id", "nunique"),
        )
        .sort_values("checkpoint_step")
    )
    checkpoints.insert(0, "method", method)
    samples.insert(0, "method", method)
    return checkpoints, samples


def padded_sample_curves(path: Path, final_step: int, max_position: int) -> dict[str, np.ndarray]:
    tokens = pd.read_parquet(path.expanduser().resolve())
    tokens = tokens[tokens["checkpoint_step"] == final_step]
    curves: dict[str, np.ndarray] = {}
    for sample_id, sample in tokens.groupby("sample_id"):
        values = sample.sort_values("token_position")[METRIC].to_numpy(dtype=np.float64)
        cumulative = np.cumsum(values)
        padded = np.full(max_position, cumulative[-1], dtype=np.float64)
        padded[: cumulative.size] = cumulative
        curves[str(sample_id)] = padded
    return curves


def final_position_curve(method: str, path: Path, final_step: int, max_position: int) -> pd.DataFrame:
    curves = padded_sample_curves(path, final_step, max_position)
    matrix = np.stack(list(curves.values()))
    return pd.DataFrame(
        {
            "method": method,
            "generation_position": np.arange(1, max_position + 1),
            "mean_cumulative_budget_gap": matrix.mean(axis=0),
            "median_cumulative_budget_gap": np.median(matrix, axis=0),
            "n_samples": matrix.shape[0],
        }
    )


def paired_position_difference(
    opsd_path: Path,
    sft_path: Path,
    final_step: int,
    max_position: int,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    opsd = padded_sample_curves(opsd_path, final_step, max_position)
    sft = padded_sample_curves(sft_path, final_step, max_position)
    sample_ids = sorted(set(opsd) & set(sft))
    if len(sample_ids) != 100:
        raise ValueError(f"Expected 100 paired samples, found {len(sample_ids)}")
    differences = np.stack([sft[sample_id] - opsd[sample_id] for sample_id in sample_ids])
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty((int(resamples), max_position), dtype=np.float32)
    for index in range(int(resamples)):
        bootstrap[index] = differences[
            rng.integers(0, differences.shape[0], size=differences.shape[0])
        ].mean(axis=0)
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "generation_position": np.arange(1, max_position + 1),
            "mean_sft_minus_opsd": differences.mean(axis=0),
            "bootstrap_95_ci_low": low,
            "bootstrap_95_ci_high": high,
            "n_paired_samples": len(sample_ids),
        }
    )


def paired_final_ci(samples: pd.DataFrame, resamples: int, seed: int) -> dict[str, float | int]:
    final_step = int(samples["checkpoint_step"].max())
    final = samples[samples["checkpoint_step"] == final_step]
    opsd = final[final["method"] == "OPSD"][["sample_id", "cumulative_budget_gap"]].rename(
        columns={"cumulative_budget_gap": "opsd"}
    )
    sft = final[final["method"] == "SFT"][["sample_id", "cumulative_budget_gap"]].rename(
        columns={"cumulative_budget_gap": "sft"}
    )
    paired = opsd.merge(sft, on="sample_id", validate="one_to_one")
    differences = (paired["sft"] - paired["opsd"]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        bootstrap[index] = differences[
            rng.integers(0, differences.size, size=differences.size)
        ].mean()
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "checkpoint_step": final_step,
        "n_paired_samples": int(differences.size),
        "mean_sft_minus_opsd": float(differences.mean()),
        "bootstrap_95_ci_low": float(low),
        "bootstrap_95_ci_high": float(high),
    }


def save_plots(
    checkpoints: pd.DataFrame,
    positions: pd.DataFrame,
    position_difference: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for method, frame in checkpoints.groupby("method", sort=False):
        axis.plot(
            frame["checkpoint_step"],
            frame["mean_cumulative_budget_gap"],
            marker="o",
            linewidth=2,
            markersize=4.5,
            label=method,
        )
    axis.set_xlabel("Training examples seen")
    axis.set_ylabel(r"Mean per-response cumulative $KL(p_{25}\,\Vert\,p_{20})$")
    axis.set_title("Accumulated VisionZip budget gap across checkpoints")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"sft_vs_opsd_cumulative_gap_by_checkpoint.{suffix}", dpi=240)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for method, frame in positions.groupby("method", sort=False):
        axis.plot(
            frame["generation_position"],
            frame["mean_cumulative_budget_gap"],
            linewidth=2,
            label=method,
        )
    axis.set_xlabel("Generation position")
    axis.set_ylabel(r"Mean cumulative $KL(p_{25}\,\Vert\,p_{20})$")
    axis.set_title("Final checkpoint: accumulated budget gap during generation")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"sft_vs_opsd_cumulative_gap_by_position.{suffix}", dpi=240)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    x = position_difference["generation_position"].to_numpy()
    mean = position_difference["mean_sft_minus_opsd"].to_numpy()
    low = position_difference["bootstrap_95_ci_low"].to_numpy()
    high = position_difference["bootstrap_95_ci_high"].to_numpy()
    axis.plot(x, mean, linewidth=2, label="SFT minus OPSD")
    axis.fill_between(x, low, high, alpha=0.18, linewidth=0, label="95% paired bootstrap CI")
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Generation position")
    axis.set_ylabel(r"Cumulative gap difference: SFT $-$ OPSD")
    axis.set_title("Final checkpoint: where budget-gap accumulation differs")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"sft_minus_opsd_cumulative_gap_by_position_ci.{suffix}",
            dpi=240,
        )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"OPSD": args.opsd_token_metrics, "SFT": args.sft_token_metrics}
    checkpoint_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    max_position = 0
    for method, path in paths.items():
        checkpoints, samples = summarize(method, path)
        checkpoint_frames.append(checkpoints)
        sample_frames.append(samples)
        tokens = pd.read_parquet(path.expanduser().resolve(), columns=["checkpoint_step", "token_position"])
        final_step = int(tokens["checkpoint_step"].max())
        max_position = max(
            max_position,
            int(tokens[tokens["checkpoint_step"] == final_step]["token_position"].max()) + 1,
        )
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    samples = pd.concat(sample_frames, ignore_index=True)
    final_step = int(checkpoints["checkpoint_step"].max())
    positions = pd.concat(
        [
            final_position_curve(method, path, final_step, max_position)
            for method, path in paths.items()
        ],
        ignore_index=True,
    )
    position_difference = paired_position_difference(
        args.opsd_token_metrics,
        args.sft_token_metrics,
        final_step,
        max_position,
        args.bootstrap_resamples,
        args.seed,
    )
    paired = paired_final_ci(samples, args.bootstrap_resamples, args.seed)
    checkpoints.to_csv(output_dir / "cumulative_budget_gap_by_checkpoint.csv", index=False)
    positions.to_csv(output_dir / "cumulative_budget_gap_by_position.csv", index=False)
    position_difference.to_csv(
        output_dir / "cumulative_budget_gap_position_difference.csv", index=False
    )
    atomic_write(
        output_dir / "cumulative_budget_gap_summary.json",
        json.dumps(
            {
                "metric": METRIC,
                "definition": "sum token-level KL within each response, then equally average sample sums",
                "position_padding": "carry each sample's final cumulative value after EOS",
                "paired_final": paired,
                "bootstrap_resamples": int(args.bootstrap_resamples),
                "seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    save_plots(checkpoints, positions, position_difference, output_dir)
    print(checkpoints.to_string(index=False))
    print(json.dumps(paired, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
