#!/usr/bin/env python3
"""Analyze entropy diagnostics for the r010->r012 token-selection probe."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("step0_heldout50_topF_entropy_details.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = []
    for sample in payload["rows"]:
        for token in sample["all_token_details"]:
            if not token.get("eligible", False):
                continue
            row = dict(token)
            row["sample_id"] = str(sample["sample_id"])
            row["entropy_delta"] = row["entropy_b_plus"] - row["entropy_b"]
            row["entropy_gap_pre"] = abs(row["entropy_b"] - row["entropy_full"])
            row["entropy_gap_post"] = abs(row["entropy_b_plus"] - row["entropy_full"])
            row["entropy_gap_contraction"] = (
                row["entropy_gap_pre"] - row["entropy_gap_post"]
            )
            rows.append(row)
    return rows


def array(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def bootstrap_pooled_spearman(
    rows: list[dict], x_key: str, y_key: str, repeats: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
    sample_ids = np.asarray(sorted(by_sample))
    values = []
    for _ in range(repeats):
        selected_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        sampled = []
        for draw_index, sample_id in enumerate(selected_ids):
            for row in by_sample[sample_id]:
                copied = dict(row)
                copied["sample_id"] = f"{sample_id}:{draw_index}"
                sampled.append(copied)
        rho = spearmanr(array(sampled, x_key), array(sampled, y_key)).statistic
        if np.isfinite(rho):
            values.append(float(rho))
    return tuple(np.quantile(values, [0.025, 0.5, 0.975]))


def sample_bootstrap_mean(
    rows: list[dict], key: str, repeats: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(float(row[key]))
    macro = np.asarray([np.mean(values) for values in by_sample.values()])
    draws = np.asarray(
        [np.mean(rng.choice(macro, size=len(macro), replace=True)) for _ in range(repeats)]
    )
    return float(macro.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = load_rows(args.input)

    # Match the proposed training semantics: rank independently inside every
    # trajectory, retaining ceil(50%) by OPSD KL and then ceil(20%) by F.
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
    high_kl = []
    selected = []
    for sample_rows in by_sample.values():
        sample_high_kl = sorted(
            sample_rows, key=lambda row: row["opd_kl"], reverse=True
        )[: math.ceil(0.5 * len(sample_rows))]
        sample_selected = sorted(
            sample_high_kl,
            key=lambda row: row["projection_fraction"],
            reverse=True,
        )[: math.ceil(0.2 * len(sample_high_kl))]
        high_kl.extend(sample_high_kl)
        selected.extend(sample_selected)
    selected_ids = {id(row) for row in selected}
    omitted = [row for row in high_kl if id(row) not in selected_ids]

    correlations = {}
    for key in (
        "entropy_b",
        "entropy_delta",
        "entropy_gap_pre",
        "entropy_gap_contraction",
        "opd_kl",
        "A",
        "B",
        "C",
        "projection_mass",
    ):
        result = spearmanr(array(high_kl, "projection_fraction"), array(high_kl, key))
        ci = bootstrap_pooled_spearman(
            high_kl, "projection_fraction", key, args.bootstrap, rng
        )
        correlations[key] = {
            "rho": float(result.statistic),
            "sample_clustered_bootstrap_ci95": [ci[0], ci[2]],
        }

    contraction_predictors = {}
    for key in ("projection_fraction", "projection_mass", "A", "B", "opd_kl"):
        result = spearmanr(array(high_kl, key), array(high_kl, "entropy_gap_contraction"))
        ci = bootstrap_pooled_spearman(
            high_kl, key, "entropy_gap_contraction", args.bootstrap, rng
        )
        contraction_predictors[key] = {
            "rho": float(result.statistic),
            "sample_clustered_bootstrap_ci95": [ci[0], ci[2]],
        }

    quantile_edges = np.quantile(array(high_kl, "projection_fraction"), np.linspace(0, 1, 5))
    quartiles = []
    for index in range(4):
        low, high = quantile_edges[index : index + 2]
        group = [
            row
            for row in high_kl
            if row["projection_fraction"] >= low
            and (row["projection_fraction"] <= high if index == 3 else row["projection_fraction"] < high)
        ]
        mean, ci_low, ci_high = sample_bootstrap_mean(
            group, "entropy_gap_contraction", args.bootstrap, rng
        )
        quartiles.append(
            {
                "quartile": index + 1,
                "f_range": [float(low), float(high)],
                "token_count": len(group),
                "mean_f": float(array(group, "projection_fraction").mean()),
                "mean_entropy_gap_contraction": mean,
                "ci95": [ci_low, ci_high],
                "fraction_moving_toward_teacher_entropy": float(
                    np.mean(array(group, "entropy_gap_contraction") > 0)
                ),
            }
        )

    def group_summary(group: list[dict]) -> dict:
        return {
            "token_count": len(group),
            "mean_entropy_full": float(array(group, "entropy_full").mean()),
            "mean_entropy_b": float(array(group, "entropy_b").mean()),
            "mean_entropy_b_plus": float(array(group, "entropy_b_plus").mean()),
            "mean_entropy_delta": float(array(group, "entropy_delta").mean()),
            "mean_abs_entropy_delta": float(np.abs(array(group, "entropy_delta")).mean()),
            "mean_entropy_gap_pre": float(array(group, "entropy_gap_pre").mean()),
            "mean_entropy_gap_post": float(array(group, "entropy_gap_post").mean()),
            "mean_entropy_gap_contraction": float(
                array(group, "entropy_gap_contraction").mean()
            ),
            "fraction_moving_toward_teacher_entropy": float(
                np.mean(array(group, "entropy_gap_contraction") > 0)
            ),
            "fraction_budget_sharpens": float(np.mean(array(group, "entropy_delta") < 0)),
            "fraction_student_overconfident": float(
                np.mean(array(group, "entropy_b") < array(group, "entropy_full"))
            ),
        }

    summary = {
        "input": str(args.input),
        "sample_count": len({row["sample_id"] for row in rows}),
        "eligible_token_count": len(rows),
        "gate": {
            "scope": "within_each_trajectory",
            "high_kl_fraction": 0.5,
            "high_f_fraction_within_high_kl": 0.2,
            "rounding": "ceil",
        },
        "groups": {
            "all": group_summary(rows),
            "high_kl_top50": group_summary(high_kl),
            "high_kl_high_f_top20": group_summary(selected),
            "high_kl_lower_f_bottom80": group_summary(omitted),
        },
        "high_f_share_of_high_kl_net_entropy_gap_contraction": float(
            array(selected, "entropy_gap_contraction").sum()
            / array(high_kl, "entropy_gap_contraction").sum()
        ),
        "correlations_with_f_within_high_kl": correlations,
        "predictors_of_entropy_gap_contraction_within_high_kl": contraction_predictors,
        "f_quartiles_within_high_kl": quartiles,
    }
    summary_path = args.output_dir / "entropy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    hb = axes[0].hexbin(
        array(high_kl, "projection_fraction"),
        array(high_kl, "entropy_gap_contraction"),
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Projection fraction F")
    axes[0].set_ylabel("Entropy-gap contraction")
    axes[0].set_title("Budget moves confidence toward teacher")
    fig.colorbar(hb, ax=axes[0], label="Token count")

    qmeans = np.asarray([item["mean_entropy_gap_contraction"] for item in quartiles])
    qlow = np.asarray([item["ci95"][0] for item in quartiles])
    qhigh = np.asarray([item["ci95"][1] for item in quartiles])
    axes[1].errorbar(
        np.arange(1, 5),
        qmeans,
        yerr=np.vstack([qmeans - qlow, qhigh - qmeans]),
        marker="o",
        capsize=4,
        color="#1f6f8b",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(1, 5), ["Q1", "Q2", "Q3", "Q4"])
    axes[1].set_xlabel("F quartile within high-KL tokens")
    axes[1].set_ylabel("Mean entropy-gap contraction")
    axes[1].set_title("95% sample-bootstrap CI")

    labels = ["High F\n(top 20%)", "Lower F\n(bottom 80%)"]
    before = [
        summary["groups"]["high_kl_high_f_top20"]["mean_entropy_gap_pre"],
        summary["groups"]["high_kl_lower_f_bottom80"]["mean_entropy_gap_pre"],
    ]
    after = [
        summary["groups"]["high_kl_high_f_top20"]["mean_entropy_gap_post"],
        summary["groups"]["high_kl_lower_f_bottom80"]["mean_entropy_gap_post"],
    ]
    x = np.arange(2)
    width = 0.34
    axes[2].bar(x - width / 2, before, width, label="r010")
    axes[2].bar(x + width / 2, after, width, label="r012")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Mean |H(student) - H(teacher)|")
    axes[2].set_title("Scalar confidence mismatch")
    axes[2].legend(frameon=False)

    fig.suptitle("Entropy audit: step-0, r010 to r012, 50 held-out LLaVA-CoT samples")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"entropy_projection_audit.{suffix}", dpi=220)
    plt.close(fig)

    selected_summary = summary["groups"]["high_kl_high_f_top20"]
    omitted_summary = summary["groups"]["high_kl_lower_f_bottom80"]
    corr = correlations
    predictors = contraction_predictors
    report = f"""# Entropy Audit

This diagnostic uses the same 50 held-out trajectories at step 0 and the native
r010 to r012 VisionZip intervention. Entropy is computed exactly in FP32 over
the full vocabulary. The audited gate first keeps the top 50% of tokens by
forward OPSD KL within each trajectory, then selects the top 20% by projection
fraction `F` within that trajectory's high-KL subset.

## Main Findings

1. `F` is not a proxy for raw uncertainty. Within high-KL tokens,
   Spearman `rho(F, H(p_b)) = {corr['entropy_b']['rho']:.3f}` with a
   sample-clustered 95% bootstrap interval
   [{corr['entropy_b']['sample_clustered_bootstrap_ci95'][0]:.3f},
   {corr['entropy_b']['sample_clustered_bootstrap_ci95'][1]:.3f}].

2. `F` strongly tracks whether adding 2% visual budget moves the student's
   scalar confidence toward the full-token teacher. Spearman correlation with
   entropy-gap contraction is {corr['entropy_gap_contraction']['rho']:.3f}
   (95% sample-bootstrap interval
   [{corr['entropy_gap_contraction']['sample_clustered_bootstrap_ci95'][0]:.3f},
   {corr['entropy_gap_contraction']['sample_clustered_bootstrap_ci95'][1]:.3f}]).
   This is substantially stronger than raw teacher JSD `A`
   (`rho={predictors['A']['rho']:.3f}`), native budget JSD `B`
   (`rho={predictors['B']['rho']:.3f}`), or forward OPSD KL
   (`rho={predictors['opd_kl']['rho']:.3f}`) as predictors of the same
   entropy-gap contraction target.

3. In the high-KL/high-F group, mean entropy mismatch falls from
   {selected_summary['mean_entropy_gap_pre']:.4f} at r010 to
   {selected_summary['mean_entropy_gap_post']:.4f} at r012;
   {100 * selected_summary['fraction_moving_toward_teacher_entropy']:.1f}% of
   tokens move toward teacher entropy. In the omitted high-KL/lower-F group,
   the mismatch changes only from {omitted_summary['mean_entropy_gap_pre']:.4f}
   to {omitted_summary['mean_entropy_gap_post']:.4f}, and
   {100 * omitted_summary['fraction_moving_toward_teacher_entropy']:.1f}% move
   toward teacher entropy. Although it contains only 20% of the high-KL
   tokens, the high-F group accounts for
   {100 * summary['high_f_share_of_high_kl_net_entropy_gap_contraction']:.1f}%
   of the net entropy-gap contraction.

4. High `F` does not simply mean entropy decreases. Only
   {100 * selected_summary['fraction_budget_sharpens']:.1f}% of selected tokens
   sharpen under r012. The useful signal is direction toward teacher
   confidence: budget can either reduce under-confidence or correct
   over-confidence.

## Interpretation

Entropy supplies a useful interpretation of `F`: high `F` identifies tokens
whose teacher mismatch is budget-resolvable and whose confidence state is often
corrected by a small visual-budget increase. It does not establish token
importance and should not replace distributional JSD/KL. Conversely, low `F`
high-KL tokens mix genuinely persistent utilization gaps with anti-aligned or
noisy budget responses; entropy alone cannot separate those cases reliably.
"""
    (args.output_dir / "ENTROPY_AUDIT.md").write_text(report)
    print(summary_path)
    print(args.output_dir / "ENTROPY_AUDIT.md")
    print(args.output_dir / "entropy_projection_audit.pdf")


if __name__ == "__main__":
    main()
