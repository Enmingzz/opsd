#!/usr/bin/env python3
"""Measure token-level dependence between Budget Advantage and projection mass P."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args()


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
    }


def main() -> None:
    args = parse_args()
    ba_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    within_spearman: list[float] = []
    overlaps: list[float] = []
    jaccards: list[float] = []
    p_mass_selected_by_ba = 0.0
    positive_p_mass = 0.0
    ba_mass_selected_by_p = 0.0
    ba_mass = 0.0

    for line in args.scores.read_text().splitlines():
        if not line.strip():
            continue
        metrics = json.loads(line)["metrics"]
        ba = np.asarray(metrics["student_distribution_budget_advantage"], dtype=np.float64)
        a = np.asarray(metrics["student_distribution_teacher_js_b"], dtype=np.float64)
        b = np.asarray(metrics["student_distribution_js"], dtype=np.float64)
        c = np.asarray(metrics["student_distribution_teacher_js_b_plus"], dtype=np.float64)
        projection = 0.5 * (a + b - c)
        ba_parts.append(ba)
        p_parts.append(projection)
        if np.ptp(ba) > 0.0 and np.ptp(projection) > 0.0:
            within_spearman.append(float(spearmanr(ba, projection).statistic))

        k = max(1, math.ceil(args.top_fraction * len(ba)))
        top_ba = np.argpartition(ba, -k)[-k:]
        top_p = np.argpartition(projection, -k)[-k:]
        intersection = len(set(top_ba) & set(top_p))
        overlaps.append(intersection / k)
        jaccards.append(intersection / (2 * k - intersection))
        positive_p = np.clip(projection, 0.0, None)
        p_mass_selected_by_ba += float(positive_p[top_ba].sum())
        positive_p_mass += float(positive_p.sum())
        ba_mass_selected_by_p += float(ba[top_p].sum())
        ba_mass += float(ba.sum())

    ba = np.concatenate(ba_parts)
    projection = np.concatenate(p_parts)
    ba_positive = ba > 0.0
    p_positive = projection > 0.0
    both_positive = ba_positive & p_positive
    summary = {
        "definitions": {
            "BA": "max(log p_bplus(y_t) - log p_b(y_t), 0)",
            "P": "(JSD(q,p_b) + JSD(p_b,p_bplus) - JSD(q,p_bplus)) / 2",
        },
        "token_count": int(len(ba)),
        "all_tokens": correlation(ba, projection),
        "ba_positive_tokens": {
            "count": int(ba_positive.sum()),
            **correlation(ba[ba_positive], projection[ba_positive]),
        },
        "both_positive_tokens": {
            "count": int(both_positive.sum()),
            **correlation(ba[both_positive], projection[both_positive]),
        },
        "positive_P_transform_all_tokens_spearman": float(
            spearmanr(ba, np.clip(projection, 0.0, None)).statistic
        ),
        "within_trajectory_spearman_mean": float(np.mean(within_spearman)),
        "within_trajectory_spearman_median": float(np.median(within_spearman)),
        "top_fraction": args.top_fraction,
        "top_set_overlap_mean": float(np.mean(overlaps)),
        "top_set_overlap_median": float(np.median(overlaps)),
        "top_set_jaccard_mean": float(np.mean(jaccards)),
        "random_expected_overlap": args.top_fraction,
        "top_ba_positive_P_mass_capture_pooled": p_mass_selected_by_ba / positive_p_mass,
        "top_P_BA_mass_capture_pooled": ba_mass_selected_by_p / ba_mass,
        "positive_ba_token_fraction": float(ba_positive.mean()),
        "positive_P_token_fraction": float(p_positive.mean()),
        "P_positive_given_BA_positive": float(both_positive.sum() / ba_positive.sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
