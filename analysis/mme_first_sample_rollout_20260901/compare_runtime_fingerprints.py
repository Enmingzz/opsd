#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_NAMES = (
    "pixel_values",
    "image_grid_thw",
    "input_ids",
    "attention_mask",
    "vision_attention_scores",
    "dominant_indices_ranked",
    "dominant_indices_model_order",
    "contextual_anchor_indices",
    "retained_indices_model_order",
    "dropped_indices_model_order",
    "contextual_merge_assignment",
    "contextual_merge_source_indices",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two exported runtime fingerprints.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def compare_array(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "shape_equal": left.shape == right.shape,
        "dtype_equal": left.dtype == right.dtype,
    }
    if left.shape != right.shape:
        result["exact_equal"] = False
        return result
    result["exact_equal"] = bool(np.array_equal(left, right))
    mismatch = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
    result["mismatch_count"] = int(mismatch.size)
    result["first_mismatch_flat_index"] = int(mismatch[0]) if mismatch.size else None
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        result["max_abs_difference"] = float(difference.max(initial=0.0))
        result["mean_abs_difference"] = float(difference.mean()) if difference.size else 0.0
    return result


def main() -> None:
    args = parse_args()
    left_summary = json.loads((args.left / "fingerprint.json").read_text())
    right_summary = json.loads((args.right / "fingerprint.json").read_text())
    arrays = {}
    for name in ARRAY_NAMES:
        left = np.load(args.left / f"{name}.npy", allow_pickle=False)
        right = np.load(args.right / f"{name}.npy", allow_pickle=False)
        arrays[name] = compare_array(left, right)

    left_retained = set(left_summary["visionzip"]["retained_indices_model_order"])
    right_retained = set(right_summary["visionzip"]["retained_indices_model_order"])
    union = left_retained | right_retained
    report = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "same_image_sha256": (
            left_summary["sample"]["image_sha256"]
            == right_summary["sample"]["image_sha256"]
        ),
        "same_model_revision": (
            left_summary["model"]["revision"] == right_summary["model"]["revision"]
        ),
        "source_sha256": {
            "left": left_summary["runtime"]["source_sha256"],
            "right": right_summary["runtime"]["source_sha256"],
        },
        "retained_set": {
            "left_count": len(left_retained),
            "right_count": len(right_retained),
            "intersection_count": len(left_retained & right_retained),
            "union_count": len(union),
            "jaccard": len(left_retained & right_retained) / len(union) if union else 1.0,
            "left_only": sorted(left_retained - right_retained),
            "right_only": sorted(right_retained - left_retained),
        },
        "raw_rollout_equal": (
            left_summary["rollout"]["raw_prediction"]
            == right_summary["rollout"]["raw_prediction"]
        ),
        "arrays": arrays,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

