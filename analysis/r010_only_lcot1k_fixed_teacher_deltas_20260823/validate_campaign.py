#!/usr/bin/env python3
"""Independent integrity audit for the consolidated divergence campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
ROOT = HERE / "outputs/consolidated"
DELTAS = ("d01", "d02", "d05")
PAIRS = ("A", "B", "C")
METRICS = ("jsd", "forward_kl", "reverse_kl")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parquet_path = ROOT / "per_token_metrics_all_steps.parquet"
    rollout_path = ROOT / "rollouts_all_steps.jsonl"
    score_path = ROOT / "scores_all_steps.jsonl"
    manifest_path = ROOT / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet_file = pq.ParquetFile(parquet_path)
    expected_metrics = [
        f"{delta}_{pair}_{metric}"
        for delta in DELTAS
        for pair in PAIRS
        for metric in METRICS
    ]
    expected_rows = int(manifest["token_rows"])
    if parquet_file.metadata.num_rows != expected_rows:
        raise ValueError("Parquet row count disagrees with campaign manifest")
    if any(column not in parquet_file.schema_arrow.names for column in expected_metrics):
        raise ValueError("Missing one or more metric columns")

    ranges: dict[str, dict[str, float]] = {}
    for column in expected_metrics:
        values = parquet_file.read(columns=[column]).column(0).to_numpy(zero_copy_only=False)
        if values.shape != (expected_rows,) or not bool(np.isfinite(values).all()):
            raise ValueError(f"Invalid values in {column}")
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < -1e-7:
            raise ValueError(f"Negative divergence in {column}: {minimum}")
        ranges[column] = {"min": minimum, "max": maximum}

    for metric in METRICS:
        reference = parquet_file.read(columns=[f"d01_A_{metric}"]).column(0).to_numpy()
        for delta in ("d02", "d05"):
            comparison = parquet_file.read(columns=[f"{delta}_A_{metric}"]).column(0).to_numpy()
            if not np.array_equal(reference, comparison):
                raise ValueError(f"A mismatch across interventions for {metric}")

    rollout_lines = count_lines(rollout_path)
    score_lines = count_lines(score_path)
    expected_cells = int(manifest["expected_checkpoint_sample_cells"])
    if rollout_lines != expected_cells or score_lines != expected_cells:
        raise ValueError("Consolidated JSONL line counts are incomplete")
    identities: set[tuple[int, str]] = set()
    with rollout_path.open("r", encoding="utf-8") as rollouts, score_path.open(
        "r", encoding="utf-8"
    ) as scores:
        for rollout_line, score_line in zip(rollouts, scores, strict=True):
            rollout = json.loads(rollout_line)
            score = json.loads(score_line)
            identity = (int(score["checkpoint_step"]), str(score["sample_id"]))
            if identity != (int(rollout["checkpoint_step"]), str(rollout["sample_id"])):
                raise ValueError(f"Rollout/score identity mismatch: {identity}")
            if score["generated_token_ids"] != rollout["generated_token_ids"]:
                raise ValueError(f"Rollout/score token mismatch: {identity}")
            identities.add(identity)
    if len(identities) != expected_cells:
        raise ValueError("Duplicate checkpoint/sample identities in consolidated JSONL")

    artifacts = [parquet_path, rollout_path, score_path, manifest_path]
    hashes = {path.name: sha256_file(path) for path in artifacts}
    (ROOT / "artifact_hashes.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8"
    )
    report = {
        "status": "passed",
        "checkpoint_sample_cells": expected_cells,
        "unique_checkpoint_sample_identities": len(identities),
        "token_rows": expected_rows,
        "parquet_columns": len(parquet_file.schema_arrow.names),
        "per_token_metric_columns": len(expected_metrics),
        "all_metrics_finite_nonnegative": True,
        "A_exact_across_interventions": True,
        "rollout_score_token_ids_exact": True,
        "rollout_jsonl_lines": rollout_lines,
        "score_jsonl_lines": score_lines,
        "metric_ranges": ranges,
        "artifact_sha256": hashes,
    }
    atomic_json(ROOT / "validation_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "metric_ranges"}, indent=2))


if __name__ == "__main__":
    main()
