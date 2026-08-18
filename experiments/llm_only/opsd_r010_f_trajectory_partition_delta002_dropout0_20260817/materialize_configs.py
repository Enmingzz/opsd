#!/usr/bin/env python3
"""Create the two isolated 10%->12% trajectory-partition configs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml


EXPERIMENT = Path(__file__).resolve().parent
SOURCE = EXPERIMENT.parent / "opsd_r010_f_partition_ablation_dropout0_20260817"
OUTPUT_ROOT = Path(
    "/scratch/enmingzz/outputs/llm_only/"
    "opsd_r010_f_trajectory_partition_delta002_dropout0_20260817"
)
VARIANTS = ("trajectory_top20", "trajectory_bottom80")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    manifest = {
        "status": "materialized",
        "student_ratio": 0.10,
        "probe_ratio": 0.12,
        "absolute_budget_delta": 0.02,
        "source_experiment": str(SOURCE),
        "output_root": str(OUTPUT_ROOT),
        "variants": {},
    }
    for variant in VARIANTS:
        source_path = SOURCE / "configs" / f"train_{variant}.yaml"
        cfg = copy.deepcopy(yaml.safe_load(source_path.read_text(encoding="utf-8")))
        cfg["experiment"].update(
            {
                "name": f"opsd_r010_{variant}_Fdelta002_dropout0_10240",
                "family": "r010_f_trajectory_partition_delta002_dropout0",
                "comparison_id": "r010_f_partition_delta002_sufficiency_20260817",
                "output_root": str(OUTPUT_ROOT),
                "metric_intervention_pair": "native_visionzip_r010_to_r012",
                "variant": variant,
            }
        )
        cfg["opsd"]["native_budget_weighting"]["budget_delta_mode"] = "absolute"
        cfg["opsd"]["native_budget_weighting"]["budget_delta"] = 0.02
        # The inherited center file was diagnostic metadata for the 7.5-point
        # global-F run and is not consumed by either hard partition objective.
        cfg["experiment"].pop("center_check_reference_path", None)
        cfg["experiment"].pop("center_check_reference_sha256", None)
        output = EXPERIMENT / "configs" / f"train_{variant}.yaml"
        atomic_text(output, yaml.safe_dump(cfg, sort_keys=False))
        manifest["variants"][variant] = {
            "source_config": str(source_path),
            "source_sha256": sha256(source_path),
            "config": str(output),
            "config_sha256": sha256(output),
            "run_dir": str(OUTPUT_ROOT / variant / "run"),
        }
    atomic_text(
        EXPERIMENT / "manifests" / "materialized_configs.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
