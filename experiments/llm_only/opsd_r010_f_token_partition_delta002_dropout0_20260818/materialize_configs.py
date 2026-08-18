#!/usr/bin/env python3
"""Materialize fail-closed 2%-intervention token-partition configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "opsd_r010_f_partition_ablation_dropout0_20260817"
OUTPUT_ROOT = Path(
    "/scratch/enmingzz/outputs/llm_only/"
    "opsd_r010_f_token_partition_delta002_dropout0_20260818"
)


def materialize(selection: str) -> None:
    source = PARENT / "configs" / f"train_token_{selection}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    experiment = config["experiment"]
    experiment["name"] = f"opsd_r010_token_{selection}_delta002_dropout0_10240"
    experiment["family"] = "r010_f_token_partition_delta002_dropout0"
    experiment["comparison_id"] = "r010_f_token_partition_delta002_20260818"
    experiment["output_root"] = str(OUTPUT_ROOT)
    experiment["intervention_ablation_parent_config"] = str(source.relative_to(HERE.parents[2]))
    experiment["budget_intervention"] = "r010_to_r012"
    config["opsd"]["native_budget_weighting"]["budget_delta"] = 0.02
    destination = HERE / "configs" / f"train_token_{selection}.yaml"
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, width=1000), encoding="utf-8"
    )


def main() -> None:
    for selection in ("top20", "bottom80"):
        materialize(selection)


if __name__ == "__main__":
    main()
