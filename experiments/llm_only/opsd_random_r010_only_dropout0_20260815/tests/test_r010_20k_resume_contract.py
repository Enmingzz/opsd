from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def resume_contract(config: dict) -> dict:
    contract = copy.deepcopy(config)
    contract["training"].pop("start_step", None)
    contract["training"].pop("adapter_path", None)
    contract["checkpointing"].pop("resume_from", None)
    contract["checkpointing"].pop("stop_at_step", None)
    return contract


def test_r010_20k_stages_have_identical_resume_contract() -> None:
    stage1 = load("train_20000_stage1.yaml")
    stage2 = load("train_20000_stage2.yaml")

    assert resume_contract(stage1) == resume_contract(stage2)
    assert stage1["checkpointing"]["stop_at_step"] == 10240
    assert stage1["checkpointing"]["resume_from"] == ""
    assert stage2["checkpointing"]["stop_at_step"] == 20000
    assert stage2["checkpointing"]["resume_from"].endswith(
        "/resume_checkpoints/step_010240"
    )


def test_every_published_r010_config_uses_only_ten_percent_retention() -> None:
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["pruning"]["method"] == "visionzip", path
        assert config["pruning"]["train_retention_ratios"] == [0.10], path
        assert config["experiment"]["parameter_scope"] == "language_decoder_only", path
        assert config["training"]["lora_dropout"] == 0.0, path


def test_20k_data_order_is_shared_across_stages() -> None:
    stage1 = load("train_20000_stage1.yaml")
    stage2 = load("train_20000_stage2.yaml")

    assert stage1["dataset"] == stage2["dataset"]
    assert stage1["dataset"]["expected_rows"] == 20000
    assert stage1["dataset"]["shuffle"] is False
    assert stage1["training"]["max_steps"] == stage2["training"]["max_steps"] == 20000
