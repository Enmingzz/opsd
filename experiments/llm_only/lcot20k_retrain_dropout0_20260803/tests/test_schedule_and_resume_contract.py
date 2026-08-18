import copy
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from opsd.visionzip_aokvqa.train import checkpoint_contract_sha256, sample_retention_ratio


def _progressive_config():
    return {
        "training": {"seed": 42, "max_steps": 20_000},
        "pruning": {
            "retention_ratio_schedule": "progressive",
            "train_retention_ratios": [0.4, 0.3, 0.2, 0.1],
            "progressive_phase_end_steps": [4_992, 9_984, 14_976, 20_000],
        },
        "checkpointing": {"stop_at_step": 9_984, "resume_from": ""},
    }


def test_optimizer_aligned_progressive_counts():
    config = _progressive_config()
    counts = {ratio: 0 for ratio in (0.4, 0.3, 0.2, 0.1)}
    rng = random.Random(42)
    for global_index in range(20_000):
        ratio = sample_retention_ratio(
            config,
            rng,
            progress_step=global_index,
            total_steps=20_000,
            sample_id=f"sample-{global_index}",
        )
        counts[ratio] += 1
    assert counts == {0.4: 4_992, 0.3: 4_992, 0.2: 4_992, 0.1: 5_024}


def test_progressive_phase_boundaries():
    config = _progressive_config()
    rng = random.Random(42)
    expected = {
        0: 0.4,
        4_991: 0.4,
        4_992: 0.3,
        9_983: 0.3,
        9_984: 0.2,
        14_975: 0.2,
        14_976: 0.1,
        19_999: 0.1,
    }
    for global_index, ratio in expected.items():
        assert sample_retention_ratio(
            config,
            rng,
            progress_step=global_index,
            total_steps=20_000,
            sample_id=f"sample-{global_index}",
        ) == ratio


def test_segment_controls_do_not_change_checkpoint_contract():
    stage1 = _progressive_config()
    stage2 = copy.deepcopy(stage1)
    stage2["checkpointing"]["stop_at_step"] = 20_000
    stage2["checkpointing"]["resume_from"] = "/tmp/step_009984"
    assert checkpoint_contract_sha256(stage1) == checkpoint_contract_sha256(stage2)


def test_legacy_progressive_schedule_is_unchanged():
    config = {
        "pruning": {
            "retention_ratio_schedule": "progressive",
            "train_retention_ratios": [0.4, 0.3, 0.2, 0.1],
        }
    }
    rng = random.Random(42)
    assert sample_retention_ratio(config, rng, 0, 9_984) == 0.4
    assert sample_retention_ratio(config, rng, 2_496, 9_984) == 0.3
    assert sample_retention_ratio(config, rng, 4_992, 9_984) == 0.2
    assert sample_retention_ratio(config, rng, 7_488, 9_984) == 0.1
