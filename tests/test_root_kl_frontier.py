from __future__ import annotations

import math

import pytest
import torch

from visionzip_aokvqa.root_kl_frontier import (
    WinsorizedRootKLFrontierState,
    root_kl_sensitivity,
)
from visionzip_aokvqa.trajectory_weighting import sensitivity_frontier_weights


def calibrated_state(**overrides) -> WinsorizedRootKLFrontierState:
    state = WinsorizedRootKLFrontierState(
        calibration_target_per_ratio=2,
        **overrides,
    )
    state.update([0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0])
    state.update([0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0])
    assert state.ready
    return state


def test_root_kl_sensitivity_is_symmetric_and_nonnegative() -> None:
    expected = abs(math.sqrt(0.04) - math.sqrt(0.01))
    assert root_kl_sensitivity(0.04, 0.01) == pytest.approx(expected)
    assert root_kl_sensitivity(0.01, 0.04) == pytest.approx(expected)
    with pytest.raises(ValueError):
        root_kl_sensitivity(-0.1, 0.1)


def test_calibration_is_ratio_specific_and_winsorized() -> None:
    state = calibrated_state(winsor_cap=4.0)
    assert state.calibration_median == pytest.approx(
        {"0.10": 1.0, "0.20": 2.0, "0.30": 3.0, "0.40": 4.0}
    )
    assert state.robust_signal(0.1, 100.0) == 4.0
    assert state.local_robustness(0.1, 100.0) == pytest.approx(0.2, abs=1e-8)


def test_uniform_contraction_advances_frontier() -> None:
    state = calibrated_state(
        winsor_cap=4.0,
        ema_half_life_trajectories=1.0,
        progress_drop_scale=0.35,
    )
    before = state.frontier_index
    update = None
    for _ in range(8):
        update = state.update([0.1, 0.2, 0.3, 0.4], [0.5, 1.0, 1.5, 2.0])
    assert update is not None
    assert update["progress_before"] <= update["progress_after"]
    assert update["frontier_after"] == pytest.approx(state.frontier_index)
    assert update["calibration_counts"] == state.calibration_counts
    assert update["initial_sensitivity"] == state.initial_sensitivity
    assert update["ema_sensitivity"] == state.ema_sensitivity
    assert state.unresolved_sensitivity < 1.0
    assert state.frontier_index > before


def test_weighting_preserves_unweighted_kl_mass() -> None:
    state = calibrated_state()
    losses = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float32)
    signals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    ratios = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    result = sensitivity_frontier_weights(
        losses,
        signals,
        ratios,
        state,
        weight_floor=0.1,
    )
    assert not result.weight.requires_grad
    assert result.weighted_loss_mass == pytest.approx(result.unweighted_loss_mass)


def test_state_round_trip_is_exact() -> None:
    original = calibrated_state()
    original.update([0.1, 0.2, 0.3, 0.4], [0.8, 1.5, 2.0, 2.5])
    restored = WinsorizedRootKLFrontierState(calibration_target_per_ratio=2)
    restored.load_state_dict(original.state_dict())
    assert restored.state_dict() == original.state_dict()
