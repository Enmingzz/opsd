from __future__ import annotations

import pytest
import torch

from opsd.visionzip_aokvqa.losses import keep_mask_above_kl_floor


def test_floor_removes_strictly_lower_valid_values() -> None:
    ranking = torch.tensor([0.0, 0.5e-5, 1.0e-5, 2.0e-5, 9.0])
    valid = torch.tensor([True, True, True, True, False])
    keep, removed = keep_mask_above_kl_floor(ranking, valid, min_kl=1e-5)

    assert keep.tolist() == [False, False, True, True, False]
    assert removed.tolist() == [0, 1]
    assert not keep.requires_grad
    assert not removed.requires_grad


def test_filtered_objective_is_mean_over_kept_tokens() -> None:
    ranking = torch.tensor([0.0, 0.5e-5, 1.0e-5, 2.0e-5])
    valid = torch.ones(4, dtype=torch.bool)
    keep, _ = keep_mask_above_kl_floor(ranking, valid, min_kl=1e-5)
    losses = torch.tensor([0.0, 0.5e-5, 1.0e-5, 2.0e-5], requires_grad=True)

    loss = losses[keep].mean()
    loss.backward()

    assert loss.item() == pytest.approx(1.5e-5)
    assert torch.equal(losses.grad[:2], torch.zeros(2))
    assert torch.allclose(losses.grad[2:], torch.full((2,), 0.5))


def test_zero_floor_is_identity_for_nonnegative_kl() -> None:
    ranking = torch.tensor([0.0, 1.0e-6, 1.0])
    valid = torch.tensor([True, False, True])
    keep, removed = keep_mask_above_kl_floor(ranking, valid, min_kl=0.0)

    assert torch.equal(keep, valid)
    assert removed.numel() == 0


def test_empty_selection_is_allowed_for_graph_connected_zero() -> None:
    ranking = torch.tensor([0.0, 0.5e-5])
    valid = torch.ones(2, dtype=torch.bool)
    keep, removed = keep_mask_above_kl_floor(ranking, valid, min_kl=1e-5)
    losses = ranking.clone().requires_grad_(True)

    loss = losses[keep].mean() if keep.any() else losses.sum() * 0.0
    loss.backward()

    assert not keep.any()
    assert removed.tolist() == [0, 1]
    assert loss.item() == 0.0
    assert torch.equal(losses.grad, torch.zeros_like(losses))


def test_floor_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        keep_mask_above_kl_floor(torch.tensor([1.0]), torch.tensor([False]), 1e-5)
    with pytest.raises(ValueError):
        keep_mask_above_kl_floor(torch.tensor([1.0]), torch.tensor([True, False]), 1e-5)
    with pytest.raises(ValueError):
        keep_mask_above_kl_floor(torch.tensor([float("nan")]), torch.tensor([True]), 1e-5)
    with pytest.raises(ValueError):
        keep_mask_above_kl_floor(torch.tensor([1.0]), torch.tensor([True]), -1e-5)
