from __future__ import annotations

import math

import pytest
import torch

from opsd.visionzip_aokvqa.native_budget_weighting import (
    projection_mass_grouped_weights,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    # The raw projection-mass and projection-fraction rankings intentionally differ:
    # token 0 has the largest P/A, while token 1 has the largest positive P.
    teacher = torch.tensor([0.01, 1.00, 0.50, 0.40], dtype=torch.float32)
    budget = torch.tensor([0.02, 0.40, 0.25, 0.10], dtype=torch.float32)
    desired_projection = torch.tensor([0.009, 0.20, 0.10, -0.10])
    teacher_plus = teacher + budget - 2.0 * desired_projection
    loss = torch.tensor([0.2, 1.4, 0.8, 0.1], dtype=torch.float32)
    valid = torch.ones(4, dtype=torch.bool)
    return teacher, budget, teacher_plus, loss, valid


def test_ranks_raw_positive_projection_mass_not_fraction() -> None:
    teacher, budget, teacher_plus, loss, valid = _inputs()
    result = projection_mass_grouped_weights(
        teacher,
        budget,
        teacher_plus,
        loss,
        valid,
        top_fraction=0.25,
        high_group_lambda=0.30,
        preserve_loss_mass=False,
    )
    assert torch.equal(result.high_mask, torch.tensor([False, True, False, False]))
    assert torch.allclose(
        result.projection_mass,
        torch.tensor([0.009, 0.20, 0.10, -0.10]),
        atol=1e-7,
    )


def test_grouped_weights_reconstruct_lambda_objective() -> None:
    teacher = torch.full((7,), 0.5)
    budget = torch.linspace(0.1, 0.7, 7)
    projection = torch.linspace(-0.1, 0.5, 7)
    teacher_plus = teacher + budget - 2.0 * projection
    loss = torch.arange(1, 8, dtype=torch.float32)
    valid = torch.ones(7, dtype=torch.bool)
    result = projection_mass_grouped_weights(
        teacher,
        budget,
        teacher_plus,
        loss,
        valid,
        top_fraction=0.10,
        high_group_lambda=0.30,
        preserve_loss_mass=False,
    )
    assert int(result.high_mask.sum()) == math.ceil(0.10 * 7)
    assert int(result.low_mask.sum()) == 6
    assert result.raw_weight[valid].mean().item() == pytest.approx(1.0)
    weighted = (result.raw_weight[valid] * loss[valid]).mean()
    expected = 0.30 * loss[result.high_mask].mean() + 0.70 * loss[result.low_mask].mean()
    assert weighted.item() == pytest.approx(expected.item(), rel=1e-6)
    ratio = result.raw_weight[result.high_mask][0] / result.raw_weight[result.low_mask][0]
    actual_high_fraction = result.high_mask.sum().item() / valid.sum().item()
    expected_ratio = (0.30 / actual_high_fraction) / (
        0.70 / (1.0 - actual_high_fraction)
    )
    assert ratio.item() == pytest.approx(expected_ratio, rel=1e-6)


def test_default_is_direct_grouped_objective_with_live_kl_gradient() -> None:
    teacher, budget, teacher_plus, base_loss, valid = _inputs()
    loss = base_loss.clone().requires_grad_(True)
    result = projection_mass_grouped_weights(
        teacher,
        budget,
        teacher_plus,
        loss,
        valid,
        top_fraction=0.25,
        high_group_lambda=0.30,
    )
    objective = (result.weight[valid] * loss[valid]).mean()
    expected = 0.30 * loss[result.high_mask].mean() + 0.70 * loss[result.low_mask].mean()
    assert objective.item() == pytest.approx(expected.item(), rel=1e-6)
    assert result.loss_mass_scale.item() == pytest.approx(1.0)
    objective.backward()
    assert torch.allclose(loss.grad, result.raw_weight / valid.sum(), atol=1e-7)


def test_explicit_detached_mass_preservation_ablation_changes_gradient() -> None:
    teacher, budget, teacher_plus, base_loss, valid = _inputs()
    loss = base_loss.clone().requires_grad_(True)
    result = projection_mass_grouped_weights(
        teacher,
        budget,
        teacher_plus,
        loss,
        valid,
        top_fraction=0.25,
        high_group_lambda=0.30,
        preserve_loss_mass=True,
    )
    assert not result.weight.requires_grad
    objective = (result.weight[valid] * loss[valid]).mean()
    assert objective.item() == pytest.approx(loss[valid].mean().item(), rel=2e-6)
    objective.backward()
    assert torch.allclose(loss.grad, result.weight / valid.sum(), atol=1e-7)
    assert not torch.allclose(loss.grad[valid], torch.full((4,), 0.25))


@pytest.mark.parametrize("valid_count", [1, 4])
def test_degenerate_projection_falls_back_to_vanilla(valid_count: int) -> None:
    teacher = torch.ones(valid_count)
    budget = torch.zeros(valid_count)
    teacher_plus = torch.ones(valid_count) + 0.2
    loss = torch.linspace(0.1, 0.4, valid_count)
    valid = torch.ones(valid_count, dtype=torch.bool)
    result = projection_mass_grouped_weights(
        teacher,
        budget,
        teacher_plus,
        loss,
        valid,
        top_fraction=0.10,
        high_group_lambda=0.30,
        preserve_loss_mass=True,
    )
    assert result.degenerate
    assert torch.equal(result.weight[valid], torch.ones(valid_count))


def test_invalid_hyperparameters_are_rejected() -> None:
    teacher, budget, teacher_plus, loss, valid = _inputs()
    with pytest.raises(ValueError, match="top_fraction"):
        projection_mass_grouped_weights(
            teacher, budget, teacher_plus, loss, valid, top_fraction=0.0
        )
    with pytest.raises(ValueError, match="high_group_lambda"):
        projection_mass_grouped_weights(
            teacher, budget, teacher_plus, loss, valid, high_group_lambda=1.0
        )
