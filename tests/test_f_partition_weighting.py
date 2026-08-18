from __future__ import annotations

import pytest
import torch

from opsd.visionzip_aokvqa.native_budget_weighting import (
    deterministic_random_token_drop_partition,
    projection_fraction_token_partition,
)
from opsd.visionzip_aokvqa.trajectory_weighting import (
    effective_batch_local_objective,
    global_f_curriculum_trajectory_weights,
    hard_trajectory_partition_weights,
)


def _token_inputs() -> tuple[torch.Tensor, ...]:
    teacher = torch.tensor([0.2, 0.4, 0.5, 0.8, 0.7, 0.3])
    budget = torch.tensor([0.1, 0.4, 0.1, 0.4, 0.4, 0.2])
    projection_fraction = torch.tensor([-0.2, 0.9, 0.1, 0.7, 0.5, 0.3])
    projection = projection_fraction * teacher
    teacher_plus = teacher + budget - 2.0 * projection
    ranking_kl = torch.tensor([1e-6, 0.5, 0.4, 0.3, 0.2, 0.1])
    valid = torch.ones(6, dtype=torch.bool)
    return teacher, budget, teacher_plus, ranking_kl, valid


def test_token_projection_top_and_bottom_are_exact_eligible_complements() -> None:
    inputs = _token_inputs()
    top = projection_fraction_token_partition(
        *inputs,
        top_fraction=0.2,
        min_kl=1e-5,
        select="top",
    )
    bottom = projection_fraction_token_partition(
        *inputs,
        top_fraction=0.2,
        min_kl=1e-5,
        select="bottom",
    )
    assert top.eligible_mask.tolist() == [False, True, True, True, True, True]
    assert top.top_mask.tolist() == [False, True, False, False, False, False]
    assert torch.equal(top.top_mask, bottom.top_mask)
    assert torch.equal(top.selected_mask | bottom.selected_mask, top.eligible_mask)
    assert not bool((top.selected_mask & bottom.selected_mask).any())
    assert int(top.selected_mask.sum()) == 1
    assert int(bottom.selected_mask.sum()) == 4
    assert not top.projection_fraction.requires_grad


def test_token_projection_selected_mean_preserves_per_token_loss_scale() -> None:
    teacher, budget, teacher_plus, ranking_kl, valid = _token_inputs()
    differentiable_kl = ranking_kl.clone().requires_grad_(True)
    partition = projection_fraction_token_partition(
        teacher,
        budget,
        teacher_plus,
        differentiable_kl,
        valid,
        top_fraction=0.4,
        min_kl=1e-5,
        select="top",
    )
    loss = differentiable_kl[partition.selected_mask].mean()
    loss.backward()
    expected = partition.selected_mask.float() / float(partition.selected_mask.sum())
    assert torch.allclose(differentiable_kl.grad, expected)


@pytest.mark.parametrize("selection", ["top", "bottom"])
def test_token_projection_allows_no_eligible_tokens(selection: str) -> None:
    teacher, budget, teacher_plus, ranking_kl, valid = _token_inputs()
    partition = projection_fraction_token_partition(
        teacher,
        budget,
        teacher_plus,
        ranking_kl,
        valid,
        min_kl=1.0,
        select=selection,
    )
    assert not bool(partition.eligible_mask.any())
    assert not bool(partition.top_mask.any())
    assert not bool(partition.selected_mask.any())


def test_token_projection_one_eligible_token_keeps_exact_complement() -> None:
    teacher, budget, teacher_plus, ranking_kl, valid = _token_inputs()
    top = projection_fraction_token_partition(
        teacher,
        budget,
        teacher_plus,
        ranking_kl,
        valid,
        min_kl=0.45,
        select="top",
    )
    bottom = projection_fraction_token_partition(
        teacher,
        budget,
        teacher_plus,
        ranking_kl,
        valid,
        min_kl=0.45,
        select="bottom",
    )
    assert int(top.eligible_mask.sum()) == 1
    assert torch.equal(top.selected_mask, top.eligible_mask)
    assert not bool(bottom.selected_mask.any())
    assert torch.equal(top.selected_mask | bottom.selected_mask, top.eligible_mask)
    assert not bool((top.selected_mask & bottom.selected_mask).any())


def test_empty_token_partition_has_graph_connected_zero_loss() -> None:
    per_token_kl = torch.tensor([0.1, 0.2], requires_grad=True)
    selected = torch.zeros(2, dtype=torch.bool)
    loss = per_token_kl[selected].mean() if selected.any() else per_token_kl.sum() * 0.0
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(per_token_kl.grad, torch.zeros_like(per_token_kl))


def test_random_drop20_is_deterministic_and_keeps_exact_eligible_complement() -> None:
    kl = torch.linspace(0.01, 0.10, 10, requires_grad=True)
    valid = torch.ones(10, dtype=torch.bool)
    first = deterministic_random_token_drop_partition(
        kl, valid, sample_key="sample-a:123", seed=42, drop_fraction=0.2
    )
    second = deterministic_random_token_drop_partition(
        kl, valid, sample_key="sample-a:123", seed=42, drop_fraction=0.2
    )
    other = deterministic_random_token_drop_partition(
        kl, valid, sample_key="sample-b:123", seed=42, drop_fraction=0.2
    )
    assert torch.equal(first.dropped_mask, second.dropped_mask)
    assert torch.equal(first.selected_mask, second.selected_mask)
    assert int(first.dropped_mask.sum()) == 2
    assert int(first.selected_mask.sum()) == 8
    assert torch.equal(first.dropped_mask | first.selected_mask, first.eligible_mask)
    assert not bool((first.dropped_mask & first.selected_mask).any())
    assert not first.selected_mask.requires_grad
    assert not torch.equal(first.dropped_mask, other.dropped_mask)


def test_random_drop20_applies_kl_floor_before_random_partition() -> None:
    kl = torch.tensor([0.0, 1e-6, 1e-5, 0.1, 0.2])
    valid = torch.ones(5, dtype=torch.bool)
    partition = deterministic_random_token_drop_partition(
        kl,
        valid,
        sample_key="sample-floor",
        seed=7,
        drop_fraction=0.2,
        min_kl=1e-5,
    )
    assert partition.eligible_mask.tolist() == [False, False, True, True, True]
    assert int(partition.dropped_mask.sum()) == 1
    assert int(partition.selected_mask.sum()) == 2


def test_trajectory_partition_is_exactly_twenty_percent_over_five_batches() -> None:
    signal = torch.arange(32, dtype=torch.float32)
    top_counts = []
    for batch_ordinal in range(5):
        top = hard_trajectory_partition_weights(
            signal,
            top_fraction=0.2,
            batch_ordinal=batch_ordinal,
            select="top",
        )
        bottom = hard_trajectory_partition_weights(
            signal,
            top_fraction=0.2,
            batch_ordinal=batch_ordinal,
            select="bottom",
        )
        top_counts.append(top.top_count)
        assert torch.equal(top.top_mask, bottom.top_mask)
        assert not bool((top.selected_mask & bottom.selected_mask).any())
        assert bool((top.selected_mask | bottom.selected_mask).all())
        assert float(top.probability_weight.sum()) == pytest.approx(1.0)
        assert float(bottom.probability_weight.sum()) == pytest.approx(1.0)
    assert top_counts == [6, 6, 7, 6, 7]
    assert sum(top_counts) == 32


def test_trajectory_partition_executor_is_selected_group_mean() -> None:
    losses = torch.arange(1, 33, dtype=torch.float32, requires_grad=True)
    result = hard_trajectory_partition_weights(
        torch.arange(32, dtype=torch.float32),
        top_fraction=0.2,
        batch_ordinal=2,
        select="top",
    )
    local = [
        effective_batch_local_objective(
            loss,
            probability,
            effective_batch_size=32,
        )
        for loss, probability in zip(losses, result.probability_weight)
    ]
    accumulated = torch.stack(local).sum() / 32.0
    assert accumulated.item() == pytest.approx(
        losses.detach()[result.selected_mask].mean().item()
    )


def test_global_f_curriculum_is_detached_and_not_batch_renormalized() -> None:
    signal = torch.tensor([-0.2, 0.0, 0.25, 0.5, 0.75, 1.0, 1.2], requires_grad=True)
    result = global_f_curriculum_trajectory_weights(signal, gamma=4.0)
    expected = torch.tensor([0.0, 0.0, 0.75, 1.0, 0.75, 0.0, 0.0])
    assert torch.allclose(result.objective_weight, expected)
    assert torch.allclose(result.probability_weight, expected / 7.0)
    assert not result.objective_weight.requires_grad
    assert float(result.probability_weight.sum()) == pytest.approx(2.5 / 7.0)
