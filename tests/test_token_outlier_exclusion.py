from __future__ import annotations

import pytest
import torch

from opsd.visionzip_aokvqa.losses import (
    compute_forward_kl,
    compute_per_token_kl,
    keep_mask_after_topk_exclusion,
    resolve_token_outlier_top_k,
)


def test_topk_exclusion_removes_largest_valid_values() -> None:
    ranking = torch.tensor([0.2, 9.0, 3.0, 7.0, 1.0])
    valid = torch.tensor([True, True, False, True, True])
    keep, removed = keep_mask_after_topk_exclusion(ranking, valid, top_k=2)
    assert removed.tolist() == [1, 3]
    assert keep.tolist() == [True, False, False, False, True]
    assert not keep.requires_grad


def test_top5_is_nested_inside_top8() -> None:
    ranking = torch.linspace(0.0, 1.0, 20)
    valid = torch.ones(20, dtype=torch.bool)
    _, removed5 = keep_mask_after_topk_exclusion(ranking, valid, top_k=5)
    _, removed8 = keep_mask_after_topk_exclusion(ranking, valid, top_k=8)
    assert set(removed5.tolist()) < set(removed8.tolist())


def test_ties_are_resolved_by_original_position() -> None:
    ranking = torch.tensor([4.0, 4.0, 4.0, 1.0])
    valid = torch.ones(4, dtype=torch.bool)
    _, removed = keep_mask_after_topk_exclusion(ranking, valid, top_k=2)
    assert removed.tolist() == [0, 1]


def test_exclusion_always_retains_one_token() -> None:
    ranking = torch.tensor([3.0, 2.0, 1.0])
    valid = torch.ones(3, dtype=torch.bool)
    keep, removed = keep_mask_after_topk_exclusion(ranking, valid, top_k=8)
    assert removed.numel() == 2
    assert keep.sum().item() == 1


def test_zero_exclusion_is_identity() -> None:
    ranking = torch.tensor([3.0, 2.0, 1.0])
    valid = torch.tensor([True, False, True])
    keep, removed = keep_mask_after_topk_exclusion(ranking, valid, top_k=0)
    assert torch.equal(keep, valid)
    assert removed.numel() == 0


def test_ratio_specific_top_k_resolution() -> None:
    mapping = {"0.10": 5, "0.20": 0, "0.30": 5, "0.40": 5}
    assert resolve_token_outlier_top_k(99, mapping, 0.10) == 5
    assert resolve_token_outlier_top_k(99, mapping, 0.20) == 0
    assert resolve_token_outlier_top_k(99, mapping, 0.30) == 5
    assert resolve_token_outlier_top_k(99, mapping, 0.40) == 5


def test_legacy_scalar_top_k_resolution_is_unchanged() -> None:
    assert resolve_token_outlier_top_k(5, None, 0.20) == 5


def test_ratio_specific_top_k_requires_complete_unambiguous_mapping() -> None:
    with pytest.raises(ValueError):
        resolve_token_outlier_top_k(0, {"0.10": 5}, 0.20)
    with pytest.raises(ValueError):
        resolve_token_outlier_top_k(0, {"0.10": -1}, 0.10)
    with pytest.raises(ValueError):
        resolve_token_outlier_top_k(0, {"0.10": 1.5}, 0.10)


def test_remaining_mean_is_renormalized_and_removed_tokens_have_zero_gradient() -> None:
    ranking = torch.tensor([9.0, 8.0, 1.0, 0.5])
    valid = torch.ones(4, dtype=torch.bool)
    keep, _ = keep_mask_after_topk_exclusion(ranking, valid, top_k=2)
    forward_losses = torch.tensor([100.0, 50.0, 4.0, 2.0], requires_grad=True)
    loss = forward_losses[keep].mean()
    loss.backward()
    assert loss.item() == pytest.approx(3.0)
    assert torch.equal(forward_losses.grad, torch.tensor([0.0, 0.0, 0.5, 0.5]))


def test_per_token_forward_kl_mean_matches_original_scalar() -> None:
    generator = torch.Generator().manual_seed(42)
    teacher = torch.randn(7, 31, generator=generator)
    student = torch.randn(7, 31, generator=generator, requires_grad=True)
    scalar = compute_forward_kl(teacher, student, temperature=1.0, chunk_size=3)
    token_mean = compute_per_token_kl(teacher, student, temperature=1.0, chunk_size=3).mean()
    assert torch.allclose(token_mean, scalar, rtol=1e-6, atol=1e-7)


def test_same_forward_kl_ranks_tokens_and_supplies_training_gradients() -> None:
    generator = torch.Generator().manual_seed(7)
    teacher = torch.randn(9, 23, generator=generator)
    student = torch.randn(9, 23, generator=generator, requires_grad=True)
    per_token = compute_per_token_kl(teacher, student, temperature=1.0, chunk_size=4)
    valid = torch.ones(9, dtype=torch.bool)
    keep, removed = keep_mask_after_topk_exclusion(per_token.detach(), valid, top_k=3)
    expected = torch.argsort(per_token.detach(), descending=True, stable=True)[:3]
    assert torch.equal(removed, expected)

    per_token[keep].mean().backward()
    assert torch.count_nonzero(student.grad[removed]).item() == 0
    assert torch.count_nonzero(student.grad[keep]).item() > 0


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        keep_mask_after_topk_exclusion(
            torch.tensor([1.0, 2.0]),
            torch.tensor([True]),
            top_k=1,
        )
    with pytest.raises(ValueError):
        keep_mask_after_topk_exclusion(
            torch.tensor([1.0, float("nan")]),
            torch.tensor([True, True]),
            top_k=1,
        )
    with pytest.raises(ValueError):
        keep_mask_after_topk_exclusion(
            torch.tensor([1.0]),
            torch.tensor([True]),
            top_k=-1,
        )
