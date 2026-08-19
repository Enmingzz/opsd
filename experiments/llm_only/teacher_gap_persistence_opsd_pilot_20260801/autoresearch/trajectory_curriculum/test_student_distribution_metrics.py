from __future__ import annotations

import math

import torch

from student_distribution_metrics import (
    compute_action_sensitivity_metrics,
    compute_student_distribution_metrics,
)


def test_identical_distributions_are_zero() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 0.1, -0.3]])
    result = compute_student_distribution_metrics(logits, logits, logits, teacher_top_k=2)
    for name in result.__dataclass_fields__:
        value = getattr(result, name)
        if name == "mixture_entropy":
            assert (value > 0.0).all()
            continue
        assert torch.allclose(value, torch.zeros_like(value), atol=1e-7), (name, value)


def test_symmetric_metrics_are_invariant_to_student_order() -> None:
    generator = torch.Generator().manual_seed(9)
    a = torch.randn(4, 17, generator=generator)
    b = torch.randn(4, 17, generator=generator)
    teacher = torch.randn(4, 17, generator=generator)
    ab = compute_student_distribution_metrics(a, b, teacher, teacher_top_k=8)
    ba = compute_student_distribution_metrics(b, a, teacher, teacher_top_k=8)
    for name in (
        "jeffreys",
        "js",
        "root_js",
        "hellinger_sq",
        "hellinger",
        "total_variation",
        "mixture_entropy",
        "normalized_js_mixture_entropy",
        "teacher_logratio_rms",
        "teacher_logratio_std",
        "teacher_logratio_l1_centered",
        "teacher_support_js",
        "teacher_normalized_root_js",
        "teacher_normalized_hellinger",
        "teacher_normalized_l2",
    ):
        assert torch.allclose(getattr(ab, name), getattr(ba, name), atol=2e-6), name


def test_metrics_are_finite_bounded_and_detached() -> None:
    a = torch.tensor([[10.0, -4.0, -8.0]], requires_grad=True)
    b = torch.tensor([[-3.0, 9.0, -7.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    result = compute_student_distribution_metrics(a, b, teacher, teacher_top_k=2)
    for name in result.__dataclass_fields__:
        value = getattr(result, name)
        assert torch.isfinite(value).all(), name
        assert not value.requires_grad, name
        assert (value >= 0).all(), name
    assert float(result.js.max()) <= math.log(2.0) + 1e-6
    assert float(result.hellinger_sq.max()) <= 1.0 + 1e-6
    assert float(result.total_variation.max()) <= 1.0 + 1e-6
    assert float(result.normalized_js_mixture_entropy.max()) <= 1.0 + 1e-6
    assert float(result.teacher_normalized_root_js.max()) <= 1.0 + 1e-6
    assert float(result.teacher_normalized_hellinger.max()) <= 1.0 + 1e-6
    assert float(result.teacher_normalized_l2.max()) <= 1.0 + 1e-6


def test_teacher_support_coarsening_suppresses_irrelevant_tail_swap() -> None:
    # Both students agree on the teacher-supported token and rearrange only
    # low-teacher-mass tail probability.
    teacher = torch.tensor([[14.0, -10.0, -10.0, -10.0]])
    a = torch.tensor([[8.0, 2.0, -8.0, -8.0]])
    b = torch.tensor([[8.0, -8.0, 2.0, -8.0]])
    result = compute_student_distribution_metrics(a, b, teacher, teacher_top_k=1)
    assert float(result.teacher_support_js) < float(result.js)
    assert float(result.teacher_logratio_std) <= float(result.teacher_logratio_rms) + 1e-7


def test_teacher_normalized_distances_remove_common_teacher_scale() -> None:
    teacher = torch.tensor([[2.0, 0.0, -1.0, -2.0]])
    a = torch.tensor([[1.8, 0.1, -0.9, -2.0]])
    b = torch.tensor([[1.6, 0.2, -0.8, -2.0]])
    result = compute_student_distribution_metrics(a, b, teacher, teacher_top_k=2)
    for name in (
        "teacher_normalized_root_js",
        "teacher_normalized_hellinger",
        "teacher_normalized_l2",
    ):
        value = getattr(result, name)
        assert torch.isfinite(value).all(), name
        assert (value >= 0.0).all() and (value <= 1.0).all(), (name, value)


def test_action_sensitivity_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]])
    tokens = torch.tensor([0, 2])
    result = compute_action_sensitivity_metrics(logits, logits, tokens)
    for name in result.__dataclass_fields__:
        value = getattr(result, name)
        assert torch.allclose(value, torch.zeros_like(value), atol=1e-7), (name, value)


def test_action_sensitivity_tracks_generated_token_probability() -> None:
    a = torch.tensor([[5.0, 0.0, -1.0]], requires_grad=True)
    b = torch.tensor([[0.0, 5.0, -1.0]], requires_grad=True)
    result = compute_action_sensitivity_metrics(a, b, torch.tensor([0]))
    assert float(result.action_logprob_delta) < 0.0
    assert float(result.budget_advantage) == 0.0
    assert float(result.action_abs_logprob_gap) > 1.0
    assert 0.0 < float(result.action_bernoulli_js) <= math.log(2.0) + 1e-6
    for name in result.__dataclass_fields__:
        assert not getattr(result, name).requires_grad, name

    reverse = compute_action_sensitivity_metrics(b, a, torch.tensor([0]))
    assert float(reverse.action_logprob_delta) > 0.0
    assert torch.allclose(reverse.budget_advantage, reverse.action_logprob_delta)
