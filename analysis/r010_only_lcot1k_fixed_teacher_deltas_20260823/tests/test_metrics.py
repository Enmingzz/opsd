from __future__ import annotations

import math

import torch

from opsd.analysis.r010_only_lcot1k_fixed_teacher_deltas_20260823.metrics import (
    pairwise_divergence,
)


def test_identical_distributions_are_zero() -> None:
    logits = torch.tensor([[1.0, -1.0, 0.5], [0.0, 0.0, 0.0]])
    result = pairwise_divergence(logits, logits)
    assert torch.allclose(result.jsd, torch.zeros(2), atol=1e-7)
    assert torch.allclose(result.forward_kl, torch.zeros(2), atol=1e-7)
    assert torch.allclose(result.reverse_kl, torch.zeros(2), atol=1e-7)


def test_jsd_is_symmetric_and_kl_directions_swap() -> None:
    first = torch.tensor([[3.0, 0.0, -2.0], [0.5, 1.5, -1.0]])
    second = torch.tensor([[-1.0, 2.0, 0.0], [2.0, -0.5, 0.0]])
    pq = pairwise_divergence(first, second, chunk_size=1)
    qp = pairwise_divergence(second, first, chunk_size=2)
    assert torch.allclose(pq.jsd, qp.jsd, atol=1e-7)
    assert torch.allclose(pq.forward_kl, qp.reverse_kl, atol=1e-7)
    assert torch.allclose(pq.reverse_kl, qp.forward_kl, atol=1e-7)
    assert bool((pq.jsd >= 0).all())
    assert bool((pq.jsd <= math.log(2.0) + 1e-7).all())


def test_returns_one_value_per_token_in_float32() -> None:
    first = torch.randn(7, 31, dtype=torch.bfloat16)
    second = torch.randn(7, 31, dtype=torch.bfloat16)
    result = pairwise_divergence(first, second, chunk_size=3)
    for value in (result.jsd, result.forward_kl, result.reverse_kl):
        assert value.shape == (7,)
        assert value.dtype == torch.float32
        assert bool(torch.isfinite(value).all())
