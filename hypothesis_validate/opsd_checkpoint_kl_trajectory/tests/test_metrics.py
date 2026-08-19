from __future__ import annotations

import torch

from opsd.algorithm1.src.metrics import per_token_forward_kl, per_token_js


def test_identical_distributions_have_zero_divergence() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.8, -0.3]])
    assert torch.allclose(per_token_forward_kl(logits, logits), torch.zeros(2), atol=1e-7)
    assert torch.allclose(per_token_js(logits, logits), torch.zeros(2), atol=1e-7)


def test_forward_kl_direction_is_teacher_to_student() -> None:
    teacher = torch.tensor([[5.0, 0.0]])
    close_student = torch.tensor([[4.0, 0.0]])
    far_student = torch.tensor([[0.0, 5.0]])
    assert per_token_forward_kl(teacher, close_student).item() < per_token_forward_kl(teacher, far_student).item()
