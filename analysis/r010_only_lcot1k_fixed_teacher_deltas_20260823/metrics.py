"""Exact detached token-level divergences for the fixed-teacher sweep."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairwiseDivergence:
    """Divergences for an ordered pair (first, second).

    ``forward_kl`` is KL(first || second), while ``reverse_kl`` is
    KL(second || first). JSD is symmetric.
    """

    jsd: torch.Tensor
    forward_kl: torch.Tensor
    reverse_kl: torch.Tensor


def _flatten(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits.reshape(-1, logits.shape[-1])
    raise ValueError(f"Expected [T,V] or [B,T,V] logits, got {tuple(logits.shape)}")


def pairwise_divergence(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 16,
) -> PairwiseDivergence:
    """Compute exact full-vocabulary FP32 JSD and directional KL per token."""

    first = _flatten(first_logits)
    second = _flatten(second_logits)
    if first.shape != second.shape:
        raise ValueError(f"Logit shapes differ: {tuple(first.shape)} vs {tuple(second.shape)}")
    if first.shape[0] == 0:
        raise ValueError("At least one token position is required")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    js_values: list[torch.Tensor] = []
    forward_values: list[torch.Tensor] = []
    reverse_values: list[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    with torch.inference_mode():
        for start in range(0, int(first.shape[0]), chunk_size):
            end = min(start + chunk_size, int(first.shape[0]))
            log_p = F.log_softmax(first[start:end].float() / float(temperature), dim=-1)
            log_q = F.log_softmax(second[start:end].float() / float(temperature), dim=-1)
            if not bool(torch.isfinite(log_p).all()) or not bool(torch.isfinite(log_q).all()):
                raise FloatingPointError("Non-finite log probabilities")
            p = log_p.exp()
            q = log_q.exp()
            log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
            forward = (p * (log_p - log_q)).sum(dim=-1).clamp_min(0.0)
            reverse = (q * (log_q - log_p)).sum(dim=-1).clamp_min(0.0)
            jsd = (
                0.5 * (p * (log_p - log_m)).sum(dim=-1)
                + 0.5 * (q * (log_q - log_m)).sum(dim=-1)
            ).clamp(0.0, math.log(2.0))
            js_values.append(jsd)
            forward_values.append(forward)
            reverse_values.append(reverse)

    return PairwiseDivergence(
        jsd=torch.cat(js_values).detach().float(),
        forward_kl=torch.cat(forward_values).detach().float(),
        reverse_kl=torch.cat(reverse_values).detach().float(),
    )
