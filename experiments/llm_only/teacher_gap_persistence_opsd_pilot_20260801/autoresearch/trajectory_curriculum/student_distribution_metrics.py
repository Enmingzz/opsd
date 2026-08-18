"""Detached token-level distances between two student budget distributions.

These diagnostics are deliberately separate from the OPSD training loss.  The
full-teacher distribution is used only to identify vocabulary support for the
teacher-anchored diagnostics; both compared distributions are student
distributions evaluated on the same text prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StudentDistributionMetrics:
    kl_b_to_plus: torch.Tensor
    kl_plus_to_b: torch.Tensor
    jeffreys: torch.Tensor
    js: torch.Tensor
    root_js: torch.Tensor
    hellinger_sq: torch.Tensor
    hellinger: torch.Tensor
    total_variation: torch.Tensor
    mixture_entropy: torch.Tensor
    normalized_js_mixture_entropy: torch.Tensor
    teacher_logratio_rms: torch.Tensor
    teacher_logratio_std: torch.Tensor
    teacher_logratio_l1_centered: torch.Tensor
    teacher_support_js: torch.Tensor
    teacher_js_b: torch.Tensor
    teacher_js_b_plus: torch.Tensor
    teacher_normalized_root_js: torch.Tensor
    teacher_normalized_hellinger: torch.Tensor
    teacher_normalized_l2: torch.Tensor


@dataclass(frozen=True)
class ActionSensitivityMetrics:
    action_logprob_delta: torch.Tensor
    budget_advantage: torch.Tensor
    action_abs_logprob_gap: torch.Tensor
    action_bernoulli_js: torch.Tensor
    action_root_bernoulli_js: torch.Tensor


@dataclass(frozen=True)
class BudgetVectorGeometryMetrics:
    probability_dot_product: torch.Tensor
    probability_teacher_norm_sq: torch.Tensor
    probability_budget_norm_sq: torch.Tensor
    probability_residual_norm_sq: torch.Tensor
    probability_cosine: torch.Tensor
    probability_projection_raw: torch.Tensor
    probability_closure_fraction: torch.Tensor
    probability_signed_progress: torch.Tensor
    probability_residual_ratio: torch.Tensor
    probability_step_ratio: torch.Tensor
    probability_orthogonal_ratio: torch.Tensor
    hellinger_dot_product: torch.Tensor
    hellinger_teacher_norm_sq: torch.Tensor
    hellinger_budget_norm_sq: torch.Tensor
    hellinger_residual_norm_sq: torch.Tensor
    hellinger_cosine: torch.Tensor
    hellinger_projection_raw: torch.Tensor
    hellinger_closure_fraction: torch.Tensor
    hellinger_signed_progress: torch.Tensor
    hellinger_residual_ratio: torch.Tensor
    hellinger_step_ratio: torch.Tensor
    hellinger_orthogonal_ratio: torch.Tensor
    support_teacher_mass: torch.Tensor
    support_student_mass: torch.Tensor
    support_student_plus_mass: torch.Tensor


def _flatten(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits.reshape(-1, logits.shape[-1])
    raise ValueError(f"Expected [T,V] or [B,T,V] logits, got {tuple(logits.shape)}.")


def _direction_geometry(
    teacher_vector: torch.Tensor,
    student_vector: torch.Tensor,
    student_plus_vector: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, ...]:
    teacher_direction = teacher_vector - student_vector
    budget_step = student_plus_vector - student_vector
    residual = teacher_vector - student_plus_vector
    teacher_norm_sq = teacher_direction.square().sum(dim=-1)
    budget_norm_sq = budget_step.square().sum(dim=-1)
    residual_norm_sq = residual.square().sum(dim=-1)
    dot = (teacher_direction * budget_step).sum(dim=-1)
    cosine_denom = (teacher_norm_sq * budget_norm_sq).sqrt()
    cosine = torch.where(
        cosine_denom > eps,
        dot / cosine_denom.clamp_min(eps),
        torch.zeros_like(dot),
    ).clamp(-1.0, 1.0)
    projection_raw = torch.where(
        teacher_norm_sq > eps,
        dot / teacher_norm_sq.clamp_min(eps),
        torch.zeros_like(dot),
    )
    closure = projection_raw.clamp(0.0, 1.0)
    signed_progress = torch.where(
        teacher_norm_sq > eps,
        1.0 - residual_norm_sq / teacher_norm_sq.clamp_min(eps),
        torch.zeros_like(dot),
    )
    residual_ratio = torch.where(
        teacher_norm_sq > eps,
        torch.sqrt(residual_norm_sq / teacher_norm_sq.clamp_min(eps)),
        torch.ones_like(dot),
    )
    step_ratio = torch.where(
        teacher_norm_sq > eps,
        torch.sqrt(budget_norm_sq / teacher_norm_sq.clamp_min(eps)),
        torch.zeros_like(dot),
    )
    parallel_norm_sq = torch.where(
        teacher_norm_sq > eps,
        dot.square() / teacher_norm_sq.clamp_min(eps),
        torch.zeros_like(dot),
    )
    orthogonal_norm_sq = (budget_norm_sq - parallel_norm_sq).clamp_min(0.0)
    orthogonal_ratio = torch.where(
        teacher_norm_sq > eps,
        torch.sqrt(orthogonal_norm_sq / teacher_norm_sq.clamp_min(eps)),
        torch.zeros_like(dot),
    )
    return (
        dot,
        teacher_norm_sq,
        budget_norm_sq,
        residual_norm_sq,
        cosine,
        projection_raw,
        closure,
        signed_progress,
        residual_ratio,
        step_ratio,
        orthogonal_ratio,
    )


def compute_budget_vector_geometry(
    logits_b: torch.Tensor,
    logits_b_plus: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int = 100,
    temperature: float = 1.0,
    chunk_size: int = 16,
    eps: float = 1e-12,
) -> BudgetVectorGeometryMetrics:
    """Measure whether a native budget increase moves toward the teacher.

    At every generated-token position, the support is the top-k vocabulary
    items under ``max(q, p_b, p_b_plus)`` plus one residual-mass bucket. This
    keeps important wrong student alternatives as well as teacher-preferred
    alternatives while producing a fixed, interpretable vector. Geometry is
    computed both in probability space and in the square-root probability
    (Hellinger) embedding.
    """

    logits_b = _flatten(logits_b)
    logits_b_plus = _flatten(logits_b_plus)
    teacher_logits = _flatten(teacher_logits)
    if logits_b.shape != logits_b_plus.shape or logits_b.shape != teacher_logits.shape:
        raise ValueError(
            "Student/student-plus/teacher logits must align: "
            f"{tuple(logits_b.shape)}, {tuple(logits_b_plus.shape)}, "
            f"{tuple(teacher_logits.shape)}."
        )
    if logits_b.shape[0] == 0:
        raise ValueError("At least one token position is required.")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    vocab_size = int(logits_b.shape[-1])
    support_size = min(max(1, int(top_k)), vocab_size)
    chunk_size = max(1, int(chunk_size))
    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in BudgetVectorGeometryMetrics.__dataclass_fields__
    }

    with torch.no_grad():
        for start in range(0, int(logits_b.shape[0]), chunk_size):
            end = min(start + chunk_size, int(logits_b.shape[0]))
            p = F.softmax(logits_b[start:end].float() / float(temperature), dim=-1)
            p_plus = F.softmax(
                logits_b_plus[start:end].float() / float(temperature), dim=-1
            )
            q = F.softmax(teacher_logits[start:end].float() / float(temperature), dim=-1)
            support_score = torch.maximum(torch.maximum(p, p_plus), q)
            indices = torch.topk(
                support_score,
                k=support_size,
                dim=-1,
                sorted=False,
            ).indices
            p_support = p.gather(-1, indices)
            p_plus_support = p_plus.gather(-1, indices)
            q_support = q.gather(-1, indices)
            support_student_mass = p_support.sum(dim=-1)
            support_student_plus_mass = p_plus_support.sum(dim=-1)
            support_teacher_mass = q_support.sum(dim=-1)

            def coarse(support: torch.Tensor) -> torch.Tensor:
                residual = (1.0 - support.sum(dim=-1, keepdim=True)).clamp_min(0.0)
                result = torch.cat((support, residual), dim=-1)
                return result / result.sum(dim=-1, keepdim=True).clamp_min(float(eps))

            p_coarse = coarse(p_support)
            p_plus_coarse = coarse(p_plus_support)
            q_coarse = coarse(q_support)
            probability = _direction_geometry(
                q_coarse,
                p_coarse,
                p_plus_coarse,
                float(eps),
            )
            hellinger = _direction_geometry(
                q_coarse.sqrt(),
                p_coarse.sqrt(),
                p_plus_coarse.sqrt(),
                float(eps),
            )
            for prefix, metrics in (("probability", probability), ("hellinger", hellinger)):
                for suffix, metric in zip(
                    (
                        "dot_product",
                        "teacher_norm_sq",
                        "budget_norm_sq",
                        "residual_norm_sq",
                        "cosine",
                        "projection_raw",
                        "closure_fraction",
                        "signed_progress",
                        "residual_ratio",
                        "step_ratio",
                        "orthogonal_ratio",
                    ),
                    metrics,
                ):
                    values[f"{prefix}_{suffix}"].append(metric)
            values["support_teacher_mass"].append(support_teacher_mass)
            values["support_student_mass"].append(support_student_mass)
            values["support_student_plus_mass"].append(support_student_plus_mass)

    return BudgetVectorGeometryMetrics(
        **{name: torch.cat(parts, dim=0).detach() for name, parts in values.items()}
    )


def compute_student_distribution_metrics(
    logits_b: torch.Tensor,
    logits_b_plus: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 16,
    teacher_top_k: int = 128,
    eps: float = 1e-12,
) -> StudentDistributionMetrics:
    """Compute exact student distances and teacher-support diagnostics in FP32.

    ``teacher_logratio_std`` is

        sqrt(Var_{v ~ q_full}[log p_b(v) - log p_b_plus(v)]).

    Centering removes the scalar log-ratio offset, while weighting by the
    teacher suppresses movement on vocabulary tails irrelevant to the target
    distribution. ``teacher_support_js`` is a coarse-grained JS divergence on
    the teacher's top-k tokens plus one residual-mass bucket.
    """

    logits_b = _flatten(logits_b)
    logits_b_plus = _flatten(logits_b_plus)
    teacher_logits = _flatten(teacher_logits)
    if logits_b.shape != logits_b_plus.shape or logits_b.shape != teacher_logits.shape:
        raise ValueError(
            "Student/student-plus/teacher logits must align: "
            f"{tuple(logits_b.shape)}, {tuple(logits_b_plus.shape)}, "
            f"{tuple(teacher_logits.shape)}."
        )
    if logits_b.shape[0] == 0:
        raise ValueError("At least one token position is required.")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    chunk_size = max(1, int(chunk_size))
    top_k = min(max(1, int(teacher_top_k)), int(logits_b.shape[-1]))

    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in StudentDistributionMetrics.__dataclass_fields__
    }
    with torch.no_grad():
        for start in range(0, int(logits_b.shape[0]), chunk_size):
            end = min(start + chunk_size, int(logits_b.shape[0]))
            log_p = F.log_softmax(logits_b[start:end].float() / temperature, dim=-1)
            log_p_plus = F.log_softmax(logits_b_plus[start:end].float() / temperature, dim=-1)
            log_q = F.log_softmax(teacher_logits[start:end].float() / temperature, dim=-1)
            p = log_p.exp()
            p_plus = log_p_plus.exp()
            q = log_q.exp()

            log_m = torch.logaddexp(log_p, log_p_plus) - torch.log(
                torch.tensor(2.0, device=log_p.device, dtype=log_p.dtype)
            )
            kl_b_to_plus = (p * (log_p - log_p_plus)).sum(dim=-1).clamp_min(0.0)
            kl_plus_to_b = (p_plus * (log_p_plus - log_p)).sum(dim=-1).clamp_min(0.0)
            js = (
                0.5 * (p * (log_p - log_m)).sum(dim=-1)
                + 0.5 * (p_plus * (log_p_plus - log_m)).sum(dim=-1)
            ).clamp_min(0.0)
            mixture_probs = 0.5 * (p + p_plus)
            mixture_entropy = (-(mixture_probs * log_m).sum(dim=-1)).clamp_min(0.0)
            normalized_js_mixture_entropy = js / (mixture_entropy + float(eps))
            hellinger_sq = (1.0 - torch.sqrt(p * p_plus).sum(dim=-1)).clamp(0.0, 1.0)
            total_variation = (0.5 * torch.abs(p - p_plus).sum(dim=-1)).clamp(0.0, 1.0)

            log_m_b_teacher = torch.logaddexp(log_p, log_q) - torch.log(
                torch.tensor(2.0, device=log_p.device, dtype=log_p.dtype)
            )
            log_m_plus_teacher = torch.logaddexp(log_p_plus, log_q) - torch.log(
                torch.tensor(2.0, device=log_p.device, dtype=log_p.dtype)
            )
            teacher_js_b = (
                0.5 * (p * (log_p - log_m_b_teacher)).sum(dim=-1)
                + 0.5 * (q * (log_q - log_m_b_teacher)).sum(dim=-1)
            ).clamp_min(0.0)
            teacher_js_b_plus = (
                0.5 * (p_plus * (log_p_plus - log_m_plus_teacher)).sum(dim=-1)
                + 0.5 * (q * (log_q - log_m_plus_teacher)).sum(dim=-1)
            ).clamp_min(0.0)

            # sqrt(JS), Hellinger, and Euclidean probability distance are
            # metrics. Dividing the adjacent-budget distance by the two legs
            # through the full teacher therefore gives a bounded, scale-free
            # measure of how much the budget intervention changes the current
            # teacher correction geometry.
            root_js = torch.sqrt(js)
            root_teacher_js_b = torch.sqrt(teacher_js_b)
            root_teacher_js_b_plus = torch.sqrt(teacher_js_b_plus)
            normalized_root_js = root_js / (
                root_teacher_js_b + root_teacher_js_b_plus + float(eps)
            )

            teacher_hellinger_b = torch.sqrt(
                (1.0 - torch.sqrt(p * q).sum(dim=-1)).clamp(0.0, 1.0)
            )
            teacher_hellinger_b_plus = torch.sqrt(
                (1.0 - torch.sqrt(p_plus * q).sum(dim=-1)).clamp(0.0, 1.0)
            )
            normalized_hellinger = torch.sqrt(hellinger_sq) / (
                teacher_hellinger_b + teacher_hellinger_b_plus + float(eps)
            )

            budget_l2 = torch.linalg.vector_norm(p - p_plus, dim=-1)
            teacher_l2_b = torch.linalg.vector_norm(p - q, dim=-1)
            teacher_l2_b_plus = torch.linalg.vector_norm(p_plus - q, dim=-1)
            normalized_l2 = budget_l2 / (
                teacher_l2_b + teacher_l2_b_plus + float(eps)
            )

            log_ratio = log_p - log_p_plus
            log_ratio_mean = (q * log_ratio).sum(dim=-1)
            centered = log_ratio - log_ratio_mean.unsqueeze(-1)
            logratio_rms = torch.sqrt((q * log_ratio.square()).sum(dim=-1).clamp_min(0.0))
            logratio_std = torch.sqrt((q * centered.square()).sum(dim=-1).clamp_min(0.0))
            logratio_l1 = (q * centered.abs()).sum(dim=-1)

            teacher_indices = torch.topk(q, k=top_k, dim=-1, sorted=False).indices
            p_support = p.gather(-1, teacher_indices)
            p_plus_support = p_plus.gather(-1, teacher_indices)
            p_coarse = torch.cat(
                [p_support, (1.0 - p_support.sum(dim=-1, keepdim=True)).clamp_min(0.0)], dim=-1
            )
            p_plus_coarse = torch.cat(
                [
                    p_plus_support,
                    (1.0 - p_plus_support.sum(dim=-1, keepdim=True)).clamp_min(0.0),
                ],
                dim=-1,
            )
            # Softmax round-off can make a coarse distribution miss unit mass
            # by a few ulps. Renormalization keeps the divergence well-defined.
            p_coarse = p_coarse / p_coarse.sum(dim=-1, keepdim=True).clamp_min(eps)
            p_plus_coarse = p_plus_coarse / p_plus_coarse.sum(dim=-1, keepdim=True).clamp_min(eps)
            m_coarse = 0.5 * (p_coarse + p_plus_coarse)
            support_js = 0.5 * (
                p_coarse * (p_coarse.clamp_min(eps).log() - m_coarse.clamp_min(eps).log())
            ).sum(dim=-1) + 0.5 * (
                p_plus_coarse
                * (p_plus_coarse.clamp_min(eps).log() - m_coarse.clamp_min(eps).log())
            ).sum(dim=-1)

            values["kl_b_to_plus"].append(kl_b_to_plus)
            values["kl_plus_to_b"].append(kl_plus_to_b)
            values["jeffreys"].append(0.5 * (kl_b_to_plus + kl_plus_to_b))
            values["js"].append(js)
            values["root_js"].append(root_js)
            values["hellinger_sq"].append(hellinger_sq)
            values["hellinger"].append(torch.sqrt(hellinger_sq))
            values["total_variation"].append(total_variation)
            values["mixture_entropy"].append(mixture_entropy)
            values["normalized_js_mixture_entropy"].append(
                normalized_js_mixture_entropy.clamp(0.0, 1.0)
            )
            values["teacher_logratio_rms"].append(logratio_rms)
            values["teacher_logratio_std"].append(logratio_std)
            values["teacher_logratio_l1_centered"].append(logratio_l1)
            values["teacher_support_js"].append(support_js.clamp_min(0.0))
            values["teacher_js_b"].append(teacher_js_b)
            values["teacher_js_b_plus"].append(teacher_js_b_plus)
            values["teacher_normalized_root_js"].append(normalized_root_js.clamp(0.0, 1.0))
            values["teacher_normalized_hellinger"].append(
                normalized_hellinger.clamp(0.0, 1.0)
            )
            values["teacher_normalized_l2"].append(normalized_l2.clamp(0.0, 1.0))

    return StudentDistributionMetrics(
        **{name: torch.cat(parts, dim=0).detach() for name, parts in values.items()}
    )


def compute_action_sensitivity_metrics(
    logits_b: torch.Tensor,
    logits_b_plus: torch.Tensor,
    generated_token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 32,
    eps: float = 1e-12,
) -> ActionSensitivityMetrics:
    """Compare budgets only on the on-policy generated-token event."""

    logits_b = _flatten(logits_b)
    logits_b_plus = _flatten(logits_b_plus)
    token_ids = generated_token_ids.detach().reshape(-1).to(device=logits_b.device)
    if logits_b.shape != logits_b_plus.shape:
        raise ValueError(
            f"Student budget logits must align: {logits_b.shape} vs {logits_b_plus.shape}."
        )
    if logits_b.shape[0] != token_ids.numel():
        raise ValueError(
            f"Generated token count must match logits: {token_ids.numel()} vs {logits_b.shape[0]}."
        )
    if bool((token_ids < 0).any()) or bool((token_ids >= logits_b.shape[-1]).any()):
        raise ValueError("Generated token IDs must be valid vocabulary indices.")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    deltas: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, token_ids.numel(), chunk_size):
            end = min(start + chunk_size, token_ids.numel())
            indices = token_ids[start:end].unsqueeze(-1)
            log_p = F.log_softmax(logits_b[start:end].float() / float(temperature), dim=-1)
            log_p_plus = F.log_softmax(
                logits_b_plus[start:end].float() / float(temperature), dim=-1
            )
            action_log_p = log_p.gather(-1, indices).squeeze(-1)
            action_log_p_plus = log_p_plus.gather(-1, indices).squeeze(-1)
            action_p = action_log_p.exp().clamp(0.0, 1.0)
            action_p_plus = action_log_p_plus.exp().clamp(0.0, 1.0)
            coarse_p = torch.stack((action_p, 1.0 - action_p), dim=-1).clamp_min(float(eps))
            coarse_p_plus = torch.stack(
                (action_p_plus, 1.0 - action_p_plus), dim=-1
            ).clamp_min(float(eps))
            coarse_p = coarse_p / coarse_p.sum(dim=-1, keepdim=True)
            coarse_p_plus = coarse_p_plus / coarse_p_plus.sum(dim=-1, keepdim=True)
            mixture = 0.5 * (coarse_p + coarse_p_plus)
            js = 0.5 * (
                coarse_p * (coarse_p.log() - mixture.log())
            ).sum(dim=-1) + 0.5 * (
                coarse_p_plus * (coarse_p_plus.log() - mixture.log())
            ).sum(dim=-1)
            deltas.append(action_log_p_plus - action_log_p)
            divergences.append(js.clamp(0.0, math.log(2.0)))
    delta = torch.cat(deltas, dim=0).detach()
    divergence = torch.cat(divergences, dim=0).detach()
    return ActionSensitivityMetrics(
        action_logprob_delta=delta,
        budget_advantage=delta.clamp_min(0.0),
        action_abs_logprob_gap=delta.abs(),
        action_bernoulli_js=divergence,
        action_root_bernoulli_js=torch.sqrt(divergence).detach(),
    )


def compute_action_sensitivity_metrics(
    logits_b: torch.Tensor,
    logits_b_plus: torch.Tensor,
    generated_token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 32,
    eps: float = 1e-12,
) -> ActionSensitivityMetrics:
    """Compare budgets only on the on-policy generated-token event.

    Each vocabulary distribution is coarsened to ``{generated token, other}``
    before computing JS. This removes movement on vocabulary tails that cannot
    change the action actually visited by the fixed rollout.
    """

    logits_b = _flatten(logits_b)
    logits_b_plus = _flatten(logits_b_plus)
    token_ids = generated_token_ids.detach().reshape(-1).to(device=logits_b.device)
    if logits_b.shape != logits_b_plus.shape:
        raise ValueError(
            f"Student budget logits must align: {logits_b.shape} vs {logits_b_plus.shape}."
        )
    if logits_b.shape[0] != token_ids.numel():
        raise ValueError(
            f"Generated token count must match logits: {token_ids.numel()} vs {logits_b.shape[0]}."
        )
    if bool((token_ids < 0).any()) or bool((token_ids >= logits_b.shape[-1]).any()):
        raise ValueError("Generated token IDs must be valid vocabulary indices.")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    deltas: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, token_ids.numel(), max(1, int(chunk_size))):
            end = min(start + max(1, int(chunk_size)), token_ids.numel())
            indices = token_ids[start:end].unsqueeze(-1)
            log_p = F.log_softmax(logits_b[start:end].float() / float(temperature), dim=-1)
            log_p_plus = F.log_softmax(
                logits_b_plus[start:end].float() / float(temperature), dim=-1
            )
            action_log_p = log_p.gather(-1, indices).squeeze(-1)
            action_log_p_plus = log_p_plus.gather(-1, indices).squeeze(-1)
            action_p = action_log_p.exp().clamp(0.0, 1.0)
            action_p_plus = action_log_p_plus.exp().clamp(0.0, 1.0)
            coarse_p = torch.stack((action_p, 1.0 - action_p), dim=-1).clamp_min(float(eps))
            coarse_p_plus = torch.stack(
                (action_p_plus, 1.0 - action_p_plus), dim=-1
            ).clamp_min(float(eps))
            coarse_p = coarse_p / coarse_p.sum(dim=-1, keepdim=True)
            coarse_p_plus = coarse_p_plus / coarse_p_plus.sum(dim=-1, keepdim=True)
            mixture = 0.5 * (coarse_p + coarse_p_plus)
            js = 0.5 * (
                coarse_p * (coarse_p.log() - mixture.log())
            ).sum(dim=-1) + 0.5 * (
                coarse_p_plus * (coarse_p_plus.log() - mixture.log())
            ).sum(dim=-1)
            deltas.append(action_log_p_plus - action_log_p)
            divergences.append(js.clamp(0.0, math.log(2.0)))
    delta = torch.cat(deltas, dim=0).detach()
    divergence = torch.cat(divergences, dim=0).detach()
    return ActionSensitivityMetrics(
        action_logprob_delta=delta,
        budget_advantage=delta.clamp_min(0.0),
        action_abs_logprob_gap=delta.abs(),
        action_bernoulli_js=divergence,
        action_root_bernoulli_js=torch.sqrt(divergence).detach(),
    )
