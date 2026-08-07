from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeBudgetWeights:
    sensitivity: torch.Tensor
    tau: torch.Tensor
    robustness: torch.Tensor
    raw_weight: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class TeacherGapPersistenceWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    tau_teacher_gap: torch.Tensor
    rescue_fraction: torch.Tensor
    persistence: torch.Tensor
    confidence: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class SymmetricTeacherGapStabilityWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    signed_budget_change: torch.Tensor
    normalized_budget_change: torch.Tensor
    robustness: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualRescueAmplificationWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    rescue_fraction: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualTeachabilityMixtureWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    teacher_gap_rank: torch.Tensor
    rescue_fraction: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualTeachabilityModulationWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    teacher_gap_rank: torch.Tensor
    rescue_fraction: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ConditionalRescueResidualWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    teacher_gap_rank: torch.Tensor
    rescue_fraction: torch.Tensor
    expected_rescue: torch.Tensor
    rescue_residual: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class GroupedKLMassWeights:
    ranking_signal: torch.Tensor
    high_group_mask: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetConsistentRankWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    persistent_gap: torch.Tensor
    persistent_rank: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetResidualHardnessWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    persistent_gap: torch.Tensor
    teacher_gap_rank: torch.Tensor
    persistent_rank: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualBudgetBridge:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    bridge_gap: torch.Tensor
    tau_teacher_gap: torch.Tensor
    rescue_fraction: torch.Tensor
    confidence: torch.Tensor
    bridge_fraction: torch.Tensor
    loss_mass_scale: torch.Tensor
    full_teacher_weight: torch.Tensor
    bridge_teacher_weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetContrastiveGate:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    tau_teacher_gap: torch.Tensor
    rescue_fraction: torch.Tensor
    confidence: torch.Tensor
    shaping_strength: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetGradientAlignedBridgeGate:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    gradient_alignment: torch.Tensor
    positive_alignment: torch.Tensor
    tau_teacher_gap: torch.Tensor
    rescue_fraction: torch.Tensor
    confidence: torch.Tensor
    bridge_fraction: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class CounterfactualGradientResidualGate:
    teacher_gap_b: torch.Tensor
    teacher_gap_b_plus: torch.Tensor
    gradient_alignment: torch.Tensor
    raw_projection_coefficient: torch.Tensor
    clipped_projection_coefficient: torch.Tensor
    budget_rescue_indicator: torch.Tensor
    cancellation_coefficient: torch.Tensor
    residual_gradient_norm_ratio: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetTangentResidualWeights:
    teacher_gap_b: torch.Tensor
    gradient_alignment: torch.Tensor
    budget_explained_fraction: torch.Tensor
    budget_residual_fraction: torch.Tensor
    teacher_gap_rank: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetGradientConsensusWeights:
    teacher_gap_b: torch.Tensor
    teacher_gap_rank: torch.Tensor
    gradient_consensus: torch.Tensor
    gradient_norm_consistency: torch.Tensor
    invariant_priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class BudgetCounterfactualTeachabilityWeights:
    teacher_gap_b: torch.Tensor
    gradient_alignment: torch.Tensor
    budget_explained_fraction: torch.Tensor
    budget_residual_fraction: torch.Tensor
    teacher_support_coverage: torch.Tensor
    teacher_gap_rank: torch.Tensor
    support_coverage_rank: torch.Tensor
    priority: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class LossMassNormalization:
    reference_loss: torch.Tensor
    candidate_loss: torch.Tensor
    raw_weight: torch.Tensor
    loss_mass_scale: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


def counterfactual_cancellation_strength(
    base_strength: float,
    *,
    schedule: str = "constant",
    progress_step: int | None = None,
    total_steps: int | None = None,
    decay_fraction: float = 0.5,
) -> float:
    """Resolve a deterministic early counterfactual-gradient curriculum."""

    strength = float(base_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"base_strength must be in [0, 1], got {base_strength}.")
    normalized_schedule = str(schedule).strip().lower()
    if normalized_schedule == "constant":
        return strength
    if normalized_schedule != "linear_to_zero":
        raise ValueError(f"Unsupported cancellation schedule: {schedule!r}.")
    if progress_step is None or total_steps is None:
        raise ValueError("linear_to_zero requires progress_step and total_steps.")
    if int(total_steps) <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if int(progress_step) < 0:
        raise ValueError(f"progress_step must be nonnegative, got {progress_step}.")
    fraction = float(decay_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"decay_fraction must be in (0, 1], got {decay_fraction}.")
    cutoff = max(1.0, fraction * float(total_steps))
    remaining = 1.0 - min(float(progress_step) / cutoff, 1.0)
    return strength * remaining


def generated_token_valid_mask(generated_ids: torch.Tensor) -> torch.Tensor:
    """Mark emitted positions as valid without treating Qwen's EOS/pad alias as padding."""

    token_ids = generated_ids.detach().reshape(-1)
    if token_ids.numel() <= 0:
        raise ValueError("At least one generated token is required.")
    return torch.ones_like(token_ids, dtype=torch.bool)


def native_budget_robustness_weights(
    sensitivity: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> NativeBudgetWeights:
    """Convert detached native-budget sensitivity into per-sample mean-one weights."""

    with torch.no_grad():
        values = sensitivity.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=values.device, dtype=torch.bool).reshape(-1)
        if values.shape != valid.shape:
            raise ValueError(f"Sensitivity and valid mask must align: {values.shape} vs {valid.shape}.")
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Sensitivity KL must be finite on all valid positions.")
        if (values[valid] < -1e-6).any():
            raise ValueError("Sensitivity KL must be nonnegative.")
        values = values.clamp_min(0.0)
        tau = torch.quantile(values[valid], 0.5)
        robustness = torch.zeros_like(values)
        robustness[valid] = tau / (tau + values[valid] + float(eps))
        raw_weight = torch.zeros_like(values)
        raw_weight[valid] = 1.0 + robustness[valid]
        mean_raw = raw_weight[valid].mean()
        if not torch.isfinite(mean_raw) or float(mean_raw) <= 0.0:
            raise FloatingPointError(f"Invalid mean raw weight: {float(mean_raw)}")
        weight = torch.zeros_like(values)
        weight[valid] = raw_weight[valid] / mean_raw
    return NativeBudgetWeights(
        sensitivity=values.detach(),
        tau=tau.detach(),
        robustness=robustness.detach(),
        raw_weight=raw_weight.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def teacher_gap_persistence_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> TeacherGapPersistenceWeights:
    """Prioritize hard teacher gaps that an extra visual budget does not rescue.

    The final scale preserves the detached, per-sample KL mass exactly. This
    keeps the scalar loss scale matched to vanilla OPSD while redistributing
    gradients across valid generated-token positions.
    """

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b = teacher_gap_b.detach().float().reshape(-1)
        gap_b_plus = teacher_gap_b_plus.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap_b.device, dtype=torch.bool).reshape(-1)
        if gap_b.shape != gap_b_plus.shape or gap_b.shape != valid.shape:
            raise ValueError(
                "Teacher gaps and valid mask must align: "
                f"{gap_b.shape}, {gap_b_plus.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(gap_b[valid]).all() or not torch.isfinite(gap_b_plus[valid]).all():
            raise ValueError("Teacher KL must be finite on all valid positions.")
        if (gap_b[valid] < -1e-6).any() or (gap_b_plus[valid] < -1e-6).any():
            raise ValueError("Teacher KL must be nonnegative.")

        gap_b = gap_b.clamp_min(0.0)
        gap_b_plus = gap_b_plus.clamp_min(0.0)
        tau = torch.quantile(gap_b[valid], 0.5)

        rescue = torch.zeros_like(gap_b)
        rescue[valid] = ((gap_b[valid] - gap_b_plus[valid]) / (gap_b[valid] + float(eps))).clamp(0.0, 1.0)
        persistence = torch.zeros_like(gap_b)
        persistence[valid] = 1.0 - rescue[valid]
        confidence = torch.zeros_like(gap_b)
        confidence[valid] = gap_b[valid] / (gap_b[valid] + tau + float(eps))
        priority = torch.zeros_like(gap_b)
        priority[valid] = persistence[valid] * confidence[valid]

        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]
        unweighted_mass = gap_b[valid].sum()
        raw_weighted_mass = (raw_weight[valid] * gap_b[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
            weight = torch.zeros_like(gap_b)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(f"Invalid raw weighted KL mass: {float(raw_weighted_mass)}")
            loss_mass_scale = unweighted_mass / raw_weighted_mass
            weight = torch.zeros_like(gap_b)
            weight[valid] = raw_weight[valid] * loss_mass_scale

        normalized_mass = (weight[valid] * gap_b[valid]).sum()
        if not torch.allclose(normalized_mass, unweighted_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError(
                "KL-mass normalization failed: "
                f"weighted={float(normalized_mass)}, unweighted={float(unweighted_mass)}."
            )

    return TeacherGapPersistenceWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        tau_teacher_gap=tau.detach(),
        rescue_fraction=rescue.detach(),
        persistence=persistence.detach(),
        confidence=confidence.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def symmetric_teacher_gap_stability_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.25,
    eps: float = 1e-8,
) -> SymmetricTeacherGapStabilityWeights:
    """Downweight tokens whose teacher gap changes under a nearby visual budget.

    The symmetric relative change removes the absolute KL scale. The detached
    KL-mass normalization preserves each sample's vanilla OPSD scalar loss and
    changes only the distribution of gradients across generated tokens.
    """

    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError(f"alpha must be in [0, 1); got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b = teacher_gap_b.detach().float().reshape(-1)
        gap_b_plus = teacher_gap_b_plus.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap_b.device, dtype=torch.bool).reshape(-1)
        if gap_b.shape != gap_b_plus.shape or gap_b.shape != valid.shape:
            raise ValueError(
                "Teacher gaps and valid mask must align: "
                f"{gap_b.shape}, {gap_b_plus.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(gap_b[valid]).all() or not torch.isfinite(gap_b_plus[valid]).all():
            raise ValueError("Teacher KL must be finite on all valid positions.")
        if (gap_b[valid] < -1e-6).any() or (gap_b_plus[valid] < -1e-6).any():
            raise ValueError("Teacher KL must be nonnegative.")

        gap_b = gap_b.clamp_min(0.0)
        gap_b_plus = gap_b_plus.clamp_min(0.0)
        signed_change = torch.zeros_like(gap_b)
        signed_change[valid] = (gap_b[valid] - gap_b_plus[valid]) / (
            gap_b[valid] + gap_b_plus[valid] + float(eps)
        )
        normalized_change = torch.zeros_like(gap_b)
        normalized_change[valid] = signed_change[valid].abs().clamp(0.0, 1.0)
        robustness = torch.zeros_like(gap_b)
        robustness[valid] = 1.0 - normalized_change[valid]

        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 - float(alpha) * normalized_change[valid]
        unweighted_mass = gap_b[valid].sum()
        raw_weighted_mass = (raw_weight[valid] * gap_b[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
            weight = torch.zeros_like(gap_b)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(f"Invalid raw weighted KL mass: {float(raw_weighted_mass)}")
            loss_mass_scale = unweighted_mass / raw_weighted_mass
            weight = torch.zeros_like(gap_b)
            weight[valid] = raw_weight[valid] * loss_mass_scale

        normalized_mass = (weight[valid] * gap_b[valid]).sum()
        if not torch.allclose(normalized_mass, unweighted_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError(
                "KL-mass normalization failed: "
                f"weighted={float(normalized_mass)}, unweighted={float(unweighted_mass)}."
            )

    return SymmetricTeacherGapStabilityWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        signed_budget_change=signed_change.detach(),
        normalized_budget_change=normalized_change.detach(),
        robustness=robustness.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_rescue_amplification_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> CounterfactualRescueAmplificationWeights:
    """Amplify full-teacher OPSD where extra visual budget closes the gap.

    The adjacent native budget is used only as a detached teachability probe.
    It never replaces or mixes the full-token teacher target. A bounded rescue
    fraction mildly amplifies the original forward-KL gradient, while detached
    per-sample KL-mass normalization preserves vanilla OPSD's scalar loss.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid])
            / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * rescue[valid]
        normalized = normalize_candidate_loss_mass(
            gap_b,
            gap_b,
            raw_weight,
            valid,
            eps=eps,
        )
    return CounterfactualRescueAmplificationWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        rescue_fraction=rescue.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_teachability_mixture_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    rescue_mix: float = 0.1,
    eps: float = 1e-8,
) -> CounterfactualTeachabilityMixtureWeights:
    """Prioritize hard corrections with a mild adjacent-budget teachability signal.

    The deployed-budget teacher gap supplies the primary within-response
    difficulty rank. The positive fraction closed by ``b_plus`` is a bounded
    counterfactual probe of whether the same model can absorb that correction
    when given slightly more visual evidence. The full-token teacher remains
    the sole target and detached KL-mass normalization preserves the vanilla
    OPSD scalar loss for every sample.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if not 0.0 <= float(rescue_mix) <= 1.0:
        raise ValueError(f"rescue_mix must be in [0, 1]; got {rescue_mix}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        gap_rank = torch.zeros_like(gap_b)
        gap_rank[valid] = _ordinal_percentile_rank(gap_b[valid])
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid])
            / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)
        priority = torch.zeros_like(gap_b)
        priority[valid] = (
            (1.0 - float(rescue_mix)) * gap_rank[valid]
            + float(rescue_mix) * rescue[valid]
        )
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]
        normalized = normalize_candidate_loss_mass(
            gap_b,
            gap_b,
            raw_weight,
            valid,
            eps=eps,
        )

    return CounterfactualTeachabilityMixtureWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        teacher_gap_rank=gap_rank.detach(),
        rescue_fraction=rescue.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_teachability_modulation_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    rescue_modulation: float = 0.1,
    eps: float = 1e-8,
) -> CounterfactualTeachabilityModulationWeights:
    """Modulate hard full-teacher corrections by adjacent-budget teachability.

    The deployed-budget teacher-gap rank remains the base priority. Positive
    gap closure under the native ``b_plus`` intervention can only increase
    that priority by a bounded multiplicative factor, so an easy token cannot
    outrank a hard correction solely because its relative rescue is large.
    The probe and all weights are detached, the full-token teacher remains the
    sole target, and per-sample KL-mass normalization preserves vanilla OPSD's
    scalar loss exactly.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if not 0.0 <= float(rescue_modulation) <= 1.0:
        raise ValueError(
            "rescue_modulation must be in [0, 1]; "
            f"got {rescue_modulation}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        gap_rank = torch.zeros_like(gap_b)
        gap_rank[valid] = _ordinal_percentile_rank(gap_b[valid])
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid])
            / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)
        priority = torch.zeros_like(gap_b)
        priority[valid] = gap_rank[valid] * (
            1.0 + float(rescue_modulation) * rescue[valid]
        )
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]
        normalized = normalize_candidate_loss_mass(
            gap_b,
            gap_b,
            raw_weight,
            valid,
            eps=eps,
        )

    return CounterfactualTeachabilityModulationWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        teacher_gap_rank=gap_rank.detach(),
        rescue_fraction=rescue.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def conditional_rescue_residual_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.1,
    difficulty_bins: int = 5,
    eps: float = 1e-8,
) -> ConditionalRescueResidualWeights:
    """Reweight only budget rescue unexplained by within-response difficulty.

    Raw positive rescue is strongly correlated with the deployed-budget
    teacher gap. We remove that first-order confound by subtracting the median
    rescue in each within-response teacher-gap quantile. The residual is then
    centered over valid tokens and used as a small symmetric perturbation of
    unit weights. Detached KL-mass normalization preserves vanilla OPSD's
    scalar loss exactly for every response.
    """

    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError(f"alpha must be in [0, 1); got {alpha}.")
    if int(difficulty_bins) < 2:
        raise ValueError(f"difficulty_bins must be at least 2; got {difficulty_bins}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        gap_rank = torch.zeros_like(gap_b)
        gap_rank[valid] = _ordinal_percentile_rank(gap_b[valid])
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid])
            / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)

        valid_ranks = gap_rank[valid]
        bins = (
            torch.ceil(valid_ranks * float(difficulty_bins)).long() - 1
        ).clamp(0, int(difficulty_bins) - 1)
        expected = torch.zeros_like(gap_b)
        valid_expected = torch.empty_like(rescue[valid])
        valid_rescue = rescue[valid]
        for bin_index in range(int(difficulty_bins)):
            in_bin = bins == bin_index
            if in_bin.any():
                valid_expected[in_bin] = valid_rescue[in_bin].median()
        expected[valid] = valid_expected

        residual = torch.zeros_like(gap_b)
        centered = valid_rescue - valid_expected
        centered = (centered - centered.mean()).clamp(-1.0, 1.0)
        residual[valid] = centered
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * residual[valid]
        normalized = normalize_candidate_loss_mass(
            gap_b,
            gap_b,
            raw_weight,
            valid,
            eps=eps,
        )

    return ConditionalRescueResidualWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        teacher_gap_rank=gap_rank.detach(),
        rescue_fraction=rescue.detach(),
        expected_rescue=expected.detach(),
        rescue_residual=residual.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def grouped_kl_mass_weights(
    reference_loss: torch.Tensor,
    ranking_signal: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_fraction: float = 0.2,
    high_group_mass: float = 0.5,
    eps: float = 1e-8,
) -> GroupedKLMassWeights:
    """Allocate fixed loss mass to a ranked token group without scale drift.

    This is the token-group normalization used for the native-budget rescue
    candidate and its teacher-gap-only control. The detached ranking signal
    chooses the high group; the full-teacher KL remains the sole objective.
    A final detached KL-mass correction makes the forward scalar exactly match
    vanilla OPSD for each response while retaining the grouped gradient ratio.
    """

    if not 0.0 < float(top_fraction) < 1.0:
        raise ValueError(f"top_fraction must be in (0, 1); got {top_fraction}.")
    if not 0.0 < float(high_group_mass) < 1.0:
        raise ValueError(
            f"high_group_mass must be in (0, 1); got {high_group_mass}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        loss = reference_loss.detach().float().reshape(-1).clamp_min(0.0)
        signal = ranking_signal.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=loss.device, dtype=torch.bool).reshape(-1)
        if loss.shape != signal.shape or loss.shape != valid.shape:
            raise ValueError(
                "Reference loss, ranking signal, and valid mask must align: "
                f"{loss.shape}, {signal.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(loss[valid]).all() or not torch.isfinite(signal[valid]).all():
            raise ValueError("Grouped loss and ranking signal must be finite.")
        if (loss[valid] < -1e-6).any():
            raise ValueError("Reference KL must be nonnegative.")

        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        n_valid = int(valid_indices.numel())
        n_high = (
            max(1, min(n_valid - 1, math.ceil(float(top_fraction) * n_valid)))
            if n_valid > 1
            else 1
        )
        order = torch.argsort(signal[valid], descending=True, stable=True)
        high_indices = valid_indices[order[:n_high]]
        high = torch.zeros_like(valid)
        high[high_indices] = True
        low = valid & ~high

        raw_weight = torch.zeros_like(loss)
        if low.any():
            raw_weight[high] = float(high_group_mass) * n_valid / float(n_high)
            raw_weight[low] = (
                (1.0 - float(high_group_mass))
                * n_valid
                / float(n_valid - n_high)
            )
        else:
            raw_weight[high] = 1.0
        normalized = normalize_candidate_loss_mass(
            loss,
            loss,
            raw_weight,
            valid,
            eps=eps,
        )

    return GroupedKLMassWeights(
        ranking_signal=signal.detach(),
        high_group_mask=high.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def _validate_gap_inputs(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gap_b = teacher_gap_b.detach().float().reshape(-1)
    gap_b_plus = teacher_gap_b_plus.detach().float().reshape(-1)
    valid = valid_mask.detach().to(device=gap_b.device, dtype=torch.bool).reshape(-1)
    if gap_b.shape != gap_b_plus.shape or gap_b.shape != valid.shape:
        raise ValueError(
            "Teacher gaps and valid mask must align: "
            f"{gap_b.shape}, {gap_b_plus.shape}, {valid.shape}."
        )
    if not valid.any():
        raise ValueError("At least one valid generated-token position is required.")
    if not torch.isfinite(gap_b[valid]).all() or not torch.isfinite(gap_b_plus[valid]).all():
        raise ValueError("Teacher KL must be finite on all valid positions.")
    if (gap_b[valid] < -1e-6).any() or (gap_b_plus[valid] < -1e-6).any():
        raise ValueError("Teacher KL must be nonnegative.")
    return gap_b.clamp_min(0.0), gap_b_plus.clamp_min(0.0), valid


def _ordinal_percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic ranks in (0, 1], used only on detached signals."""

    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("Rank input must be a nonempty vector.")
    order = torch.argsort(values, stable=True)
    ordinal = torch.empty_like(order)
    ordinal[order] = torch.arange(1, values.numel() + 1, device=values.device)
    return ordinal.float() / float(values.numel())


def normalize_candidate_loss_mass(
    reference_loss: torch.Tensor,
    candidate_loss: torch.Tensor,
    raw_weight: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> LossMassNormalization:
    """Match a detached weighted candidate objective to reference KL mass."""

    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    with torch.no_grad():
        reference = reference_loss.detach().float().reshape(-1)
        candidate = candidate_loss.detach().float().reshape(-1)
        raw = raw_weight.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=reference.device, dtype=torch.bool).reshape(-1)
        if not (reference.shape == candidate.shape == raw.shape == valid.shape):
            raise ValueError(
                "Reference loss, candidate loss, raw weight, and valid mask must align: "
                f"{reference.shape}, {candidate.shape}, {raw.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid token is required for loss-mass normalization.")
        for name, values in (("reference", reference), ("candidate", candidate), ("weight", raw)):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} values must be finite on valid positions.")
        if (reference[valid] < -1e-6).any() or (candidate[valid] < -1e-6).any():
            raise ValueError("KL losses must be nonnegative on valid positions.")
        if (raw[valid] <= 0.0).any():
            raise ValueError("Raw weights must be positive on valid positions.")
        reference = reference.clamp_min(0.0)
        candidate = candidate.clamp_min(0.0)
        reference_mass = reference[valid].sum()
        candidate_mass = (raw[valid] * candidate[valid]).sum()
        if float(reference_mass) <= float(eps):
            scale = torch.zeros((), device=reference.device, dtype=torch.float32)
            weight = torch.zeros_like(reference)
        else:
            if not torch.isfinite(candidate_mass) or float(candidate_mass) <= 0.0:
                raise FloatingPointError(f"Invalid weighted candidate loss mass: {float(candidate_mass)}")
            scale = reference_mass / candidate_mass
            weight = torch.zeros_like(reference)
            weight[valid] = raw[valid] * scale
        normalized = (weight[valid] * candidate[valid]).sum()
        if not torch.allclose(normalized, reference_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError(
                "Candidate loss-mass normalization failed: "
                f"normalized={float(normalized)}, reference={float(reference_mass)}."
            )
    return LossMassNormalization(
        reference_loss=reference.detach(),
        candidate_loss=candidate.detach(),
        raw_weight=raw.detach(),
        loss_mass_scale=scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_consistent_rank_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> BudgetConsistentRankWeights:
    """Prioritize divergence that remains high at both native token budgets.

    ``min(K_b, K_b_plus)`` is a conservative lower bound on the teacher gap
    across the two budgets. Per-sample ranks remove absolute KL-scale and
    response-length confounds. KL-mass normalization keeps the forward scalar
    loss identical to vanilla OPSD while changing token-level gradients.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(teacher_gap_b, teacher_gap_b_plus, valid_mask)
        persistent_gap = torch.zeros_like(gap_b)
        persistent_gap[valid] = torch.minimum(gap_b[valid], gap_b_plus[valid])
        persistent_rank = torch.zeros_like(gap_b)
        persistent_rank[valid] = _ordinal_percentile_rank(persistent_gap[valid])
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * persistent_rank[valid]

        unweighted_mass = gap_b[valid].sum()
        raw_weighted_mass = (raw_weight[valid] * gap_b[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
            weight = torch.zeros_like(gap_b)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(f"Invalid raw weighted KL mass: {float(raw_weighted_mass)}")
            loss_mass_scale = unweighted_mass / raw_weighted_mass
            weight = torch.zeros_like(gap_b)
            weight[valid] = raw_weight[valid] * loss_mass_scale
        if not torch.allclose(
            (weight[valid] * gap_b[valid]).sum(), unweighted_mass, rtol=2e-6, atol=2e-7
        ):
            raise FloatingPointError("Budget-consistent rank KL-mass normalization failed.")

    return BudgetConsistentRankWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        persistent_gap=persistent_gap.detach(),
        persistent_rank=persistent_rank.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_residual_hardness_weights(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    persistence_mix: float = 0.1,
    eps: float = 1e-8,
) -> BudgetResidualHardnessWeights:
    """Reweight hard tokens while requiring evidence across adjacent budgets.

    The deployed-budget teacher gap remains the primary signal. A small,
    preregistered persistence term favors gaps that remain large after adding
    native visual tokens. Per-sample ranks make the two terms commensurate;
    KL-mass normalization preserves vanilla OPSD's detached scalar loss.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if not 0.0 <= float(persistence_mix) <= 1.0:
        raise ValueError(
            f"persistence_mix must be in [0, 1]; got {persistence_mix}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        persistent_gap = torch.zeros_like(gap_b)
        persistent_gap[valid] = (
            2.0
            * gap_b[valid]
            * gap_b_plus[valid]
            / (gap_b[valid] + gap_b_plus[valid] + float(eps))
        )
        teacher_gap_rank = torch.zeros_like(gap_b)
        teacher_gap_rank[valid] = _ordinal_percentile_rank(gap_b[valid])
        persistent_rank = torch.zeros_like(gap_b)
        persistent_rank[valid] = _ordinal_percentile_rank(persistent_gap[valid])
        priority = torch.zeros_like(gap_b)
        priority[valid] = (
            (1.0 - float(persistence_mix)) * teacher_gap_rank[valid]
            + float(persistence_mix) * persistent_rank[valid]
        )
        raw_weight = torch.zeros_like(gap_b)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]

        unweighted_mass = gap_b[valid].sum()
        raw_weighted_mass = (raw_weight[valid] * gap_b[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
            weight = torch.zeros_like(gap_b)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(raw_weighted_mass) or float(raw_weighted_mass) <= 0.0:
                raise FloatingPointError(
                    f"Invalid budget-residual weighted KL mass: {float(raw_weighted_mass)}"
                )
            loss_mass_scale = unweighted_mass / raw_weighted_mass
            weight = torch.zeros_like(gap_b)
            weight[valid] = raw_weight[valid] * loss_mass_scale
        if not torch.allclose(
            (weight[valid] * gap_b[valid]).sum(),
            unweighted_mass,
            rtol=2e-6,
            atol=2e-7,
        ):
            raise FloatingPointError(
                "Budget-residual hardness KL-mass normalization failed."
            )

    return BudgetResidualHardnessWeights(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        persistent_gap=persistent_gap.detach(),
        teacher_gap_rank=teacher_gap_rank.detach(),
        persistent_rank=persistent_rank.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_budget_bridge(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    bridge_gap: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_bridge_fraction: float = 0.5,
    eps: float = 1e-8,
) -> CounterfactualBudgetBridge:
    """Route budget-rescuable tokens toward a proximal higher-budget teacher.

    The bridge is enabled only where the native ``b_plus`` student is closer
    to the full teacher than the deployed-budget student. The routed loss is
    KL-mass normalized so its detached scalar value equals vanilla OPSD.
    """

    if not 0.0 <= float(max_bridge_fraction) <= 1.0:
        raise ValueError(
            f"max_bridge_fraction must be in [0, 1]; got {max_bridge_fraction}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(teacher_gap_b, teacher_gap_b_plus, valid_mask)
        bridge = bridge_gap.detach().float().reshape(-1)
        if bridge.shape != gap_b.shape:
            raise ValueError(f"Bridge gap must align with teacher gaps: {bridge.shape} vs {gap_b.shape}.")
        if not torch.isfinite(bridge[valid]).all() or (bridge[valid] < -1e-6).any():
            raise ValueError("Bridge KL must be finite and nonnegative on valid positions.")
        bridge = bridge.clamp_min(0.0)

        tau = torch.quantile(gap_b[valid], 0.5)
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = ((gap_b[valid] - gap_b_plus[valid]) / (gap_b[valid] + float(eps))).clamp(0.0, 1.0)
        confidence = torch.zeros_like(gap_b)
        confidence[valid] = gap_b[valid] / (gap_b[valid] + tau + float(eps))
        bridge_fraction = torch.zeros_like(gap_b)
        bridge_fraction[valid] = float(max_bridge_fraction) * rescue[valid] * confidence[valid]

        routed = torch.zeros_like(gap_b)
        routed[valid] = (
            (1.0 - bridge_fraction[valid]) * gap_b[valid]
            + bridge_fraction[valid] * bridge[valid]
        )
        unweighted_mass = gap_b[valid].sum()
        routed_mass = routed[valid].sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap_b.device, dtype=torch.float32)
        else:
            if not torch.isfinite(routed_mass) or float(routed_mass) <= 0.0:
                raise FloatingPointError(f"Invalid routed KL mass: {float(routed_mass)}")
            loss_mass_scale = unweighted_mass / routed_mass
        full_teacher_weight = torch.zeros_like(gap_b)
        bridge_teacher_weight = torch.zeros_like(gap_b)
        full_teacher_weight[valid] = loss_mass_scale * (1.0 - bridge_fraction[valid])
        bridge_teacher_weight[valid] = loss_mass_scale * bridge_fraction[valid]
        normalized = (
            full_teacher_weight[valid] * gap_b[valid]
            + bridge_teacher_weight[valid] * bridge[valid]
        ).sum()
        if not torch.allclose(normalized, unweighted_mass, rtol=2e-6, atol=2e-7):
            raise FloatingPointError("Counterfactual bridge KL-mass normalization failed.")

    return CounterfactualBudgetBridge(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        bridge_gap=bridge.detach(),
        tau_teacher_gap=tau.detach(),
        rescue_fraction=rescue.detach(),
        confidence=confidence.detach(),
        bridge_fraction=bridge_fraction.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        full_teacher_weight=full_teacher_weight.detach(),
        bridge_teacher_weight=bridge_teacher_weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_contrastive_gate(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta_max: float = 0.5,
    eps: float = 1e-8,
) -> BudgetContrastiveGate:
    """Gate budget-contrastive target shaping by observed teacher-gap rescue."""

    if not 0.0 <= float(beta_max) <= 2.0:
        raise ValueError(f"beta_max must be in [0, 2], got {beta_max}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(teacher_gap_b, teacher_gap_b_plus, valid_mask)
        tau = torch.quantile(gap_b[valid], 0.5)
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = ((gap_b[valid] - gap_b_plus[valid]) / (gap_b[valid] + float(eps))).clamp(0.0, 1.0)
        confidence = torch.zeros_like(gap_b)
        confidence[valid] = gap_b[valid] / (gap_b[valid] + tau + float(eps))
        strength = torch.zeros_like(gap_b)
        strength[valid] = float(beta_max) * rescue[valid] * confidence[valid]
    return BudgetContrastiveGate(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        tau_teacher_gap=tau.detach(),
        rescue_fraction=rescue.detach(),
        confidence=confidence.detach(),
        shaping_strength=strength.detach(),
        valid_mask=valid.detach(),
    )


def budget_gradient_aligned_bridge_gate(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    gradient_alignment: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_bridge_fraction: float = 0.5,
    eps: float = 1e-8,
) -> BudgetGradientAlignedBridgeGate:
    """Gate a proximal budget bridge by causal improvement and gradient agreement."""

    if not 0.0 <= float(max_bridge_fraction) <= 1.0:
        raise ValueError(
            f"max_bridge_fraction must be in [0, 1]; got {max_bridge_fraction}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        alignment = gradient_alignment.detach().float().reshape(-1)
        if alignment.shape != gap_b.shape:
            raise ValueError(
                f"Gradient alignment must match teacher gaps: {alignment.shape} vs {gap_b.shape}."
            )
        if not torch.isfinite(alignment[valid]).all():
            raise ValueError("Gradient alignment must be finite on valid positions.")
        if (alignment[valid].abs() > 1.0 + 1e-5).any():
            raise ValueError("Gradient alignment must lie in [-1, 1].")
        alignment = alignment.clamp(-1.0, 1.0)
        positive_alignment = torch.zeros_like(alignment)
        positive_alignment[valid] = alignment[valid].clamp(0.0, 1.0)

        tau = torch.quantile(gap_b[valid], 0.5)
        rescue = torch.zeros_like(gap_b)
        rescue[valid] = (
            (gap_b[valid] - gap_b_plus[valid]) / (gap_b[valid] + float(eps))
        ).clamp(0.0, 1.0)
        confidence = torch.zeros_like(gap_b)
        confidence[valid] = gap_b[valid] / (gap_b[valid] + tau + float(eps))
        bridge_fraction = torch.zeros_like(gap_b)
        bridge_fraction[valid] = (
            float(max_bridge_fraction)
            * rescue[valid]
            * confidence[valid]
            * positive_alignment[valid]
        )

    return BudgetGradientAlignedBridgeGate(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        gradient_alignment=alignment.detach(),
        positive_alignment=positive_alignment.detach(),
        tau_teacher_gap=tau.detach(),
        rescue_fraction=rescue.detach(),
        confidence=confidence.detach(),
        bridge_fraction=bridge_fraction.detach(),
        valid_mask=valid.detach(),
    )


def counterfactual_gradient_residual_gate(
    teacher_gap_b: torch.Tensor,
    teacher_gap_b_plus: torch.Tensor,
    gradient_alignment: torch.Tensor,
    projection_coefficient: torch.Tensor,
    teacher_gradient_norm_sq: torch.Tensor,
    budget_gradient_norm_sq: torch.Tensor,
    gradient_dot_product: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    cancellation_strength: float = 0.5,
    max_projection_coefficient: float = 1.0,
    eps: float = 1e-8,
) -> CounterfactualGradientResidualGate:
    """Remove a bounded portion of the teacher gradient explained by ``b_plus``.

    A token is eligible only when the native adjacent-budget intervention
    lowers its full-teacher KL. The cancellation coefficient never exceeds
    the positive orthogonal-projection coefficient, so the residual cannot
    reverse the teacher gradient along the budget direction. All values are
    detached; this gate is used with a stop-gradient scalar correction that
    preserves the vanilla OPSD forward loss exactly.
    """

    if not 0.0 <= float(cancellation_strength) <= 1.0:
        raise ValueError(
            f"cancellation_strength must be in [0, 1]; got {cancellation_strength}."
        )
    if float(max_projection_coefficient) <= 0.0:
        raise ValueError(
            "max_projection_coefficient must be positive; "
            f"got {max_projection_coefficient}."
        )
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    with torch.no_grad():
        gap_b, gap_b_plus, valid = _validate_gap_inputs(
            teacher_gap_b, teacher_gap_b_plus, valid_mask
        )
        alignment = gradient_alignment.detach().float().reshape(-1)
        raw_projection = projection_coefficient.detach().float().reshape(-1)
        teacher_norm_sq = teacher_gradient_norm_sq.detach().float().reshape(-1)
        budget_norm_sq = budget_gradient_norm_sq.detach().float().reshape(-1)
        dot_product = gradient_dot_product.detach().float().reshape(-1)
        if not (
            gap_b.shape
            == alignment.shape
            == raw_projection.shape
            == teacher_norm_sq.shape
            == budget_norm_sq.shape
            == dot_product.shape
        ):
            raise ValueError("Counterfactual gradient-geometry tensors must align by token.")
        for name, values in (
            ("gradient alignment", alignment),
            ("projection coefficient", raw_projection),
            ("teacher gradient norm", teacher_norm_sq),
            ("budget gradient norm", budget_norm_sq),
            ("gradient dot product", dot_product),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (teacher_norm_sq[valid] < -1e-7).any() or (budget_norm_sq[valid] < -1e-7).any():
            raise ValueError("Squared gradient norms must be nonnegative.")

        alignment = alignment.clamp(-1.0, 1.0)
        clipped_projection = torch.zeros_like(raw_projection)
        clipped_projection[valid] = raw_projection[valid].clamp(
            0.0, float(max_projection_coefficient)
        )
        rescue_indicator = torch.zeros_like(gap_b)
        rescue_indicator[valid] = (gap_b[valid] > gap_b_plus[valid]).float()
        coefficient = torch.zeros_like(gap_b)
        coefficient[valid] = (
            float(cancellation_strength)
            * clipped_projection[valid]
            * rescue_indicator[valid]
        )

        residual_norm_sq = (
            teacher_norm_sq
            - 2.0 * coefficient * dot_product
            + coefficient.square() * budget_norm_sq
        ).clamp_min(0.0)
        residual_ratio = torch.ones_like(gap_b)
        nonzero_teacher = valid & (teacher_norm_sq > float(eps))
        residual_ratio[nonzero_teacher] = (
            residual_norm_sq[nonzero_teacher]
            / teacher_norm_sq[nonzero_teacher].clamp_min(float(eps))
        ).sqrt()

        # Since coefficient <= the positive projection coefficient, the
        # residual must not have a larger component along g_budget.
        residual_dot = dot_product - coefficient * budget_norm_sq
        eligible = valid & (coefficient > 0.0)
        if eligible.any() and (residual_dot[eligible] < -2e-6).any():
            raise FloatingPointError("Gradient cancellation reversed the budget-direction component.")

    return CounterfactualGradientResidualGate(
        teacher_gap_b=gap_b.detach(),
        teacher_gap_b_plus=gap_b_plus.detach(),
        gradient_alignment=alignment.detach(),
        raw_projection_coefficient=raw_projection.detach(),
        clipped_projection_coefficient=clipped_projection.detach(),
        budget_rescue_indicator=rescue_indicator.detach(),
        cancellation_coefficient=coefficient.detach(),
        residual_gradient_norm_ratio=residual_ratio.detach(),
        valid_mask=valid.detach(),
    )


def budget_tangent_residual_weights(
    teacher_gap_b: torch.Tensor,
    gradient_alignment: torch.Tensor,
    budget_explained_fraction: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> BudgetTangentResidualWeights:
    """Prioritize hard teacher residuals not explained by a +delta budget step.

    The adjacent native VisionZip distribution defines a finite-difference
    tangent in vocabulary-probability space. A token receives high priority
    only when its deployed-budget teacher KL is high and that tangent explains
    little of the full-teacher gradient. Detached KL-mass normalization keeps
    the scalar objective exactly equal to vanilla OPSD for each sample.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap = teacher_gap_b.detach().float().reshape(-1).clamp_min(0.0)
        alignment = gradient_alignment.detach().float().reshape(-1)
        explained = budget_explained_fraction.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap.device, dtype=torch.bool).reshape(-1)
        if gap.shape != alignment.shape or gap.shape != explained.shape or gap.shape != valid.shape:
            raise ValueError(
                "Teacher gap, gradient geometry, and valid mask must align: "
                f"{gap.shape}, {alignment.shape}, {explained.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        if not torch.isfinite(gap[valid]).all() or not torch.isfinite(alignment[valid]).all():
            raise ValueError("Teacher gap and gradient alignment must be finite.")
        if not torch.isfinite(explained[valid]).all():
            raise ValueError("Budget-explained fraction must be finite.")
        if (alignment[valid].abs() > 1.0 + 1e-5).any():
            raise ValueError("Gradient alignment must lie in [-1, 1].")
        if ((explained[valid] < -1e-6) | (explained[valid] > 1.0 + 1e-6)).any():
            raise ValueError("Budget-explained fraction must lie in [0, 1].")

        alignment = alignment.clamp(-1.0, 1.0)
        explained = explained.clamp(0.0, 1.0)
        residual = torch.zeros_like(gap)
        residual[valid] = 1.0 - explained[valid]
        gap_rank = torch.zeros_like(gap)
        gap_rank[valid] = _ordinal_percentile_rank(gap[valid])
        priority = torch.zeros_like(gap)
        priority[valid] = gap_rank[valid] * residual[valid]
        raw_weight = torch.zeros_like(gap)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]

        unweighted_mass = gap[valid].sum()
        weighted_mass = (raw_weight[valid] * gap[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap.device, dtype=torch.float32)
            weight = torch.zeros_like(gap)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(weighted_mass) or float(weighted_mass) <= 0.0:
                raise FloatingPointError(f"Invalid tangent-residual weighted KL mass: {float(weighted_mass)}")
            loss_mass_scale = unweighted_mass / weighted_mass
            weight = torch.zeros_like(gap)
            weight[valid] = raw_weight[valid] * loss_mass_scale
        if not torch.allclose(
            (weight[valid] * gap[valid]).sum(),
            unweighted_mass,
            rtol=2e-6,
            atol=2e-7,
        ):
            raise FloatingPointError("Budget-tangent residual KL-mass normalization failed.")

    return BudgetTangentResidualWeights(
        teacher_gap_b=gap.detach(),
        gradient_alignment=alignment.detach(),
        budget_explained_fraction=explained.detach(),
        budget_residual_fraction=residual.detach(),
        teacher_gap_rank=gap_rank.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_gradient_consensus_weights(
    teacher_gap_b: torch.Tensor,
    gradient_consensus: torch.Tensor,
    gradient_norm_consistency: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> BudgetGradientConsensusWeights:
    """Prioritize hard teacher corrections invariant to a native budget probe."""

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap = teacher_gap_b.detach().float().reshape(-1).clamp_min(0.0)
        consensus = gradient_consensus.detach().float().reshape(-1)
        norm_consistency = gradient_norm_consistency.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap.device, dtype=torch.bool).reshape(-1)
        if not (gap.shape == consensus.shape == norm_consistency.shape == valid.shape):
            raise ValueError("Gradient-consensus weighting tensors must align by token.")
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        for name, values in (
            ("teacher gap", gap),
            ("gradient consensus", consensus),
            ("gradient norm consistency", norm_consistency),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (consensus[valid].abs() > 1.0 + 1e-5).any():
            raise ValueError("Gradient consensus must lie in [-1, 1].")
        if ((norm_consistency[valid] < -1e-6) | (norm_consistency[valid] > 1.0 + 1e-6)).any():
            raise ValueError("Gradient norm consistency must lie in [0, 1].")

        consensus = consensus.clamp(-1.0, 1.0)
        norm_consistency = norm_consistency.clamp(0.0, 1.0)
        gap_rank = torch.zeros_like(gap)
        gap_rank[valid] = _ordinal_percentile_rank(gap[valid])
        priority = torch.zeros_like(gap)
        priority[valid] = (
            gap_rank[valid]
            * consensus[valid].clamp(0.0, 1.0)
            * norm_consistency[valid]
        )
        raw_weight = torch.zeros_like(gap)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]
        normalized = normalize_candidate_loss_mass(
            gap,
            gap,
            raw_weight,
            valid,
            eps=eps,
        )
    return BudgetGradientConsensusWeights(
        teacher_gap_b=gap.detach(),
        teacher_gap_rank=gap_rank.detach(),
        gradient_consensus=consensus.detach(),
        gradient_norm_consistency=norm_consistency.detach(),
        invariant_priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=normalized.loss_mass_scale.detach(),
        weight=normalized.weight.detach(),
        valid_mask=valid.detach(),
    )


def budget_counterfactual_teachability_weights(
    teacher_gap_b: torch.Tensor,
    gradient_alignment: torch.Tensor,
    budget_explained_fraction: torch.Tensor,
    teacher_support_coverage: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> BudgetCounterfactualTeachabilityWeights:
    """Weight hard, locally teachable gaps unexplained by a +delta budget.

    Local support coverage follows the compatibility diagnostic of
    Teachability-Aware OPD. The additional adjacent-budget residual is a
    pruning-specific counterfactual: it suppresses teacher corrections that
    are already explained by adding visual evidence while prioritizing
    support-compatible utilization gaps. All signals and weights are detached,
    and the per-sample KL mass matches vanilla OPSD exactly.
    """

    if not 0.0 <= float(alpha) <= 4.0:
        raise ValueError(f"alpha must be in [0, 4]; got {alpha}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    with torch.no_grad():
        gap = teacher_gap_b.detach().float().reshape(-1).clamp_min(0.0)
        alignment = gradient_alignment.detach().float().reshape(-1)
        explained = budget_explained_fraction.detach().float().reshape(-1)
        coverage = teacher_support_coverage.detach().float().reshape(-1)
        valid = valid_mask.detach().to(device=gap.device, dtype=torch.bool).reshape(-1)
        if not (gap.shape == alignment.shape == explained.shape == coverage.shape == valid.shape):
            raise ValueError(
                "Counterfactual teachability inputs must align: "
                f"{gap.shape}, {alignment.shape}, {explained.shape}, {coverage.shape}, {valid.shape}."
            )
        if not valid.any():
            raise ValueError("At least one valid generated-token position is required.")
        for name, values in (
            ("teacher gap", gap),
            ("gradient alignment", alignment),
            ("budget explained fraction", explained),
            ("teacher support coverage", coverage),
        ):
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name} must be finite on valid positions.")
        if (alignment[valid].abs() > 1.0 + 1e-5).any():
            raise ValueError("Gradient alignment must lie in [-1, 1].")
        if ((explained[valid] < -1e-6) | (explained[valid] > 1.0 + 1e-6)).any():
            raise ValueError("Budget-explained fraction must lie in [0, 1].")
        if ((coverage[valid] < -1e-6) | (coverage[valid] > 1.0 + 1e-6)).any():
            raise ValueError("Teacher support coverage must lie in [0, 1].")

        alignment = alignment.clamp(-1.0, 1.0)
        explained = explained.clamp(0.0, 1.0)
        coverage = coverage.clamp(0.0, 1.0)
        residual = torch.zeros_like(gap)
        residual[valid] = 1.0 - explained[valid]
        gap_rank = torch.zeros_like(gap)
        gap_rank[valid] = _ordinal_percentile_rank(gap[valid])
        coverage_rank = torch.zeros_like(gap)
        coverage_rank[valid] = _ordinal_percentile_rank(coverage[valid])
        priority = torch.zeros_like(gap)
        priority[valid] = gap_rank[valid] * coverage_rank[valid] * residual[valid]
        raw_weight = torch.zeros_like(gap)
        raw_weight[valid] = 1.0 + float(alpha) * priority[valid]

        unweighted_mass = gap[valid].sum()
        weighted_mass = (raw_weight[valid] * gap[valid]).sum()
        if float(unweighted_mass) <= float(eps):
            loss_mass_scale = torch.ones((), device=gap.device, dtype=torch.float32)
            weight = torch.zeros_like(gap)
            weight[valid] = 1.0
        else:
            if not torch.isfinite(weighted_mass) or float(weighted_mass) <= 0.0:
                raise FloatingPointError(
                    f"Invalid counterfactual-teachability weighted KL mass: {float(weighted_mass)}"
                )
            loss_mass_scale = unweighted_mass / weighted_mass
            weight = torch.zeros_like(gap)
            weight[valid] = raw_weight[valid] * loss_mass_scale
        if not torch.allclose(
            (weight[valid] * gap[valid]).sum(),
            unweighted_mass,
            rtol=2e-6,
            atol=2e-7,
        ):
            raise FloatingPointError("Counterfactual-teachability KL-mass normalization failed.")

    return BudgetCounterfactualTeachabilityWeights(
        teacher_gap_b=gap.detach(),
        gradient_alignment=alignment.detach(),
        budget_explained_fraction=explained.detach(),
        budget_residual_fraction=residual.detach(),
        teacher_support_coverage=coverage.detach(),
        teacher_gap_rank=gap_rank.detach(),
        support_coverage_rank=coverage_rank.detach(),
        priority=priority.detach(),
        raw_weight=raw_weight.detach(),
        loss_mass_scale=loss_mass_scale.detach(),
        weight=weight.detach(),
        valid_mask=valid.detach(),
    )
