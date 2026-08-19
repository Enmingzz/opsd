from __future__ import annotations

import pytest
import torch

from opsd.visionzip_aokvqa import train as train_module
from opsd.visionzip_aokvqa.train import (
    apply_distributed_trajectory_weighting,
    validate_paired_native_budget_config,
)
from opsd.visionzip_aokvqa.trajectory_weighting import (
    AdaptiveBudgetFrontierState,
    ProgressAdaptiveFrontierState,
    RobustnessGatedCurriculumState,
    SensitivityFrontierState,
    average_rank_priority,
    competence_frontier_probability_weights,
    direct_inverse_sensitivity_probability_weights,
    effective_batch_local_objective,
    globally_calibrated_trajectory_weights,
    inverse_sensitivity_probability_weights,
    ratio_group_angle_probability_weights,
    ratio_group_angle_sample_probability_weights,
    ratio_group_fraction_probability_weights,
    ratio_group_projection_probability_weights,
    ratio_group_signal_probability_weights,
    residualized_budget_sensitivity,
    robustness_gated_curriculum_weights,
    sensitivity_frontier_weights,
    softmax_inverse_sensitivity_probability_weights,
    teacher_gap_mass_robustness,
    trajectory_rank_downweights,
    trajectory_sigmoid_downweights,
    uniform_trajectory_probability_weights,
)


def _calibrate_adaptive_budget_state(
    state: AdaptiveBudgetFrontierState,
) -> None:
    ratios = [state.select_ratio(index, f"sample-{index}") for index in range(256)]
    values = [
        {0.1: 0.040, 0.2: 0.020, 0.3: 0.010, 0.4: 0.006}[ratio]
        for ratio in ratios
    ]
    for start in range(0, 256, 32):
        state.update(ratios[start : start + 32], values[start : start + 32])


def _paired_native_budget_cfg() -> dict:
    return {
        "training": {"lora_dropout": 0.0, "max_sample_retries": 0},
        "pruning": {
            "retention_ratio_schedule": "paired_deterministic_uniform",
            "train_retention_ratios": [0.1, 0.2, 0.3, 0.4],
        },
        "paired_sampling": {"enabled": True},
        "opsd": {
            "native_budget_weighting": {
                "enabled": True,
                "mode": "trajectory_probe",
                "budget_delta_mode": "relative",
                "budget_delta_fraction": 0.25,
                "sensitivity_temperature": 1.0,
            },
            "trajectory_weighting": {
                "enabled": True,
                "mode": "ratio_group_counterfactual_teachability_batch",
                "group_statistic": "teacher_directed_projection_cosine",
                "group_transform": "softmax",
                "temperature": 0.15,
                "normalization": "probability_sum_one",
                "normalization_scope": "effective_batch",
            },
        },
    }


@pytest.mark.parametrize("pruning_method", ["visionzip", "random"])
def test_paired_native_budget_validation_accepts_supported_backends(
    pruning_method: str,
) -> None:
    validate_paired_native_budget_config(
        _paired_native_budget_cfg(),
        method="opsd_nogt",
        parameter_scope="language_decoder_only",
        pruning_method=pruning_method,
    )


def test_paired_native_budget_validation_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="VisionZip or RandomPruner"):
        validate_paired_native_budget_config(
            _paired_native_budget_cfg(),
            method="opsd_nogt",
            parameter_scope="language_decoder_only",
            pruning_method="divprune",
        )


def test_global_f_validation_accepts_absolute_seven_point_five_delta() -> None:
    cfg = _paired_native_budget_cfg()
    cfg["pruning"]["train_retention_ratios"] = [0.1]
    cfg["paired_sampling"]["allow_custom_retention_ratios"] = True
    cfg["opsd"]["native_budget_weighting"].update(
        {"budget_delta_mode": "absolute", "budget_delta": 0.075}
    )
    cfg["opsd"]["trajectory_weighting"] = {
        "enabled": True,
        "mode": "global_calibrated_counterfactual_teachability_batch",
        "calibration": {"q05": 0.1, "q95": 0.8, "normalized_mean": 0.45},
        "coefficient": 1.0,
        "normalization": "fixed_global_centered_scale",
        "normalization_scope": "frozen_training_calibration",
    }
    validate_paired_native_budget_config(
        cfg,
        method="opsd_nogt",
        parameter_scope="language_decoder_only",
        pruning_method="visionzip",
    )


def test_adaptive_budget_warmup_is_exactly_balanced_and_deterministic() -> None:
    left = AdaptiveBudgetFrontierState(seed=123)
    right = AdaptiveBudgetFrontierState(seed=123)
    left_ratios = [left.select_ratio(index, f"sample-{index}") for index in range(256)]
    right_ratios = [right.select_ratio(index, f"sample-{index}") for index in range(256)]
    assert left_ratios == right_ratios
    assert {ratio: left_ratios.count(ratio) for ratio in left.retention_ratios} == {
        0.1: 64,
        0.2: 64,
        0.3: 64,
        0.4: 64,
    }
    for start in range(0, 256, 4):
        assert set(left_ratios[start : start + 4]) == set(left.retention_ratios)


def test_adaptive_budget_state_calibrates_updates_and_preserves_support() -> None:
    state = AdaptiveBudgetFrontierState(seed=9)
    _calibrate_adaptive_budget_state(state)
    assert state.ready
    assert state.calibration_complete_at_trajectory == 256
    initial = state.probabilities()
    assert sum(initial) == pytest.approx(1.0)
    assert min(initial) > 0.0

    # Simulate high ratios becoming solved faster than low ratios.
    for _ in range(20):
        state.update(
            [0.1] * 8 + [0.2] * 8 + [0.3] * 8 + [0.4] * 8,
            [0.030] * 8 + [0.015] * 8 + [0.004] * 8 + [0.001] * 8,
        )
    final = state.probabilities()
    assert sum(final) == pytest.approx(1.0)
    assert min(final) > 0.0
    assert (final[0] + final[1]) > (initial[0] + initial[1])


def test_adaptive_budget_state_resume_reproduces_future_sequence() -> None:
    state = AdaptiveBudgetFrontierState(seed=77, namespace="resume-test")
    _calibrate_adaptive_budget_state(state)
    state.update([0.1, 0.2, 0.3, 0.4], [0.03, 0.02, 0.008, 0.003])
    restored = AdaptiveBudgetFrontierState(seed=77, namespace="resume-test")
    restored.load_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()
    expected = [state.select_ratio(index, f"id-{index}") for index in range(256, 512)]
    actual = [restored.select_ratio(index, f"id-{index}") for index in range(256, 512)]
    assert actual == expected


def test_uniform_probability_weights_exactly_recover_vanilla_mean() -> None:
    losses = torch.tensor([0.01, 0.03, 0.02, 0.08], requires_grad=True)
    weights = uniform_trajectory_probability_weights(len(losses))
    weighted = torch.dot(weights.probability_weight, losses)
    assert weighted.item() == pytest.approx(losses.mean().item(), abs=1e-8)
    assert not weights.probability_weight.requires_grad
    weighted.backward()
    assert torch.allclose(losses.grad, torch.full_like(losses, 0.25))


def test_global_calibration_constant_center_recovers_vanilla_mean() -> None:
    signal = torch.full((4,), 0.5, requires_grad=True)
    result = globally_calibrated_trajectory_weights(
        signal,
        q05=0.0,
        q95=1.0,
        calibration_mean=0.5,
        coefficient=1.0,
    )
    assert torch.allclose(result.objective_weight, torch.ones(4))
    assert torch.allclose(result.probability_weight, torch.full((4,), 0.25))
    assert not result.objective_weight.requires_grad
    assert not result.probability_weight.requires_grad


def test_global_calibration_does_not_renormalize_current_batch() -> None:
    low_batch = globally_calibrated_trajectory_weights(
        torch.tensor([0.0, 0.25]),
        q05=0.0,
        q95=1.0,
        calibration_mean=0.5,
        coefficient=1.0,
    )
    high_batch = globally_calibrated_trajectory_weights(
        torch.tensor([0.75, 1.0]),
        q05=0.0,
        q95=1.0,
        calibration_mean=0.5,
        coefficient=1.0,
    )
    assert float(low_batch.objective_weight.mean()) == pytest.approx(0.625)
    assert float(high_batch.objective_weight.mean()) == pytest.approx(1.375)
    assert float(low_batch.probability_weight.sum()) == pytest.approx(0.625)
    assert float(high_batch.probability_weight.sum()) == pytest.approx(1.375)


def test_global_calibration_executor_is_mean_weighted_loss() -> None:
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    result = globally_calibrated_trajectory_weights(
        torch.tensor([0.0, 0.25, 0.75, 1.0]),
        q05=0.0,
        q95=1.0,
        calibration_mean=0.5,
        coefficient=1.0,
    )
    local = [
        effective_batch_local_objective(
            loss,
            probability,
            effective_batch_size=losses.numel(),
        )
        for loss, probability in zip(losses, result.probability_weight)
    ]
    accumulated = torch.stack(local).sum() / losses.numel()
    expected = (result.objective_weight * losses).mean()
    assert torch.allclose(accumulated, expected)
    accumulated.backward()
    assert torch.allclose(losses.grad, result.objective_weight / losses.numel())


def test_global_calibration_rejects_nonpositive_weight_floor() -> None:
    with pytest.raises(ValueError, match="must remain positive"):
        globally_calibrated_trajectory_weights(
            torch.tensor([0.0, 1.0]),
            q05=0.0,
            q95=1.0,
            calibration_mean=0.6,
            coefficient=2.0,
        )


def test_global_f_training_objective_uses_raw_f_and_fixed_scale() -> None:
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    metrics = [
        {
            "native_trajectory_budget_explained_fraction": 0.0,
            "native_trajectory_budget_projection_mass": 0.0,
            "native_trajectory_teacher_js_mass": 1.0,
            "native_student_budget_jsd_mean": 0.1,
            "sampled_b": 0.1,
        },
        {
            "native_trajectory_budget_explained_fraction": 0.25,
            "native_trajectory_budget_projection_mass": 0.5,
            "native_trajectory_teacher_js_mass": 2.0,
            "native_student_budget_jsd_mean": 0.1,
            "sampled_b": 0.1,
        },
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "global_calibrated_counterfactual_teachability_batch",
                "calibration": {"q05": 0.0, "q95": 1.0, "normalized_mean": 0.5},
                "coefficient": 1.0,
                "eps": 1e-8,
            }
        }
    }
    weighted, output = apply_distributed_trajectory_weighting(
        losses, metrics, cfg, distributed=False, rank=0, world_size=1
    )
    assert weighted.item() == pytest.approx((0.5 * 1.0 + 0.75 * 3.0) / 2.0)
    assert output["trajectory_probability_weight_sum"] == pytest.approx(0.625)
    assert output["trajectory_global_objective_weight_mean"] == pytest.approx(0.625)
    assert output["trajectory_global_batch_renormalized"] is False
    assert output["trajectory_normalization"] == "fixed_global_calibration_no_batch_renormalization"
    weighted.backward()
    assert losses[0].grad.item() == pytest.approx(0.25)
    assert losses[1].grad.item() == pytest.approx(0.375)


def test_adaptive_budget_signal_supports_robust_need_and_kl_control() -> None:
    metrics = {
        "native_student_budget_jsd_mean": 0.002,
        "native_teacher_gap_b_mean": 0.04,
        "native_trajectory_budget_explained_fraction": 0.25,
    }
    robust_cfg = {
        "opsd": {"trajectory_weighting": {"sampler_metric": "robust_need"}}
    }
    kl_cfg = {"opsd": {"trajectory_weighting": {"sampler_metric": "teacher_kl"}}}
    assert train_module.trajectory_sensitivity_signal(
        metrics, robust_cfg, "adaptive_budget_frontier_sampler_batch"
    ) == pytest.approx(0.03)
    assert train_module.trajectory_sensitivity_signal(
        metrics, kl_cfg, "adaptive_budget_frontier_sampler_batch"
    ) == pytest.approx(0.04)


def test_adaptive_budget_online_update_keeps_vanilla_loss_and_updates_state() -> None:
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "adaptive_budget_frontier_sampler_batch",
                "sampler_metric": "robust_need",
                "eps": 1e-8,
            }
        }
    }
    state = AdaptiveBudgetFrontierState(
        calibration_target_per_ratio=1,
        ema_half_life_per_ratio=4,
        seed=42,
    )
    losses = [
        torch.tensor(value, requires_grad=True)
        for value in (0.04, 0.02, 0.01, 0.006)
    ]
    metrics = [
        {
            "native_teacher_gap_b_mean": gap,
            "native_trajectory_budget_explained_fraction": explained,
            "sampled_b": ratio,
        }
        for ratio, gap, explained in (
            (0.1, 0.04, 0.10),
            (0.2, 0.02, 0.20),
            (0.3, 0.01, 0.40),
            (0.4, 0.006, 0.70),
        )
    ]
    weighted, summary = apply_distributed_trajectory_weighting(
        losses,
        metrics,
        cfg,
        distributed=False,
        rank=0,
        world_size=1,
        curriculum_state=state,
    )
    vanilla = torch.stack(losses).mean()
    assert weighted.item() == pytest.approx(vanilla.item(), abs=1e-8)
    assert state.ready
    assert summary["budget_sampler_ready_before"] is False
    assert summary["budget_sampler_ready_after"] is True
    assert summary["budget_sampler_loss_is_vanilla_mean"] is True
    assert summary["budget_sampler_state_update_scope"] == "synchronized_microbatch"
    assert summary["trajectory_normalization"] == (
        "vanilla_mean_after_adaptive_ratio_sampling"
    )
    assert summary["trajectory_loss_scale_ratio"] == pytest.approx(1.0)
    weighted.backward()
    assert all(item.grad.item() == pytest.approx(0.25) for item in losses)


def test_competence_frontier_is_bounded_symmetric_and_peaks_at_tau() -> None:
    losses = torch.ones(3)
    result = competence_frontier_probability_weights(
        losses,
        torch.tensor([0.25, 1.0, 4.0]),
        tau=1.0,
    )
    assert torch.all((result.raw_weight >= 0.0) & (result.raw_weight <= 1.0))
    assert result.raw_weight[1].item() == pytest.approx(1.0)
    assert result.raw_weight[0].item() == pytest.approx(result.raw_weight[2].item())


def test_competence_frontier_preserves_scalar_kl_and_detaches_signal() -> None:
    losses = torch.tensor([0.01, 0.03, 0.02, 0.08], requires_grad=True)
    signal = torch.tensor([0.002, 0.008, 0.016, 0.050], requires_grad=True)
    result = competence_frontier_probability_weights(losses, signal, tau=0.008)
    weighted = torch.dot(result.probability_weight, losses)
    assert weighted.item() == pytest.approx(losses.detach().mean().item(), abs=2e-7)
    assert result.weighted_loss.item() == pytest.approx(
        result.unweighted_loss.item(), abs=2e-7
    )
    assert not result.probability_weight.requires_grad
    weighted.backward()
    assert signal.grad is None
    assert losses.grad is not None


def test_competence_frontier_constant_tau_signal_is_exact_vanilla() -> None:
    losses = torch.tensor([0.01, 0.03, 0.02, 0.08])
    result = competence_frontier_probability_weights(
        losses, torch.full_like(losses, 0.008), tau=0.008
    )
    assert torch.equal(result.objective_weight, torch.ones_like(losses))
    assert torch.equal(
        result.probability_weight, torch.full_like(losses, 1.0 / losses.numel())
    )


def test_progress_adaptive_frontier_state_updates_and_roundtrips() -> None:
    state = ProgressAdaptiveFrontierState(
        initial_tau=0.008,
        initial_robust_need_mean=0.020,
        ema_robust_need_mean=0.020,
        ema_half_life_trajectories=32.0,
    )
    assert state.tau == pytest.approx(0.008)
    first = state.update(0.010, 32)
    assert first["tau_after"] > first["tau_before"]
    assert state.tau == pytest.approx(0.013)

    restored = ProgressAdaptiveFrontierState(
        initial_tau=0.008,
        initial_robust_need_mean=0.020,
        ema_robust_need_mean=0.020,
        ema_half_life_trajectories=32.0,
    )
    restored.load_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()


def test_progress_adaptive_frontier_never_moves_below_initial_tau() -> None:
    state = ProgressAdaptiveFrontierState(
        initial_tau=0.008,
        initial_robust_need_mean=0.020,
        ema_robust_need_mean=0.020,
        ema_half_life_trajectories=32.0,
    )
    state.update(0.040, 32)
    assert state.tau == pytest.approx(0.008)


def test_progress_adaptive_signal_is_teacher_gap_times_unexplained_fraction() -> None:
    metrics = {
        "native_student_budget_jsd_mean": 0.01,
        "native_teacher_gap_b_mean": 0.04,
        "native_trajectory_budget_explained_fraction": 0.25,
    }
    signal = train_module.trajectory_sensitivity_signal(
        metrics,
        {"opsd": {"trajectory_weighting": {"eps": 1e-8}}},
        "progress_adaptive_robust_frontier_batch",
    )
    assert signal == pytest.approx(0.03)


def test_progress_adaptive_frontier_training_helper_preserves_batch_kl() -> None:
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "progress_adaptive_robust_frontier_batch",
                "eps": 1e-8,
            }
        }
    }
    state = ProgressAdaptiveFrontierState(
        initial_tau=0.008,
        initial_robust_need_mean=0.020,
        ema_robust_need_mean=0.020,
        ema_half_life_trajectories=32.0,
    )
    losses = [
        torch.tensor(value, requires_grad=True)
        for value in (0.01, 0.03, 0.02, 0.08)
    ]
    metrics = [
        {
            "native_teacher_gap_b_mean": gap,
            "native_trajectory_budget_explained_fraction": explained,
        }
        for gap, explained in (
            (0.01, 0.2),
            (0.03, 0.5),
            (0.02, 0.8),
            (0.08, 0.1),
        )
    ]
    weighted, summary = apply_distributed_trajectory_weighting(
        losses,
        metrics,
        cfg,
        distributed=False,
        rank=0,
        world_size=1,
        curriculum_state=state,
    )
    vanilla = torch.stack(losses).mean()
    assert weighted.detach().item() == pytest.approx(vanilla.detach().item(), abs=2e-7)
    assert summary["trajectory_global_scalar_error"] == pytest.approx(0.0, abs=2e-7)
    assert summary["trajectory_normalization"] == "synchronized_kl_mass_preserving"
    assert summary["trajectory_weight_detached"] is True
    weighted.backward()
    assert all(loss.grad is not None for loss in losses)


def test_inverse_sensitivity_probability_weights_sum_to_one_and_prefer_robust() -> None:
    sensitivity = torch.tensor([0.05, 0.20, 0.80], requires_grad=True)
    result = inverse_sensitivity_probability_weights(sensitivity)
    assert result.tau.item() == pytest.approx(0.20)
    assert result.probability_weight.sum().item() == pytest.approx(1.0)
    assert result.probability_weight[0] > result.probability_weight[1]
    assert result.probability_weight[1] > result.probability_weight[2]
    assert not result.probability_weight.requires_grad
    assert sensitivity.grad is None


def test_inverse_sensitivity_probability_weights_are_uniform_at_zero() -> None:
    result = inverse_sensitivity_probability_weights(torch.zeros(4))
    assert torch.allclose(result.probability_weight, torch.full((4,), 0.25))
    assert result.probability_weight.sum().item() == pytest.approx(1.0)


def test_direct_inverse_weights_have_no_tau_and_are_probability_normalized() -> None:
    sensitivity = torch.tensor([0.05, 0.10, 0.20], requires_grad=True)
    result = direct_inverse_sensitivity_probability_weights(sensitivity)
    expected_raw = sensitivity.detach().reciprocal()
    expected = expected_raw / expected_raw.sum()
    assert not hasattr(result, "tau")
    assert torch.allclose(result.probability_weight, expected)
    assert result.probability_weight.sum().item() == pytest.approx(1.0)
    assert result.probability_weight[0] > result.probability_weight[1]
    assert result.probability_weight[1] > result.probability_weight[2]
    assert not result.probability_weight.requires_grad
    assert sensitivity.grad is None


def test_direct_inverse_uniform_signal_exactly_recovers_mean_loss() -> None:
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights = direct_inverse_sensitivity_probability_weights(
        torch.full((4,), 0.1)
    ).probability_weight
    assert torch.allclose(weights, torch.full((4,), 0.25))
    assert torch.dot(weights, losses).item() == pytest.approx(losses.mean().item())


@pytest.mark.parametrize("temperature", [0.05, 0.1])
def test_softmax_inverse_sensitivity_weights(temperature: float) -> None:
    sensitivity = torch.tensor([0.05, 0.10, 0.20], requires_grad=True)
    result = softmax_inverse_sensitivity_probability_weights(
        sensitivity, temperature=temperature
    )
    expected = torch.softmax(-sensitivity.detach() / temperature, dim=0)
    assert torch.allclose(result.probability_weight, expected)
    assert result.probability_weight.sum().item() == pytest.approx(1.0)
    assert result.probability_weight[0] > result.probability_weight[1]
    assert result.probability_weight[1] > result.probability_weight[2]
    assert result.temperature.item() == pytest.approx(temperature)
    assert not result.probability_weight.requires_grad
    assert sensitivity.grad is None


def test_softmax_uniform_signal_exactly_recovers_mean_loss() -> None:
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights = softmax_inverse_sensitivity_probability_weights(
        torch.full((4,), 0.1), temperature=0.05
    ).probability_weight
    assert torch.allclose(weights, torch.full((4,), 0.25))
    assert torch.dot(weights, losses).item() == pytest.approx(losses.mean().item())


@pytest.mark.parametrize("temperature", [0.05, 0.1])
def test_single_process_jsd_current_kl_softmax_weighting(temperature: float) -> None:
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    metrics = [
        {"native_student_budget_jsd_mean": 0.01, "native_teacher_gap_b_mean": 0.10},
        {"native_student_budget_jsd_mean": 0.08, "native_teacher_gap_b_mean": 0.10},
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "jsd_over_current_kl_softmax_batch",
                "temperature": temperature,
                "eps": 1e-8,
            }
        }
    }
    weighted, output = apply_distributed_trajectory_weighting(
        losses, metrics, cfg, distributed=False, rank=0, world_size=1
    )
    expected_weights = softmax_inverse_sensitivity_probability_weights(
        torch.tensor([0.1, 0.8]), temperature=temperature
    )
    expected = (expected_weights.probability_weight * torch.tensor([1.0, 3.0])).sum()
    assert weighted.item() == pytest.approx(expected.item())
    assert output["trajectory_probability_weight_sum"] == pytest.approx(1.0)
    assert output["trajectory_batch_tau"] is None
    assert output["trajectory_weight_transform"] == "softmax_negative_sensitivity"
    assert output["trajectory_weight_temperature"] == pytest.approx(temperature)
    weighted.backward()
    assert losses[0].grad.item() > losses[1].grad.item()


def test_single_process_jsd_current_kl_direct_inverse_weighting() -> None:
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    metrics = [
        {"native_student_budget_jsd_mean": 0.01, "native_teacher_gap_b_mean": 0.10},
        {"native_student_budget_jsd_mean": 0.08, "native_teacher_gap_b_mean": 0.10},
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "jsd_over_current_kl_direct_inverse_batch",
                "eps": 1e-8,
            }
        }
    }
    weighted, output = apply_distributed_trajectory_weighting(
        losses,
        metrics,
        cfg,
        distributed=False,
        rank=0,
        world_size=1,
    )
    expected_weights = direct_inverse_sensitivity_probability_weights(torch.tensor([0.1, 0.8]))
    expected = (expected_weights.probability_weight * torch.tensor([1.0, 3.0])).sum()
    assert weighted.item() == pytest.approx(expected.item())
    assert output["trajectory_probability_weight_sum"] == pytest.approx(1.0)
    assert output["trajectory_batch_tau"] is None
    assert output["trajectory_weight_transform"] == "direct_inverse"
    weighted.backward()
    assert losses[0].grad.item() > losses[1].grad.item()


def test_ratio_group_signal_weights_discard_within_ratio_ranking() -> None:
    signal = torch.tensor([0.2, 0.6, 0.8, 0.4], requires_grad=True)
    ratios = torch.tensor([0.1, 0.1, 0.2, 0.2])
    result = ratio_group_signal_probability_weights(signal, ratios)
    assert torch.allclose(result.group_signal, torch.tensor([0.4, 0.4, 0.6, 0.6]))
    assert result.probability_weight.sum().item() == pytest.approx(1.0)
    assert result.probability_weight[0] == pytest.approx(result.probability_weight[1])
    assert result.probability_weight[2] == pytest.approx(result.probability_weight[3])
    assert result.probability_weight[2] > result.probability_weight[0]
    assert not result.probability_weight.requires_grad
    assert signal.grad is None


def test_ratio_group_fraction_weights_reduce_mass_before_dividing() -> None:
    numerator = torch.tensor([0.1, 0.3, 0.4, 0.2], requires_grad=True)
    denominator = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
    ratios = torch.tensor([0.1, 0.1, 0.2, 0.2])
    result = ratio_group_fraction_probability_weights(
        numerator, denominator, ratios
    )
    assert torch.allclose(result.group_signal, torch.tensor([0.4, 0.4, 0.6, 0.6]))
    assert result.probability_weight.sum().item() == pytest.approx(1.0)
    assert result.probability_weight[0] == pytest.approx(result.probability_weight[1])
    assert result.probability_weight[2] == pytest.approx(result.probability_weight[3])
    assert result.probability_weight[2] > result.probability_weight[0]
    assert numerator.grad is None
    assert denominator.grad is None


def test_ratio_group_fraction_softmax_weights_apply_positive_temperature() -> None:
    numerator = torch.tensor([0.10, 0.10, 0.40, 0.40], requires_grad=True)
    denominator = torch.ones(4, requires_grad=True)
    ratios = torch.tensor([0.1, 0.1, 0.2, 0.2])
    result = ratio_group_fraction_probability_weights(
        numerator,
        denominator,
        ratios,
        transform="softmax",
        temperature=0.1,
    )
    expected = torch.softmax(torch.tensor([0.1, 0.1, 0.4, 0.4]) / 0.1, dim=0)
    assert torch.allclose(result.group_signal, torch.tensor([0.1, 0.1, 0.4, 0.4]))
    assert torch.allclose(result.probability_weight, expected)
    assert result.probability_weight.sum().item() == pytest.approx(1.0)
    assert result.probability_weight[2] > result.probability_weight[0]
    assert not result.probability_weight.requires_grad
    assert numerator.grad is None
    assert denominator.grad is None


def test_ratio_group_angle_softmax_is_count_invariant_and_detached() -> None:
    projection = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
    teacher = torch.ones(4, requires_grad=True)
    budget = torch.ones(4, requires_grad=True)
    ratios = torch.tensor([0.1, 0.1, 0.1, 0.2])
    result = ratio_group_angle_probability_weights(
        projection,
        teacher,
        budget,
        ratios,
        temperature=0.15,
    )
    assert torch.allclose(result.group_signal, torch.full((4,), 0.5))
    assert result.probability_weight[:3].sum().item() == pytest.approx(0.5)
    assert result.probability_weight[3].item() == pytest.approx(0.5)
    assert torch.allclose(
        result.probability_weight[:3], torch.full((3,), 1.0 / 6.0)
    )
    assert result.probability_weight.sum().item() == pytest.approx(1.0)
    assert not result.probability_weight.requires_grad
    assert projection.grad is None
    assert teacher.grad is None
    assert budget.grad is None


def test_ratio_group_angle_sample_softmax_recovers_uniform_sample_mean() -> None:
    projection = torch.full((4,), 0.5, requires_grad=True)
    teacher = torch.ones(4, requires_grad=True)
    budget = torch.ones(4, requires_grad=True)
    ratios = torch.tensor([0.1, 0.1, 0.1, 0.2])
    result = ratio_group_angle_sample_probability_weights(
        projection,
        teacher,
        budget,
        ratios,
        temperature=0.15,
    )
    assert torch.allclose(result.group_signal, torch.full((4,), 0.5))
    assert torch.allclose(result.probability_weight, torch.full((4,), 0.25))
    assert result.probability_weight[:3].sum().item() == pytest.approx(0.75)
    assert result.probability_weight[3].item() == pytest.approx(0.25)
    assert not result.probability_weight.requires_grad
    assert projection.grad is None
    assert teacher.grad is None
    assert budget.grad is None


def test_ratio_group_angle_sample_objective_is_original_mean_for_equal_signal() -> None:
    losses = [torch.tensor(float(index + 1), requires_grad=True) for index in range(4)]
    metrics = [
        {
            "native_student_budget_jsd_mean": 1.0,
            "native_trajectory_budget_explained_fraction": 0.5,
            "native_trajectory_budget_projection_mass": 0.5,
            "native_trajectory_teacher_js_mass": 1.0,
            "sampled_b": ratio,
        }
        for ratio in (0.10, 0.10, 0.10, 0.20)
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "ratio_group_counterfactual_teachability_batch",
                "group_statistic": (
                    "teacher_directed_projection_cosine_sample_normalized"
                ),
                "group_transform": "softmax",
                "temperature": 0.15,
                "eps": 1e-8,
            }
        }
    }
    weighted, output = apply_distributed_trajectory_weighting(
        losses, metrics, cfg, distributed=False, rank=0, world_size=1
    )
    expected = torch.stack(losses).mean()
    assert weighted.item() == pytest.approx(expected.item())
    assert output["trajectory_probability_weight_sum"] == pytest.approx(1.0)
    assert output["trajectory_weight_transform"] == (
        "ratio_group_softmax_angle_sample_normalized"
    )
    weighted.backward()
    for loss in losses:
        assert loss.grad.item() == pytest.approx(0.25)


def test_ratio_group_projection_softmax_is_count_invariant_and_detached() -> None:
    projection = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
    teacher = torch.ones(4, requires_grad=True)
    ratios = torch.tensor([0.1, 0.1, 0.1, 0.2])
    result = ratio_group_projection_probability_weights(
        projection,
        teacher,
        ratios,
        temperature=0.15,
    )
    assert torch.allclose(result.group_signal, torch.full((4,), 0.5))
    assert result.probability_weight[:3].sum().item() == pytest.approx(0.5)
    assert result.probability_weight[3].item() == pytest.approx(0.5)
    assert torch.allclose(
        result.probability_weight[:3], torch.full((3,), 1.0 / 6.0)
    )
    assert result.probability_weight.sum().item() == pytest.approx(1.0)
    assert not result.probability_weight.requires_grad
    assert projection.grad is None
    assert teacher.grad is None


def test_ratio_group_angle_objective_is_exact_group_mean_softmax() -> None:
    losses = [torch.tensor(float(index + 1), requires_grad=True) for index in range(4)]
    metrics = [
        {
            "native_student_budget_jsd_mean": 1.0,
            "native_trajectory_budget_explained_fraction": 0.5,
            "native_trajectory_budget_projection_mass": 0.5,
            "native_trajectory_teacher_js_mass": 1.0,
            "sampled_b": 0.10,
        },
        {
            "native_student_budget_jsd_mean": 1.0,
            "native_trajectory_budget_explained_fraction": 0.5,
            "native_trajectory_budget_projection_mass": 0.5,
            "native_trajectory_teacher_js_mass": 1.0,
            "sampled_b": 0.10,
        },
        {
            "native_student_budget_jsd_mean": 1.0,
            "native_trajectory_budget_explained_fraction": 0.5,
            "native_trajectory_budget_projection_mass": 0.5,
            "native_trajectory_teacher_js_mass": 1.0,
            "sampled_b": 0.10,
        },
        {
            "native_student_budget_jsd_mean": 1.0,
            "native_trajectory_budget_explained_fraction": 0.5,
            "native_trajectory_budget_projection_mass": 0.5,
            "native_trajectory_teacher_js_mass": 1.0,
            "sampled_b": 0.20,
        },
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "ratio_group_counterfactual_teachability_batch",
                "group_statistic": "teacher_directed_projection_cosine",
                "group_transform": "softmax",
                "temperature": 0.15,
                "eps": 1e-8,
            }
        }
    }
    weighted, output = apply_distributed_trajectory_weighting(
        losses, metrics, cfg, distributed=False, rank=0, world_size=1
    )
    expected = 0.5 * torch.tensor([1.0, 2.0, 3.0]).mean() + 0.5 * 4.0
    assert weighted.item() == pytest.approx(expected.item())
    assert output["trajectory_probability_weight_sum"] == pytest.approx(1.0)
    assert output["trajectory_weight_transform"] == "ratio_group_softmax_angle"
    weighted.backward()
    assert losses[0].grad.item() == pytest.approx(1.0 / 6.0)
    assert losses[3].grad.item() == pytest.approx(0.5)


def test_ratio_group_teachability_softmax_objective_is_normalized() -> None:
    losses = [torch.tensor(float(index + 1), requires_grad=True) for index in range(4)]
    metrics = [
        {"native_trajectory_budget_explained_fraction": 0.1, "native_trajectory_budget_projection_mass": 0.1, "native_trajectory_teacher_js_mass": 1.0, "sampled_b": 0.10},
        {"native_trajectory_budget_explained_fraction": 0.1, "native_trajectory_budget_projection_mass": 0.1, "native_trajectory_teacher_js_mass": 1.0, "sampled_b": 0.10},
        {"native_trajectory_budget_explained_fraction": 0.4, "native_trajectory_budget_projection_mass": 0.4, "native_trajectory_teacher_js_mass": 1.0, "sampled_b": 0.20},
        {"native_trajectory_budget_explained_fraction": 0.4, "native_trajectory_budget_projection_mass": 0.4, "native_trajectory_teacher_js_mass": 1.0, "sampled_b": 0.20},
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "ratio_group_counterfactual_teachability_batch",
                "group_transform": "softmax",
                "temperature": 0.1,
                "eps": 1e-8,
            }
        }
    }
    weighted, output = apply_distributed_trajectory_weighting(
        losses, metrics, cfg, distributed=False, rank=0, world_size=1
    )
    expected_weights = torch.softmax(torch.tensor([0.1, 0.1, 0.4, 0.4]) / 0.1, dim=0)
    expected = torch.dot(expected_weights, torch.arange(1.0, 5.0))
    assert weighted.item() == pytest.approx(expected.item())
    assert output["trajectory_probability_weight_sum"] == pytest.approx(1.0)
    assert output["trajectory_weight_transform"] == "ratio_group_softmax_projection"
    assert output["trajectory_weight_temperature"] == pytest.approx(0.1)
    weighted.backward()
    assert losses[2].grad.item() > losses[0].grad.item()


def test_effective_batch_local_objective_has_expected_detached_scale() -> None:
    loss = torch.tensor(2.0, requires_grad=True)
    source_weight = torch.tensor(0.125, requires_grad=True)
    objective = effective_batch_local_objective(
        loss, source_weight, effective_batch_size=32
    )
    assert objective.item() == pytest.approx(8.0)
    objective.backward()
    assert loss.grad.item() == pytest.approx(4.0)
    assert source_weight.grad is None


def test_single_process_jsd_current_kl_weighting_is_probability_normalized() -> None:
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    metrics = [
        {"native_student_budget_jsd_mean": 0.01, "native_teacher_gap_b_mean": 0.10},
        {"native_student_budget_jsd_mean": 0.08, "native_teacher_gap_b_mean": 0.10},
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "jsd_over_current_kl_batch",
                "eps": 1e-8,
            }
        }
    }
    weighted, output = apply_distributed_trajectory_weighting(
        losses,
        metrics,
        cfg,
        distributed=False,
        rank=0,
        world_size=1,
    )
    expected_weights = inverse_sensitivity_probability_weights(torch.tensor([0.1, 0.8]))
    expected = (expected_weights.probability_weight * torch.tensor([1.0, 3.0])).sum()
    assert weighted.item() == pytest.approx(expected.item())
    assert output["trajectory_probability_weight_sum"] == pytest.approx(1.0)
    assert output["trajectory_signal_min"] == pytest.approx(0.1)
    assert output["trajectory_signal_max"] == pytest.approx(0.8)
    weighted.backward()
    assert losses[0].grad.item() > losses[1].grad.item()


def test_step0_kl_weighting_uses_ratio_specific_frozen_denominator() -> None:
    losses = [torch.tensor(1.0), torch.tensor(2.0)]
    metrics = [
        {"native_student_budget_jsd_mean": 0.01, "sampled_b": 0.10},
        {"native_student_budget_jsd_mean": 0.01, "sampled_b": 0.20},
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "jsd_over_step0_kl_batch",
                "eps": 1e-8,
                "step0_teacher_kl_by_ratio": {"0.10": 0.10, "0.20": 0.02},
            }
        }
    }
    _, output = apply_distributed_trajectory_weighting(
        losses,
        metrics,
        cfg,
        distributed=False,
        rank=0,
        world_size=1,
    )
    assert output["trajectory_signal_min"] == pytest.approx(0.1)
    assert output["trajectory_signal_max"] == pytest.approx(0.5)
    assert output["trajectory_probability_weight_sum"] == pytest.approx(1.0)


def test_ratio_group_teachability_weighting_is_probability_normalized() -> None:
    losses = [
        torch.tensor(1.0, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(3.0, requires_grad=True),
        torch.tensor(4.0, requires_grad=True),
    ]
    metrics = [
        {"native_trajectory_budget_explained_fraction": 0.2, "native_trajectory_budget_projection_mass": 0.1, "native_trajectory_teacher_js_mass": 0.5, "sampled_b": 0.10},
        {"native_trajectory_budget_explained_fraction": 0.6, "native_trajectory_budget_projection_mass": 0.3, "native_trajectory_teacher_js_mass": 0.5, "sampled_b": 0.10},
        {"native_trajectory_budget_explained_fraction": 0.8, "native_trajectory_budget_projection_mass": 0.4, "native_trajectory_teacher_js_mass": 0.5, "sampled_b": 0.20},
        {"native_trajectory_budget_explained_fraction": 0.4, "native_trajectory_budget_projection_mass": 0.2, "native_trajectory_teacher_js_mass": 0.5, "sampled_b": 0.20},
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "ratio_group_counterfactual_teachability_batch",
                "eps": 1e-8,
            }
        }
    }
    weighted, output = apply_distributed_trajectory_weighting(
        losses,
        metrics,
        cfg,
        distributed=False,
        rank=0,
        world_size=1,
    )
    expected = ratio_group_fraction_probability_weights(
        torch.tensor([0.1, 0.3, 0.4, 0.2]),
        torch.tensor([0.5, 0.5, 0.5, 0.5]),
        torch.tensor([0.10, 0.10, 0.20, 0.20]),
    )
    expected_loss = (expected.probability_weight * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum()
    assert weighted.item() == pytest.approx(expected_loss.item())
    assert output["trajectory_probability_weight_sum"] == pytest.approx(1.0)
    weighted.backward()
    assert losses[0].grad.item() == pytest.approx(losses[1].grad.item())
    assert losses[2].grad.item() == pytest.approx(losses[3].grad.item())
    assert losses[2].grad.item() > losses[0].grad.item()


def test_teacher_gap_mass_robustness_rewards_smaller_calibrated_gap() -> None:
    teacher_gap = torch.tensor([0.20, 0.10, 10.0])
    valid = torch.tensor([True, True, False])
    small = teacher_gap_mass_robustness(
        teacher_gap, torch.tensor([0.01, 0.01, 0.0]), valid
    )
    large = teacher_gap_mass_robustness(
        teacher_gap, torch.tensor([0.10, 0.10, 0.0]), valid
    )
    assert small > large
    assert not small.requires_grad


def test_robustness_gated_curriculum_moves_easy_to_hard_and_preserves_mass() -> None:
    losses = torch.tensor([0.01, 0.10], requires_grad=True)
    robustness = torch.tensor([0.8, 0.8], requires_grad=True)
    teacher_gap = losses.detach().clone()
    early = robustness_gated_curriculum_weights(
        losses,
        robustness,
        teacher_gap,
        curriculum_stage=0.0,
        log_teacher_gap_center=torch.log(torch.tensor(0.03)).item(),
        log_teacher_gap_scale=1.0,
        weight_floor=0.1,
    )
    late = robustness_gated_curriculum_weights(
        losses,
        robustness,
        teacher_gap,
        curriculum_stage=1.0,
        log_teacher_gap_center=torch.log(torch.tensor(0.03)).item(),
        log_teacher_gap_scale=1.0,
        weight_floor=0.1,
    )
    assert early.priority[0] > early.priority[1]
    assert late.priority[1] > late.priority[0]
    assert torch.allclose((early.weight * losses.detach()).sum(), losses.detach().sum())
    assert torch.allclose((late.weight * losses.detach()).sum(), losses.detach().sum())
    assert not early.weight.requires_grad
    assert not early.priority.requires_grad
    (early.weight * losses).mean().backward()
    assert robustness.grad is None


def test_curriculum_state_updates_and_roundtrips() -> None:
    state = RobustnessGatedCurriculumState(
        initial_teacher_gap_mean=0.03,
        ema_teacher_gap_mean=0.03,
        ema_half_life_trajectories=256.0,
        progress_power=3.0,
    )
    assert state.stage == pytest.approx(0.0)
    update = state.update(0.015, 256)
    assert update["ema_decay"] == pytest.approx(0.5)
    assert state.stage > 0.0
    restored = RobustnessGatedCurriculumState(
        initial_teacher_gap_mean=0.03,
        ema_teacher_gap_mean=0.03,
        ema_half_life_trajectories=256.0,
        progress_power=3.0,
    )
    restored.load_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()


def test_average_rank_priority_handles_ties_without_order_bias() -> None:
    values = torch.tensor([2.0, 1.0, 2.0, 4.0])
    priority = average_rank_priority(values, higher_is_better=True)
    assert torch.allclose(priority, torch.tensor([0.5, 0.0, 0.5, 1.0]))
    inverse = average_rank_priority(values, higher_is_better=False)
    assert torch.allclose(inverse, 1.0 - priority)


def test_trajectory_downweight_preserves_global_kl_mass() -> None:
    losses = torch.tensor([0.01, 0.04, 0.02, 0.08], requires_grad=True)
    signals = torch.tensor([0.1, 0.9, 0.4, 0.7], requires_grad=True)
    result = trajectory_rank_downweights(
        losses,
        signals,
        downweight_strength=0.25,
        higher_is_better=True,
    )
    assert result.raw_weight.min().item() == pytest.approx(0.75)
    assert result.raw_weight.max().item() == pytest.approx(1.0)
    assert torch.allclose(
        (result.weight * losses.detach()).sum(), losses.detach().sum(), atol=2e-7
    )
    assert not result.weight.requires_grad
    assert not result.priority.requires_grad
    weighted_loss = (result.weight * losses).mean()
    weighted_loss.backward()
    assert torch.allclose(losses.grad, result.weight / losses.numel())
    assert signals.grad is None


def test_zero_strength_is_exact_vanilla_weighting() -> None:
    losses = torch.tensor([0.01, 0.04, 0.02, 0.08])
    signals = torch.tensor([0.1, 0.9, 0.4, 0.7])
    result = trajectory_rank_downweights(
        losses,
        signals,
        downweight_strength=0.0,
        higher_is_better=True,
    )
    assert torch.equal(result.raw_weight, torch.ones_like(losses))
    assert torch.equal(result.weight, torch.ones_like(losses))
    assert result.loss_mass_scale.item() == pytest.approx(1.0)


def test_single_trajectory_cannot_change_its_own_scale() -> None:
    result = trajectory_rank_downweights(
        torch.tensor([0.03]),
        torch.tensor([0.8]),
        downweight_strength=0.5,
        higher_is_better=True,
    )
    assert result.weight.item() == pytest.approx(1.0)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        trajectory_rank_downweights(
            torch.tensor([0.1]),
            torch.tensor([0.2]),
            downweight_strength=1.0,
            higher_is_better=True,
        )
    with pytest.raises(FloatingPointError):
        average_rank_priority(torch.tensor([float("nan")]), higher_is_better=True)


def test_residualized_sensitivity_removes_gap_and_ratio_prediction() -> None:
    sensitivity = torch.tensor([0.02, 0.08])
    teacher_gap = torch.tensor([0.01, 0.04])
    ratios = torch.tensor([0.10, 0.20])
    residual = residualized_budget_sensitivity(
        sensitivity,
        teacher_gap,
        ratios,
        ratio_intercepts={
            "0.10": torch.log(torch.tensor(2.0)).item(),
            "0.20": torch.log(torch.tensor(2.0)).item(),
        },
        ratio_log_teacher_gap_coefficients={"0.10": 1.0, "0.20": 1.0},
        ratio_scales={"0.10": 1.0, "0.20": 1.0},
    )
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
    assert not residual.requires_grad


def test_residualized_sensitivity_clamps_only_fp32_roundoff() -> None:
    calibration = {
        "ratio_intercepts": {"0.10": 0.0},
        "ratio_log_teacher_gap_coefficients": {"0.10": 0.0},
        "ratio_scales": {"0.10": 1.0},
    }
    result = residualized_budget_sensitivity(
        torch.tensor([-2e-7]),
        torch.tensor([0.01]),
        torch.tensor([0.10]),
        **calibration,
    )
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError):
        residualized_budget_sensitivity(
            torch.tensor([-2e-5]),
            torch.tensor([0.01]),
            torch.tensor([0.10]),
            **calibration,
        )


def test_sigmoid_residual_weighting_prefers_smaller_gap_and_preserves_mass() -> None:
    losses = torch.tensor([0.01, 0.04, 0.02])
    residuals = torch.tensor([-2.0, 0.0, 2.0])
    result = trajectory_sigmoid_downweights(
        losses,
        residuals,
        downweight_strength=0.25,
    )
    assert result.priority[0] > result.priority[1] > result.priority[2]
    assert result.raw_weight[0] > result.raw_weight[1] > result.raw_weight[2]
    assert torch.allclose(
        (result.weight * losses).sum(), losses.sum(), rtol=2e-6, atol=2e-7
    )
    assert not result.weight.requires_grad


def _calibrated_frontier_state(*, target: int = 2) -> SensitivityFrontierState:
    state = SensitivityFrontierState(
        calibration_target_per_ratio=target,
        ema_half_life_trajectories=16.0,
        progress_drop_scale=0.5,
        progress_power=2.0,
    )
    ratios = [0.1, 0.2, 0.3, 0.4]
    for _ in range(target):
        state.update(ratios, [1.0, 1.0, 1.0, 1.0])
    assert state.ready
    return state


def test_sensitivity_frontier_calibrates_exact_first_n_per_ratio() -> None:
    state = SensitivityFrontierState(calibration_target_per_ratio=2)
    first = state.update(
        [0.1, 0.1, 0.2, 0.3, 0.4],
        [1.0, 3.0, 2.0, 4.0, 8.0],
    )
    assert not first["ready_after"]
    second = state.update([0.2, 0.3, 0.4], [6.0, 8.0, 16.0])
    assert second["ready_after"]
    assert state.initial_sensitivity == pytest.approx(
        {"0.10": 2.0, "0.20": 4.0, "0.30": 6.0, "0.40": 12.0}
    )
    assert state.ema_sensitivity == pytest.approx(state.initial_sensitivity)
    assert state.calibration_complete_at_trajectory == 8


def test_sensitivity_frontier_rewards_smaller_gap_and_preserves_kl_mass() -> None:
    state = _calibrated_frontier_state()
    losses = torch.tensor([0.01, 0.03], requires_grad=True)
    sensitivities = torch.tensor([0.1, 2.0], requires_grad=True)
    result = sensitivity_frontier_weights(
        losses,
        sensitivities,
        torch.tensor([0.4, 0.4]),
        state,
        weight_floor=0.1,
    )
    assert result.local_robustness[0] > result.local_robustness[1]
    assert result.raw_weight[0] > result.raw_weight[1]
    assert result.weight[0] > result.weight[1]
    assert torch.allclose(
        (result.weight * losses.detach()).sum(), losses.detach().sum(), atol=2e-7
    )
    assert not result.weight.requires_grad
    assert not result.local_robustness.requires_grad
    (result.weight * losses).mean().backward()
    assert sensitivities.grad is None


def test_sensitivity_frontier_weight_floor_one_is_exact_vanilla() -> None:
    state = _calibrated_frontier_state()
    losses = torch.tensor([0.01, 0.03, 0.02])
    result = sensitivity_frontier_weights(
        losses,
        torch.tensor([0.1, 1.0, 5.0]),
        torch.tensor([0.4, 0.3, 0.2]),
        state,
        weight_floor=1.0,
    )
    assert torch.equal(result.raw_weight, torch.ones_like(losses))
    assert torch.equal(result.weight, torch.ones_like(losses))
    assert result.loss_mass_scale.item() == pytest.approx(1.0)


def test_sensitivity_frontier_moves_from_high_to_low_ratio() -> None:
    state = _calibrated_frontier_state()
    assert state.ratio_gate(0.4) == pytest.approx(1.0)
    assert state.ratio_gate(0.3) == pytest.approx(0.0)

    target_unresolved = 1.0 - 0.5 * (1.0 / 3.0) ** 0.5
    state.ema_sensitivity = {
        key: value * target_unresolved
        for key, value in state.initial_sensitivity.items()
    }
    assert state.frontier_index == pytest.approx(1.0)
    assert state.ratio_gate(0.3) == pytest.approx(1.0)

    target_unresolved = 1.0 - 0.5 * (2.0 / 3.0) ** 0.5
    state.ema_sensitivity = {
        key: value * target_unresolved
        for key, value in state.initial_sensitivity.items()
    }
    assert state.frontier_index == pytest.approx(2.0)
    assert state.ratio_gate(0.2) == pytest.approx(1.0)

    state.ema_sensitivity = {
        key: value * 0.5 for key, value in state.initial_sensitivity.items()
    }
    assert state.frontier_index == pytest.approx(3.0)
    assert state.ratio_gate(0.1) == pytest.approx(1.0)


def test_sensitivity_frontier_state_roundtrip_is_exact() -> None:
    state = _calibrated_frontier_state()
    state.update([0.1, 0.2, 0.3, 0.4], [0.8, 0.7, 0.6, 0.5])
    restored = SensitivityFrontierState(
        calibration_target_per_ratio=2,
        ema_half_life_trajectories=16.0,
        progress_drop_scale=0.5,
        progress_power=2.0,
    )
    restored.load_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()


def test_sensitivity_frontier_rejects_unknown_ratio_and_nonfinite_signal() -> None:
    state = _calibrated_frontier_state()
    with pytest.raises(ValueError):
        state.update([0.25], [1.0])
    with pytest.raises(ValueError):
        state.update([0.2], [float("nan")])


def test_distributed_sensitivity_frontier_warmup_and_weighted_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)

    def fake_all_gather(outputs: list[torch.Tensor], value: torch.Tensor) -> None:
        for output in outputs:
            output.copy_(value)

    monkeypatch.setattr(train_module.dist, "all_gather", fake_all_gather)
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": "sensitivity_frontier",
                "weight_floor": 0.1,
                "downweight_strength": 0.9,
                "eps": 1e-8,
            }
        }
    }
    state = SensitivityFrontierState(calibration_target_per_ratio=1)
    losses = [torch.tensor(value, requires_grad=True) for value in (0.01, 0.02, 0.03, 0.04)]
    metrics = [
        {
            "sampled_b": ratio,
            "native_trajectory_teacher_gap_sensitivity": sensitivity,
        }
        for ratio, sensitivity in zip((0.1, 0.2, 0.3, 0.4), (1.0, 1.0, 1.0, 1.0))
    ]
    warmup_loss, warmup_metrics = apply_distributed_trajectory_weighting(
        losses,
        metrics,
        cfg,
        distributed=True,
        rank=0,
        world_size=2,
        curriculum_state=state,
    )
    assert state.ready
    assert torch.allclose(warmup_loss, torch.stack(losses).mean(), atol=2e-7)
    assert warmup_metrics["trajectory_frontier_ready_before"] is False
    assert warmup_metrics["trajectory_frontier_ready_after"] is True

    weighted_loss, weighted_metrics = apply_distributed_trajectory_weighting(
        losses,
        metrics,
        cfg,
        distributed=True,
        rank=0,
        world_size=2,
        curriculum_state=state,
    )
    assert weighted_metrics["trajectory_frontier_ready_before"] is True
    assert weighted_metrics["trajectory_weight_detached"] is True
    assert weighted_metrics["trajectory_global_scalar_error"] == pytest.approx(0.0, abs=2e-7)
    assert torch.allclose(
        weighted_loss.detach(), torch.stack(losses).mean().detach(), atol=2e-7
    )
