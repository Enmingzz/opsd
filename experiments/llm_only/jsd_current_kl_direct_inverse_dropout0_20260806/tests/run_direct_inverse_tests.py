#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

OPSD_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "opsd")
sys.path.insert(0, str(OPSD_ROOT.parent))

from opsd.visionzip_aokvqa.train import apply_distributed_trajectory_weighting
from opsd.visionzip_aokvqa.trajectory_weighting import (
    direct_inverse_sensitivity_probability_weights,
    effective_batch_local_objective,
)


MODE = "jsd_over_current_kl_direct_inverse_batch"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    sensitivity = torch.tensor([0.05, 0.10, 0.20], requires_grad=True)
    result = direct_inverse_sensitivity_probability_weights(sensitivity)
    expected_raw = sensitivity.detach().reciprocal()
    expected = expected_raw / expected_raw.sum()
    check(not hasattr(result, "tau"), "direct inverse unexpectedly contains tau")
    check(torch.allclose(result.probability_weight, expected), "direct inverse formula mismatch")
    check(math.isclose(float(result.probability_weight.sum()), 1.0, abs_tol=1e-7), "weights do not sum to one")
    check(bool(result.probability_weight[0] > result.probability_weight[1] > result.probability_weight[2]), "lower sensitivity did not receive more weight")
    check(not result.probability_weight.requires_grad, "weights are not detached")

    uniform = direct_inverse_sensitivity_probability_weights(torch.full((4,), 0.1))
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    check(torch.allclose(uniform.probability_weight, torch.full((4,), 0.25)), "uniform signal is not uniform")
    check(torch.allclose(torch.dot(uniform.probability_weight, losses), losses.mean()), "uniform weights do not recover vanilla mean")

    sample_losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    metrics = [
        {"native_student_budget_jsd_mean": 0.01, "native_teacher_gap_b_mean": 0.10},
        {"native_student_budget_jsd_mean": 0.08, "native_teacher_gap_b_mean": 0.10},
    ]
    cfg = {"opsd": {"trajectory_weighting": {"enabled": True, "mode": MODE, "eps": 1e-8}}}
    objective, output = apply_distributed_trajectory_weighting(
        sample_losses, metrics, cfg, distributed=False, rank=0, world_size=1
    )
    expected_probability = direct_inverse_sensitivity_probability_weights(
        torch.tensor([0.1, 0.8])
    ).probability_weight
    expected_objective = torch.dot(expected_probability, torch.tensor([1.0, 3.0]))
    check(torch.allclose(objective.detach(), expected_objective), "training objective mismatch")
    check(output["trajectory_batch_tau"] is None, "tau is still present")
    check(output["trajectory_weight_transform"] == "direct_inverse", "wrong transform logged")
    check(math.isclose(output["trajectory_probability_weight_sum"], 1.0, abs_tol=1e-7), "training weights do not sum to one")
    objective.backward()
    check(float(sample_losses[0].grad) > float(sample_losses[1].grad), "robust sample did not receive larger gradient")

    scalar_loss = torch.tensor(2.0, requires_grad=True)
    source_weight = torch.tensor(0.125, requires_grad=True)
    local_objective = effective_batch_local_objective(
        scalar_loss, source_weight, effective_batch_size=32
    )
    local_objective.backward()
    check(math.isclose(float(scalar_loss.grad), 4.0), "effective-batch gradient scale mismatch")
    check(source_weight.grad is None, "probability weight leaked gradients")
    print("PASS direct inverse no-tau weighting tests")


if __name__ == "__main__":
    main()
