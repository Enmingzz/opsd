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
    effective_batch_local_objective,
    softmax_inverse_sensitivity_probability_weights,
)


MODE = "jsd_over_current_kl_softmax_batch"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_temperature(temperature: float) -> None:
    source = torch.tensor([0.05, 0.10, 0.20], requires_grad=True)
    result = softmax_inverse_sensitivity_probability_weights(
        source, temperature=temperature
    )
    expected = torch.softmax(-source.detach() / temperature, dim=0)
    check(torch.allclose(result.probability_weight, expected), "softmax formula mismatch")
    check(
        math.isclose(float(result.probability_weight.sum()), 1.0, abs_tol=1e-6),
        "weights do not sum to one within FP32 tolerance",
    )
    check(bool(result.probability_weight[0] > result.probability_weight[1] > result.probability_weight[2]), "lower sensitivity did not receive more weight")
    check(not result.probability_weight.requires_grad, "weight branch is not detached")
    check(source.grad is None, "weight branch populated sensitivity gradients")

    uniform = softmax_inverse_sensitivity_probability_weights(
        torch.full((4,), 0.1), temperature=temperature
    )
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    check(torch.allclose(uniform.probability_weight, torch.full((4,), 0.25)), "uniform signal is not uniform")
    check(torch.allclose(torch.dot(uniform.probability_weight, losses), losses.mean()), "uniform weights do not recover vanilla mean")

    sample_losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    metrics = [
        {"native_student_budget_jsd_mean": 0.01, "native_teacher_gap_b_mean": 0.10},
        {"native_student_budget_jsd_mean": 0.08, "native_teacher_gap_b_mean": 0.10},
    ]
    cfg = {
        "opsd": {
            "trajectory_weighting": {
                "enabled": True,
                "mode": MODE,
                "temperature": temperature,
                "eps": 1e-8,
            }
        }
    }
    objective, output = apply_distributed_trajectory_weighting(
        sample_losses, metrics, cfg, distributed=False, rank=0, world_size=1
    )
    expected_probability = softmax_inverse_sensitivity_probability_weights(
        torch.tensor([0.1, 0.8]), temperature=temperature
    ).probability_weight
    check(torch.allclose(objective.detach(), torch.dot(expected_probability, torch.tensor([1.0, 3.0]))), "training objective mismatch")
    check(output["trajectory_batch_tau"] is None, "softmax unexpectedly uses tau")
    check(output["trajectory_weight_transform"] == "softmax_negative_sensitivity", "wrong transform logged")
    check(math.isclose(output["trajectory_weight_temperature"], temperature), "wrong temperature logged")
    objective.backward()
    check(float(sample_losses[0].grad) > float(sample_losses[1].grad), "robust sample did not receive larger gradient")


def main() -> None:
    for temperature in (0.05, 0.1):
        run_temperature(temperature)
        print(f"PASS softmax trajectory weighting T={temperature}")
    scalar_loss = torch.tensor(2.0, requires_grad=True)
    source_weight = torch.tensor(0.125, requires_grad=True)
    objective = effective_batch_local_objective(
        scalar_loss, source_weight, effective_batch_size=32
    )
    objective.backward()
    check(math.isclose(float(scalar_loss.grad), 4.0), "effective-batch scale mismatch")
    check(source_weight.grad is None, "probability weight leaked gradients")
    print("PASS all softmax no-tau weighting tests")


if __name__ == "__main__":
    main()
