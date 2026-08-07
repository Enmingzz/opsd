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
    inverse_sensitivity_probability_weights,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_probability_weights() -> None:
    source = torch.tensor([0.05, 0.20, 0.80], requires_grad=True)
    result = inverse_sensitivity_probability_weights(source)
    check(math.isclose(float(result.probability_weight.sum()), 1.0, abs_tol=1e-7), "weights do not sum to one")
    check(bool(result.probability_weight[0] > result.probability_weight[1] > result.probability_weight[2]), "lower sensitivity did not receive more weight")
    check(not result.probability_weight.requires_grad, "weighting branch is not detached")
    check(source.grad is None, "weighting branch populated sensitivity gradients")
    zero = inverse_sensitivity_probability_weights(torch.zeros(4))
    check(torch.allclose(zero.probability_weight, torch.full((4,), 0.25)), "zero signal is not uniform")

    loss = torch.tensor(2.0, requires_grad=True)
    source_weight = torch.tensor(0.125, requires_grad=True)
    objective = effective_batch_local_objective(
        loss, source_weight, effective_batch_size=32
    )
    objective.backward()
    check(math.isclose(float(loss.grad), 4.0), "effective-batch gradient scale is wrong")
    check(source_weight.grad is None, "probability weight was not detached")


def run_mode(mode: str) -> None:
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    if mode == "jsd_over_current_kl_batch":
        metrics = [
            {"native_student_budget_jsd_mean": 0.01, "native_teacher_gap_b_mean": 0.10},
            {"native_student_budget_jsd_mean": 0.08, "native_teacher_gap_b_mean": 0.10},
        ]
        expected_signal = torch.tensor([0.1, 0.8])
        extra = {}
    else:
        metrics = [
            {"native_student_budget_jsd_mean": 0.01, "sampled_b": 0.10},
            {"native_student_budget_jsd_mean": 0.01, "sampled_b": 0.20},
        ]
        expected_signal = torch.tensor([0.1, 0.5])
        extra = {"step0_teacher_kl_by_ratio": {"0.10": 0.10, "0.20": 0.02}}
    cfg = {"opsd": {"trajectory_weighting": {"enabled": True, "mode": mode, "eps": 1e-8, **extra}}}
    weighted, output = apply_distributed_trajectory_weighting(
        losses, metrics, cfg, distributed=False, rank=0, world_size=1
    )
    expected_weight = inverse_sensitivity_probability_weights(expected_signal).probability_weight
    expected_loss = (expected_weight * torch.tensor([1.0, 3.0])).sum()
    check(torch.allclose(weighted.detach(), expected_loss), f"{mode}: weighted objective mismatch")
    check(math.isclose(output["trajectory_probability_weight_sum"], 1.0, abs_tol=1e-7), f"{mode}: normalization mismatch")
    weighted.backward()
    check(float(losses[0].grad) > float(losses[1].grad), f"{mode}: robust trajectory did not get larger gradient")


def main() -> None:
    tests = [test_probability_weights]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    for mode in ("jsd_over_current_kl_batch", "jsd_over_step0_kl_batch"):
        run_mode(mode)
        print(f"PASS {mode}")
    print("PASS all trajectory JSD weighting tests")


if __name__ == "__main__":
    main()
