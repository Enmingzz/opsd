#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

OPSD_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "opsd")
sys.path.insert(0, str(OPSD_ROOT.parent))

from opsd.visionzip_aokvqa.trajectory_weighting import (
    effective_batch_local_objective,
    inverse_sensitivity_probability_weights,
)


def run(mode: str, rank: int, world_size: int) -> None:
    accumulation_steps = 8
    effective_batch_size = world_size * accumulation_steps
    coefficients = torch.arange(1, effective_batch_size + 1, dtype=torch.float32)
    signal = torch.linspace(0.05, 1.60, effective_batch_size)
    probability = inverse_sensitivity_probability_weights(signal).probability_weight
    theta = torch.tensor(1.0, requires_grad=True)
    local_objective = torch.zeros((), dtype=torch.float32)
    for accumulation_index in range(accumulation_steps):
        global_index = rank * accumulation_steps + accumulation_index
        loss = coefficients[global_index] * theta
        local_objective = local_objective + effective_batch_local_objective(
            loss,
            probability[global_index],
            effective_batch_size=effective_batch_size,
        ) / accumulation_steps
    local_objective.backward()
    gradients = [torch.zeros_like(theta.grad) for _ in range(world_size)]
    dist.all_gather(gradients, theta.grad)
    if rank == 0:
        expected_gradient = float((probability * coefficients).sum())
        ddp_averaged_gradient = float(torch.stack(gradients).mean())
        if not math.isclose(ddp_averaged_gradient, expected_gradient, rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError(
                f"{mode}: DDP scaling mismatch {ddp_averaged_gradient} vs {expected_gradient}"
            )
        if not math.isclose(float(probability.sum()), 1.0, abs_tol=1e-7):
            raise AssertionError(f"{mode}: effective-batch probability weights do not sum to one")
        print(f"PASS {mode}: ddp_gradient={ddp_averaged_gradient:.8f}")


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError(f"Expected four ranks, got {world_size}")
    dist.init_process_group("gloo")
    try:
        run("jsd_over_current_kl_batch", rank, world_size)
        run("jsd_over_step0_kl_batch", rank, world_size)
    finally:
        dist.destroy_process_group()
    if rank == 0:
        print("PASS distributed trajectory JSD weighting smoke")


if __name__ == "__main__":
    main()
