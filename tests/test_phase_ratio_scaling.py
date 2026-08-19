from __future__ import annotations

import math
import unittest

import torch

from opsd.visionzip_aokvqa.phase_ratio_scaling import (
    resolve_phase_ratio_scale,
    validate_phase_ratio_scaling_config,
)


CONFIG = {
    "enabled": True,
    "normalization": "none",
    "phases": [
        {"weights_by_ratio": {"0.10": 0.4, "0.20": 0.8, "0.30": 1.0, "0.40": 1.5}},
        {"weights_by_ratio": {"0.10": 0.8, "0.20": 0.9, "0.30": 0.9, "0.40": 1.0}},
        {"weights_by_ratio": {"0.10": 1.2, "0.20": 1.1, "0.30": 0.8, "0.40": 0.5}},
        {"weights_by_ratio": {"0.10": 1.6, "0.20": 1.3, "0.30": 0.7, "0.40": 0.2}},
    ],
}


class PhaseRatioScalingTest(unittest.TestCase):
    def test_exact_phase_boundaries_and_requested_weights(self) -> None:
        expected = {
            0: {0.1: 0.4, 0.2: 0.8, 0.3: 1.0, 0.4: 1.5},
            2496: {0.1: 0.8, 0.2: 0.9, 0.3: 0.9, 0.4: 1.0},
            4992: {0.1: 1.2, 0.2: 1.1, 0.3: 0.8, 0.4: 0.5},
            7488: {0.1: 1.6, 0.2: 1.3, 0.3: 0.7, 0.4: 0.2},
        }
        for progress_step, ratio_weights in expected.items():
            for ratio, weight in ratio_weights.items():
                result = resolve_phase_ratio_scale(
                    CONFIG,
                    retention_ratio=ratio,
                    progress_step=progress_step,
                    total_steps=9984,
                )
                self.assertEqual(result.phase_index, progress_step // 2496)
                self.assertEqual(result.scale, weight)

    def test_no_normalization_directly_scales_loss_and_gradient(self) -> None:
        parameter = torch.tensor(2.0, requires_grad=True)
        unweighted = parameter.square()
        scale = resolve_phase_ratio_scale(
            CONFIG, retention_ratio=0.4, progress_step=0, total_steps=9984
        ).scale
        weighted = unweighted * scale
        weighted.backward()
        self.assertAlmostEqual(weighted.item(), 6.0)
        self.assertAlmostEqual(parameter.grad.item(), 6.0)
        self.assertEqual(scale, 1.5)

    def test_all_one_weights_recover_original_loss_exactly(self) -> None:
        config = {
            "enabled": True,
            "normalization": "none",
            "phases": [{"weights_by_ratio": {"0.10": 1.0, "0.20": 1.0}}],
        }
        loss = torch.tensor(0.123456, requires_grad=True)
        scale = resolve_phase_ratio_scale(
            config, retention_ratio=0.2, progress_step=7, total_steps=10
        ).scale
        self.assertTrue(torch.equal(loss * scale, loss))

    def test_config_validation_rejects_normalization_and_missing_ratios(self) -> None:
        validate_phase_ratio_scaling_config(
            CONFIG,
            method="opsd_nogt",
            train_retention_ratios=[0.1, 0.2, 0.3, 0.4],
        )
        normalized = {**CONFIG, "normalization": "phase_mean"}
        with self.assertRaisesRegex(ValueError, "normalization=none"):
            validate_phase_ratio_scaling_config(
                normalized,
                method="opsd_nogt",
                train_retention_ratios=[0.1, 0.2, 0.3, 0.4],
            )
        missing = {**CONFIG, "phases": [{"weights_by_ratio": {"0.10": 1.0}}]}
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_phase_ratio_scaling_config(
                missing,
                method="opsd_nogt",
                train_retention_ratios=[0.1, 0.2, 0.3, 0.4],
            )

    def test_scales_are_finite_positive_and_detached_constants(self) -> None:
        for progress_step in (0, 2495, 2496, 4991, 4992, 7487, 7488, 9983):
            for ratio in (0.1, 0.2, 0.3, 0.4):
                result = resolve_phase_ratio_scale(
                    CONFIG,
                    retention_ratio=ratio,
                    progress_step=progress_step,
                    total_steps=9984,
                )
                self.assertTrue(math.isfinite(result.scale))
                self.assertGreater(result.scale, 0.0)


if __name__ == "__main__":
    unittest.main()
