from __future__ import annotations

import random
import unittest

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from opsd.visionzip_aokvqa.epic_official import (
    enable_visual_checkpoint_input_grads,
    extract_official_epic_response_logits,
    official_epic_curriculum_bounds,
    sample_official_epic_curriculum,
)
from opsd.visionzip_aokvqa.losses import compute_forward_kl


class OfficialEpicCurriculumTests(unittest.TestCase):
    def test_visual_checkpoint_hook_preserves_trainable_downstream_gradients(self):
        class ToyVisual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = torch.nn.Linear(3, 3, bias=False)
                self.patch_embed.requires_grad_(False)
                self.scale = torch.nn.Parameter(torch.ones(3))

            def forward(self, inputs):
                hidden = self.patch_embed(inputs)
                return checkpoint(
                    lambda value: value * self.scale,
                    hidden,
                    use_reentrant=True,
                )

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = ToyVisual()

            def forward(self, inputs):
                return self.visual(inputs)

        model = ToyModel()
        module_name = enable_visual_checkpoint_input_grads(model)
        self.assertEqual(module_name, "visual.patch_embed")
        model(torch.ones(2, 3)).sum().backward()
        self.assertIsNotNone(model.visual.scale.grad)
        self.assertGreater(torch.count_nonzero(model.visual.scale.grad).item(), 0)

    def test_official_curriculum_bounds(self):
        expected = {
            0.0: (0.10, 0.10, 0.10),
            0.5: (0.18, 0.42, 0.30),
            1.0: (0.26, 0.74, 0.30),
        }
        for progress, expected_values in expected.items():
            with self.subTest(progress=progress):
                observed = official_epic_curriculum_bounds(progress)
                for actual, target in zip(observed, expected_values, strict=True):
                    self.assertAlmostEqual(actual, target)

    def test_reduction_to_retention_mapping_and_teacher_gap(self):
        sample = sample_official_epic_curriculum(
            random.Random(42), optimizer_step=156, total_optimizer_steps=312
        )
        self.assertAlmostEqual(sample.progress, 0.5)
        self.assertAlmostEqual(sample.student_reduction_ratio, 0.33346243162989205)
        self.assertAlmostEqual(sample.teacher_reduction_ratio, 0.03346243162989204)
        self.assertAlmostEqual(sample.student_retention_ratio, 1.0 - sample.student_reduction_ratio)
        self.assertAlmostEqual(sample.teacher_retention_ratio, 1.0 - sample.teacher_reduction_ratio)
        self.assertGreaterEqual(sample.teacher_retention_ratio, sample.student_retention_ratio)

    def test_all_ranks_receive_identical_samples_with_official_seed(self):
        rank_zero_rng = random.Random(42)
        rank_three_rng = random.Random(42)
        rank_zero = [
            sample_official_epic_curriculum(rank_zero_rng, optimizer_step=step, total_optimizer_steps=312)
            for step in (0, 0, 1, 100, 311)
        ]
        rank_three = [
            sample_official_epic_curriculum(rank_three_rng, optimizer_step=step, total_optimizer_steps=312)
            for step in (0, 0, 1, 100, 311)
        ]
        self.assertEqual(rank_zero, rank_three)

    def test_invalid_optimizer_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            sample_official_epic_curriculum(random.Random(42), optimizer_step=-1, total_optimizer_steps=312)
        with self.assertRaises(ValueError):
            sample_official_epic_curriculum(random.Random(42), optimizer_step=0, total_optimizer_steps=0)

    def test_unshifted_response_positions_match_official_masked_kl(self):
        generator = torch.Generator().manual_seed(42)
        teacher = torch.randn(1, 7, 11, generator=generator)
        student = torch.randn(1, 7, 11, generator=generator)
        labels = torch.full((1, 7), -100, dtype=torch.long)
        labels[:, 3:7] = torch.tensor([[2, 4, 6, 8]])
        temperature = 2.0

        teacher_probs = F.softmax(teacher / temperature, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(student / temperature, dim=-1, dtype=torch.float32)
        token_kl = F.kl_div(student_logprobs, teacher_probs, reduction="none")
        official = (token_kl * (labels != -100).unsqueeze(-1)).sum() / (labels != -100).sum()
        official = official * temperature**2

        selected_teacher = extract_official_epic_response_logits(
            teacher, response_start=3, response_count=4
        )
        selected_student = extract_official_epic_response_logits(
            student, response_start=3, response_count=4
        )
        local = compute_forward_kl(selected_teacher, selected_student, temperature=temperature)
        self.assertTrue(torch.allclose(local, official, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(selected_teacher, teacher[0, 3:7]))

    def test_official_logit_slice_rejects_invalid_ranges(self):
        logits = torch.zeros(1, 4, 3)
        with self.assertRaises(ValueError):
            extract_official_epic_response_logits(logits, response_start=3, response_count=2)
        with self.assertRaises(ValueError):
            extract_official_epic_response_logits(logits, response_start=0, response_count=0)


if __name__ == "__main__":
    unittest.main()
