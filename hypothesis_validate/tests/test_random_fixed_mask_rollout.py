from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from opsd.pruning_distill.pruners import RandomPruner  # noqa: E402
from run_random_fixed_mask_rollout_experiment import (  # noqa: E402
    build_mask_document,
    derive_sample_mask_seed,
    mask_file_name,
)


class RandomFixedMaskRolloutTest(unittest.TestCase):
    def test_sample_mask_seed_is_stable_and_sample_specific(self) -> None:
        first = derive_sample_mask_seed(42, "sample-a")
        self.assertEqual(first, derive_sample_mask_seed(42, "sample-a"))
        self.assertNotEqual(first, derive_sample_mask_seed(42, "sample-b"))
        self.assertNotEqual(first, derive_sample_mask_seed(43, "sample-a"))

    def test_mask_hash_is_stable_and_sensitive_to_indices(self) -> None:
        source = {
            "sample_id": "7",
            "sample_rank": 2,
            "image_sha256": "image-hash",
            "question": "How many?",
        }
        kwargs = dict(
            source=source,
            mask_seed_base=42,
            sample_mask_seed=123,
            retention_ratio=0.2,
            num_full_visual_tokens=10,
            prefix_hash="prefix",
            image_grid_thw=[[1, 4, 4]],
        )
        first = build_mask_document(kept_indices=[1, 8], **kwargs)
        same = build_mask_document(kept_indices=[1, 8], **kwargs)
        changed = build_mask_document(kept_indices=[2, 8], **kwargs)
        self.assertEqual(first, same)
        self.assertNotEqual(first["fixed_mask_hash"], changed["fixed_mask_hash"])
        self.assertEqual(first["num_kept_visual_tokens"], 2)
        self.assertEqual(first["realized_retention_ratio"], 0.2)

    def test_mask_file_name_is_safe(self) -> None:
        name = mask_file_name(3, "a/b c")
        self.assertEqual(name, "sample_003_a_b_c.json")
        self.assertNotIn("/", name)

    def test_explicit_sample_seed_matches_repository_metadata_path(self) -> None:
        sample_id = "1226"
        embeds = torch.zeros(1333, 4)
        derived_seed = derive_sample_mask_seed(42, sample_id)
        via_metadata = RandomPruner(seed=42).select(
            embeds,
            grid_thw=None,
            keep_ratio=0.2,
            metadata={"sample_id": sample_id},
        )
        via_explicit_seed = RandomPruner(seed=derived_seed).select(
            embeds,
            grid_thw=None,
            keep_ratio=0.2,
            metadata=None,
        )
        self.assertTrue(torch.equal(via_metadata, via_explicit_seed))

    def test_same_sample_random_masks_are_nested_across_ratios(self) -> None:
        embeds = torch.zeros(1333, 4)
        pruner = RandomPruner(seed=derive_sample_mask_seed(42, "1226"))
        selected = {
            ratio: set(pruner.select(embeds, None, ratio).tolist())
            for ratio in (0.1, 0.2, 0.3, 0.4, 1.0)
        }
        for lower, upper in zip((0.1, 0.2, 0.3, 0.4), (0.2, 0.3, 0.4, 1.0), strict=True):
            self.assertLess(selected[lower], selected[upper])


if __name__ == "__main__":
    unittest.main()
