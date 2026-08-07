from __future__ import annotations

import json
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_mathvista_pass_at_k import extract_candidate, is_correct, pass_at_k
from run_mathvista_pass_at_k import measured_visionzip_metadata, pruned_prefix_hash


class MathVistaPassKTest(unittest.TestCase):
    def test_closed_answer_uses_final_number_in_span(self) -> None:
        candidate, method = extract_candidate(
            "<think>2 + 3 = 5</think><answer>Therefore 5 + 1 = 6</answer>",
            "integer",
        )
        self.assertEqual(candidate, "6")
        self.assertEqual(method, "closed_answer_tag")

    def test_open_answer_is_parseable(self) -> None:
        candidate, method = extract_candidate("<answer>The final answer is 3.5.", "float")
        self.assertEqual(candidate, "3.5")
        self.assertEqual(method, "open_answer_tag")

    def test_reasoning_number_without_final_marker_is_not_scored(self) -> None:
        candidate, method = extract_candidate("We first read 42 from the image.", "integer")
        self.assertIsNone(candidate)
        self.assertEqual(method, "unparseable_no_final_marker")

    def test_integer_rejects_noninteger_candidate(self) -> None:
        candidate, method = extract_candidate("<answer>3.2</answer>", "integer")
        self.assertIsNone(candidate)
        self.assertEqual(method, "closed_answer_tag_noninteger")

    def test_exact_numeric_correctness(self) -> None:
        self.assertTrue(is_correct("10.4", "10.4", "float"))
        self.assertFalse(is_correct("10.5", "10.4", "float"))

    def test_pass_at_k_boundaries(self) -> None:
        self.assertEqual(pass_at_k(64, 0, 64), 0.0)
        self.assertEqual(pass_at_k(64, 1, 64), 1.0)
        self.assertAlmostEqual(pass_at_k(64, 16, 1), 0.25)

    @unittest.skipUnless(find_spec("torch") is not None, "torch is supplied by the clean evaluation environment")
    def test_fixed_mask_hash_tracks_original_positions(self) -> None:
        import torch

        base = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "position_ids": torch.tensor([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype=torch.long),
        }
        same = {key: value.clone() for key, value in base.items()}
        changed = {key: value.clone() for key, value in base.items()}
        changed["position_ids"][0, 0, 1] = 9
        self.assertEqual(pruned_prefix_hash(base), pruned_prefix_hash(same))
        self.assertNotEqual(pruned_prefix_hash(base), pruned_prefix_hash(changed))

    @unittest.skipUnless(find_spec("torch") is not None, "torch is supplied by the clean evaluation environment")
    def test_fixed_mask_hash_ignores_generate_wrapper_position_row(self) -> None:
        import torch

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        mrope = torch.tensor(
            [
                [[0, 1, 2, 3]],
                [[0, 4, 5, 6]],
                [[0, 7, 8, 9]],
            ],
            dtype=torch.long,
        )
        wrapped = torch.cat(
            [torch.tensor([[[99, 98, 97, 96]]], dtype=torch.long), mrope],
            dim=0,
        )
        direct = {"input_ids": input_ids, "position_ids": mrope}
        generated = {"input_ids": input_ids, "position_ids": wrapped}
        self.assertEqual(pruned_prefix_hash(direct), pruned_prefix_hash(generated))

    @unittest.skipUnless(find_spec("torch") is not None, "torch is supplied by the clean evaluation environment")
    def test_visionzip_metadata_uses_measured_count_for_rounding_mismatch(self) -> None:
        import torch

        class Config:
            image_token_id = 99

        class Model:
            config = Config()

        full_ids = torch.tensor([[1] + [99] * 20 + [2]], dtype=torch.long)
        pruned_ids = torch.tensor([[1] + [99] * 4 + [2]], dtype=torch.long)

        def official(*_args, **_kwargs):
            raise RuntimeError("Official VisionZip token count mismatch: kept=4, dominant+contextual=5.")

        metadata = measured_visionzip_metadata(
            Model(),
            {"input_ids": full_ids},
            int(full_ids.shape[1]),
            0.2,
            0.05,
            official,
            lambda _model: {"input_ids": pruned_ids},
        )
        self.assertEqual(metadata["num_full_visual_tokens"], 20)
        self.assertEqual(metadata["num_kept_visual_tokens"], 4)
        self.assertEqual(metadata["visionzip_target_tokens"], 5)
        self.assertEqual(metadata["visionzip_target_count_delta"], -1)
        self.assertFalse(metadata["visionzip_target_count_match"])

    def test_selected_dataset_has_no_multiple_choice(self) -> None:
        path = ROOT / "pass_at_k" / "samples" / "mathvista_open_seed42_n100.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({str(row["sample_id"]) for row in rows}), 100)
        self.assertTrue(all(row["question_type"] == "free_form" for row in rows))
        self.assertTrue(all("Choices:" not in row["prompt"] for row in rows))


if __name__ == "__main__":
    unittest.main()
