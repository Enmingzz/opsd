from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_mmstar_open_ended_samples import (  # noqa: E402
    IMAGE_PLACEHOLDER,
    OPTION_DEPENDENT_STEM_PATTERNS,
)


class MMStarOpenEndedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "pass_at_k" / "samples" / "mmstar_open_ended_seed42_n100.jsonl"
        with path.open(encoding="utf-8") as handle:
            cls.rows = [json.loads(line) for line in handle if line.strip()]

    def test_sample_and_image_ids_are_unique(self) -> None:
        self.assertEqual(len(self.rows), 100)
        self.assertEqual(len({row["sample_id"] for row in self.rows}), 100)
        self.assertEqual(len({row["image_sha256"] for row in self.rows}), 100)

    def test_no_option_list_is_exposed_to_generation(self) -> None:
        self.assertTrue(all("Options:" not in row["prompt"] for row in self.rows))
        self.assertTrue(all("provide the correct option letter" not in row["prompt"] for row in self.rows))

    def test_invalid_image_placeholder_sample_is_known(self) -> None:
        invalid = {
            row["sample_id"]
            for row in self.rows
            if IMAGE_PLACEHOLDER.search(str(row["answer"]))
        }
        self.assertEqual(invalid, {"1395"})

    def test_all_option_worded_stems_are_known(self) -> None:
        option_worded = {
            row["sample_id"]
            for row in self.rows
            if any(pattern.search(row["question"]) for pattern in OPTION_DEPENDENT_STEM_PATTERNS)
        }
        self.assertEqual(option_worded, {"242", "790", "1344", "1395", "1436", "1498"})

    def test_visual_label_reference_count_is_known(self) -> None:
        labels = {
            row["sample_id"]
            for row in self.rows
            if re.fullmatch(r"[A-Za-z]", str(row["answer"]).strip())
        }
        self.assertEqual(labels, {"621", "799", "926", "1215", "1261", "1350", "1438", "1469", "1470"})


if __name__ == "__main__":
    unittest.main()
