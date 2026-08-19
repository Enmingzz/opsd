from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval/postprocess_mathverse_strict_gt.py"
SPEC = importlib.util.spec_from_file_location("mathverse_strict_gt", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MathVerseStrictGTTest(unittest.TestCase):
    def test_parser_uses_saved_prediction(self):
        candidate, source = MODULE.parser_candidate({"prediction": "  D  "})
        self.assertEqual(candidate, "D")
        self.assertEqual(source, "saved_vlmevalkit_prediction")

    def test_parser_recovers_closed_answer_span(self):
        candidate, source = MODULE.parser_candidate(
            {"prediction": "<think>work</think>\n<answer> 64.5 </answer>"}
        )
        self.assertEqual(candidate, "64.5")
        self.assertEqual(source, "closed_answer_tag_from_saved_prediction")

    def test_parser_preserves_unclosed_fallback(self):
        raw = "<think>unfinished reasoning <answer>maybe 4"
        candidate, source = MODULE.parser_candidate({"prediction": raw})
        self.assertEqual(candidate, raw)
        self.assertEqual(source, "saved_vlmevalkit_prediction")

    def test_exact_match_is_conservative(self):
        self.assertTrue(MODULE.is_trimmed_exact_match(" D ", "D"))
        self.assertFalse(MODULE.is_trimmed_exact_match("d", "D"))
        self.assertFalse(MODULE.is_trimmed_exact_match("140°", "D"))

    def test_choice_extraction_accepts_official_variants_and_deduplicates(self):
        text = "Question\nChoice:\nA. 1\nB:2\nA. 1\nC) 3"
        self.assertEqual(MODULE.extract_choice_block(text), "A: 1\nB: 2\nC: 3")

    def test_judge_prompt_contains_question_choices_gt_and_candidate(self):
        row = {
            "question_for_eval": "Question text",
            "choices_for_judge": "A: 3\nB: 4",
            "answer": "B",
        }
        prompt = MODULE.build_judge_user_prompt(row, "4")
        self.assertIn("Question text", prompt)
        self.assertIn("A: 3", prompt)
        self.assertIn("REFERENCE ANSWER:\nB", prompt)
        self.assertIn("MODEL PARSER:\n4", prompt)

    def test_request_is_deterministic_and_binary_constrained(self):
        row = {
            "question_for_eval": "Question text",
            "choices_for_judge": "A: 3\nB: 4",
            "answer": "B",
        }
        payload = MODULE.build_request_payload(row, "4", "judge")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["structured_outputs"], {"choice": ["CORRECT", "INCORRECT"]})
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])

    def test_prompt_forbids_reference_based_repair(self):
        self.assertIn("Never fill in, repair, complete", MODULE.SYSTEM_PROMPT)
        self.assertIn("multiple unresolved alternatives", MODULE.SYSTEM_PROMPT)

    def test_verdict_parser_rejects_ambiguous_output(self):
        self.assertEqual(MODULE.parse_verdict("CORRECT"), "CORRECT")
        self.assertEqual(MODULE.parse_verdict("INCORRECT"), "INCORRECT")
        with self.assertRaises(ValueError):
            MODULE.parse_verdict("CORRECT or INCORRECT")


if __name__ == "__main__":
    unittest.main()
