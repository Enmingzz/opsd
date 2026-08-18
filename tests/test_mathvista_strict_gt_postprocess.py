from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval/postprocess_mathvista_strict_gt.py"
SPEC = importlib.util.spec_from_file_location("mathvista_strict_gt", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def base_row(**updates):
    row = {
        "index": 1,
        "question": "How many objects are shown?",
        "prediction": "0",
        "question_type": "free_form",
        "answer_type": "integer",
        "answer": "0",
        "answer_option": None,
        "choices": "[]",
        "task": "visual question answering",
        "skills": "['counting']",
    }
    row.update(updates)
    return row


class MathVistaStrictGTTest(unittest.TestCase):
    def test_correct_zero_is_resolved_without_judge(self):
        decision = MODULE.deterministic_parse(base_row())
        self.assertTrue(decision.resolved)
        self.assertTrue(decision.correct)
        self.assertEqual(decision.extracted_answer, "0")

    def test_plain_wrong_scalar_is_resolved_and_not_repaired(self):
        decision = MODULE.deterministic_parse(base_row(prediction="2"))
        self.assertTrue(decision.resolved)
        self.assertFalse(decision.correct)

    def test_range_requires_strict_judge(self):
        decision = MODULE.deterministic_parse(base_row(prediction="50 to 70", answer="50"))
        self.assertFalse(decision.resolved)
        self.assertIsNone(decision.correct)

    def test_multiple_choice_uses_pinned_option_parser(self):
        row = base_row(
            question_type="multi_choice",
            answer_type="text",
            answer="4.29",
            answer_option="B",
            choices="['3.71', '4.29', '4.53', '6.75']",
            prediction="The final answer is B.",
        )
        decision = MODULE.deterministic_parse(row)
        self.assertTrue(decision.resolved)
        self.assertTrue(decision.correct)
        self.assertEqual(decision.extracted_answer, "B")

    def test_judge_prompt_contains_question_gt_and_candidate(self):
        row = base_row(question="Question text", answer="7", prediction="candidate text")
        prompt = MODULE.build_judge_user_prompt(row)
        self.assertIn("Question text", prompt)
        self.assertIn("REFERENCE ANSWER: 7", prompt)
        self.assertIn("candidate text", prompt)

    def test_verdict_parser_is_strict(self):
        self.assertEqual(MODULE.parse_verdict("CORRECT"), "CORRECT")
        self.assertEqual(MODULE.parse_verdict("INCORRECT"), "INCORRECT")
        with self.assertRaises(ValueError):
            MODULE.parse_verdict("CORRECT or INCORRECT")

    def test_request_uses_constrained_binary_verdict(self):
        payload = MODULE.build_request_payload(base_row(), "judge-model")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(
            payload["structured_outputs"],
            {"choice": ["CORRECT", "INCORRECT"]},
        )

    def test_prompt_forbids_reference_based_repair(self):
        self.assertIn("never use it to fill in, repair", MODULE.SYSTEM_PROMPT)
        self.assertIn("truncated or unfinished", MODULE.SYSTEM_PROMPT)
        self.assertIn("range or multiple alternatives", MODULE.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
