from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


OPTION_LETTERS = ("A", "B", "C", "D")


PROMPT_TEMPLATE = """<image>

You are given an image and a multiple-choice visual reasoning question.

Question:
{question}

Options:
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}

Answer with brief reasoning, then provide the final option.
Your response must follow exactly this format:

Reasoning: ...
Final answer: ..."""


TARGET_TEMPLATE = """Reasoning: {brief_reasoning}
Final answer: {correct_option_letter}"""


OPSD_TEACHER_PROMPT_TEMPLATE = """<image>

You are given an image and a multiple-choice visual reasoning question.

Question:
{question}

Options:
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step - do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and do not be afraid to backtrack or reconsider if something does not work out.

Answer with brief reasoning, then provide the final option.
Your response must follow exactly this format:

Reasoning: ...
Final answer: ..."""


@dataclass
class FormattedAOKVQASample:
    sample_id: str
    image: Any
    question: str
    options: list[str]
    correct_index: int
    correct_letter: str
    reasoning: str
    prompt: str
    target: str
    raw: dict[str, Any]


def option_index_to_letter(index: int) -> str:
    index = int(index)
    if index < 0 or index >= len(OPTION_LETTERS):
        raise ValueError(f"A-OKVQA correct option index must be in [0, 3], got {index}.")
    return OPTION_LETTERS[index]


def strip_image_tokens(text: str) -> str:
    return re.sub(r"<\s*image\s*>", "", str(text), flags=re.IGNORECASE).strip()


def build_reasoning_prompt(question: str, options: list[str]) -> str:
    if len(options) != 4:
        raise ValueError(f"A-OKVQA prompt requires exactly four options, got {len(options)}.")
    clean_options = [str(option).strip() for option in options]
    return PROMPT_TEMPLATE.format(
        question=strip_image_tokens(question),
        option_0=clean_options[0],
        option_1=clean_options[1],
        option_2=clean_options[2],
        option_3=clean_options[3],
    )


def build_target(reasoning: str | None, correct_option_letter: str) -> str:
    brief_reasoning = str(reasoning or "").strip()
    if not brief_reasoning:
        brief_reasoning = "The correct option is supported by the image and question."
    return TARGET_TEMPLATE.format(
        brief_reasoning=brief_reasoning,
        correct_option_letter=str(correct_option_letter).strip().upper(),
    )


def build_opsd_teacher_prompt(question: str, options: list[str], reference_solution: str) -> str:
    """Build the privileged teacher prompt used by the official OPSD setup."""

    if len(options) != 4:
        raise ValueError(f"A-OKVQA teacher prompt requires exactly four options, got {len(options)}.")
    clean_options = [str(option).strip() for option in options]
    reference = str(reference_solution or "").strip()
    if not reference:
        reference = "Reasoning: The correct option is supported by the image and question."
    return OPSD_TEACHER_PROMPT_TEMPLATE.format(
        question=strip_image_tokens(question),
        option_0=clean_options[0],
        option_1=clean_options[1],
        option_2=clean_options[2],
        option_3=clean_options[3],
        reference_solution=reference,
    )


FINAL_ANSWER_PATTERNS = (
    re.compile(r"final\s+answer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"the\s+final\s+answer\s+is\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
)


def parse_final_answer(text: str) -> str | None:
    """Extract A/B/C/D from the requested final-answer format and close variants."""

    value = str(text or "")
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1).upper()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines:
        tail = lines[-1]
        match = re.search(r"\b([ABCD])\b", tail, flags=re.IGNORECASE)
        if match and "final" in tail.lower():
            return match.group(1).upper()
    return None


def has_parseable_final_answer(text: str) -> bool:
    return parse_final_answer(text) is not None


def format_chat_messages(prompt: str) -> list[dict[str, Any]]:
    """Build Qwen-style multimodal chat messages for a prompt containing <image>."""

    text = strip_image_tokens(prompt)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def format_chat_with_assistant(prompt: str, assistant_response: str) -> list[dict[str, Any]]:
    messages = format_chat_messages(prompt)
    messages.append({"role": "assistant", "content": str(assistant_response)})
    return messages
