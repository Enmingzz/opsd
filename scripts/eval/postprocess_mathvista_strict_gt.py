#!/usr/bin/env python3
"""Canonical MathVista post-processing with a strict GT-aware fallback judge.

The deterministic MathVista parser is always attempted first. Only responses
that cannot be resolved by that parser are sent to a Qwen3.6-27B judge. The
judge sees the question, reference answer, and candidate response, but is
explicitly forbidden from repairing or completing the candidate.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROTOCOL_VERSION = "mathvista-strict-gt-v1.0"
DEFAULT_MODEL = "Qwen/Qwen3.6-27B-FP8"
SYSTEM_PROMPT = (
    "You are a strict benchmark correctness judge, not a problem solver. "
    "Decide whether the candidate response itself explicitly commits to an "
    "answer semantically equivalent to the reference answer. Use the reference "
    "only for comparison; never use it to fill in, repair, or reinterpret "
    "missing or wrong content. Mark INCORRECT if the response is truncated or "
    "unfinished, lacks an explicit answer, gives a range or multiple alternatives "
    "for a single-valued question, contradicts itself without a clear final "
    "resolution, refuses to answer, or merely contains the reference somewhere "
    "without selecting it. Units and equivalent numeric formatting may be "
    "normalized. For multiple choice, an explicitly selected correct option "
    "letter or its exact answer value is acceptable. Output exactly CORRECT or "
    "INCORRECT and nothing else."
)
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
REQUIRED_COLUMNS = {
    "index",
    "question",
    "prediction",
    "question_type",
    "answer_type",
    "answer",
    "answer_option",
    "choices",
    "task",
    "skills",
}


@dataclass(frozen=True)
class ParserDecision:
    resolved: bool
    extracted_answer: str | None
    correct: bool | None
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--ratio", choices=("noprune", "r010", "r020", "r030"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--expected-rows", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sample_id(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def clean_value(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def parse_choices(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        choices = value
    else:
        choices = ast.literal_eval(str(value))
    if not isinstance(choices, list):
        raise ValueError(f"MathVista choices must be a list, got {type(choices).__name__}")
    return {chr(65 + index): choice for index, choice in enumerate(choices)}


def can_infer_option(answer: str, choices: dict[str, Any]) -> str | bool:
    """Pinned VLMEvalKit option matching without package import side effects."""

    if "Failed to obtain answer via API" in answer:
        return False
    reject = (
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    )
    if any(message in answer for message in reject):
        return "Z"

    answer_mod = copy.copy(answer)
    for character in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(character, " ")
    parts = [value.strip() for value in answer_mod.split()]
    count = sum(key in parts for key in choices)
    if count == 1:
        for key in choices:
            if key in parts and parts.index(key) > len(parts) - 5:
                return key
    elif count == 0 and sum(key in parts for key in {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer: str, choices: dict[str, Any]) -> str | bool:
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(value)) for value in choices.values()):
        return False
    normalized = {key: str(value).lower() for key, value in choices.items()}
    candidates = [key for key, value in normalized.items() if value in answer]
    return candidates[0] if len(candidates) == 1 else False


def can_infer(answer: str, choices: dict[str, Any]) -> str | bool:
    option = can_infer_option(answer, choices)
    return option if option else can_infer_text(answer, choices)


def deterministic_parse(row: dict[str, Any]) -> ParserDecision:
    prediction = clean_value(row["prediction"]).strip()
    question_type = clean_value(row["question_type"])
    answer_type = clean_value(row["answer_type"])
    answer = clean_value(row["answer"])

    if question_type == "multi_choice":
        inferred = can_infer(prediction, parse_choices(row["choices"]))
        if inferred is False:
            return ParserDecision(False, None, None, "official_mcq_parser_unresolved")
        expected = clean_value(row["answer_option"]).strip().upper()
        extracted = str(inferred).strip().upper()
        return ParserDecision(True, extracted, extracted == expected, "official_mcq_parser_resolved")

    try:
        if answer_type == "integer":
            extracted_int = int(prediction)
            return ParserDecision(
                True,
                str(extracted_int),
                extracted_int == int(answer),
                "official_integer_parser_resolved",
            )
        if answer_type == "float":
            extracted_float = float(prediction)
            return ParserDecision(
                True,
                repr(extracted_float),
                extracted_float == float(answer),
                "official_float_parser_resolved",
            )
    except (OverflowError, TypeError, ValueError):
        return ParserDecision(False, None, None, f"official_{answer_type}_parser_unresolved")

    return ParserDecision(False, None, None, f"official_{answer_type or 'unknown'}_parser_unresolved")


def build_judge_user_prompt(row: dict[str, Any]) -> str:
    option = clean_value(row["answer_option"]).strip() or "N/A"
    return (
        f"QUESTION:\n{clean_value(row['question'])}\n\n"
        f"REFERENCE ANSWER: {clean_value(row['answer'])}\n"
        f"REFERENCE OPTION LETTER (if applicable): {option}\n\n"
        "CANDIDATE RESPONSE:\n"
        f"{clean_value(row['prediction'])}"
    )


def parse_verdict(text: str) -> str:
    normalized = text.strip().upper()
    matches = re.findall(r"(?<![A-Z])(?:INCORRECT|CORRECT)(?![A-Z])", normalized)
    if len(set(matches)) != 1:
        raise ValueError(f"Expected one unambiguous verdict, got {text!r}")
    return matches[0]


def build_request_payload(row: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(row)},
        ],
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"choice": ["CORRECT", "INCORRECT"]},
    }


def request_verdict(
    row: dict[str, Any],
    *,
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> tuple[str, str, str]:
    user_prompt = build_judge_user_prompt(row)
    payload = build_request_payload(row, model)
    endpoint = base_url.rstrip("/") + "/chat/completions"
    error = ""
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            raw_output = str(body["choices"][0]["message"]["content"])
            verdict = parse_verdict(raw_output)
            return verdict, raw_output, hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        except Exception as exception:  # retries are recorded in the final failure
            error = repr(exception)
            time.sleep(min(4.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Qwen judge failed after {max_retries} attempts: {error}")


def reference_result(row: dict[str, Any]) -> str:
    if clean_value(row["question_type"]) == "multi_choice":
        return clean_value(row["answer_option"]).strip().upper()
    return clean_value(row["answer"])


def parse_skills(value: Any) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = [clean_value(value)]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def aggregate_score(frame: pd.DataFrame) -> list[dict[str, Any]]:
    total: defaultdict[str, int] = defaultdict(int)
    parser_resolved: defaultdict[str, int] = defaultdict(int)
    judge_fallback: defaultdict[str, int] = defaultdict(int)
    hits: defaultdict[str, int] = defaultdict(int)

    for row in frame.to_dict(orient="records"):
        labels = ["Overall", clean_value(row["task"]), *parse_skills(row["skills"])]
        for label in labels:
            total[label] += 1
            parser_resolved[label] += int(row["canonical_source"] == "official_parser")
            judge_fallback[label] += int(row["canonical_source"] == "qwen36_strict_gt_judge")
            hits[label] += int(bool(row["canonical_correct"]))

    output = []
    for label, count in total.items():
        output.append(
            {
                "Task&Skill": label,
                "tot": count,
                "official_parser_resolved": parser_resolved[label],
                "judge_fallback": judge_fallback[label],
                "hit": hits[label],
                "official_parser_resolved_rate": parser_resolved[label] / count * 100,
                "acc": hits[label] / count * 100,
            }
        )
    return output


def legacy_correct(row: dict[str, Any]) -> bool:
    try:
        response = clean_value(row["res"])
        if clean_value(row["question_type"]) == "multi_choice":
            inferred = can_infer(response, parse_choices(row["choices"]))
            return inferred == clean_value(row["answer_option"]).strip().upper()
        if clean_value(row["answer_type"]) == "integer":
            return int(response) == int(row["answer"])
        if clean_value(row["answer_type"]) == "float":
            return float(response) == float(row["answer"])
    except (OverflowError, TypeError, ValueError):
        return False
    return False


def read_old_score(result_file: Path) -> float | None:
    score_file = result_file.with_name(result_file.stem + "_qwen7judge_score.csv")
    if not score_file.is_file():
        return None
    rows = list(csv.DictReader(score_file.open(encoding="utf-8")))
    for row in rows:
        if row.get("Task&Skill") == "Overall":
            return float(row["acc"])
    return None


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_frame(frame: pd.DataFrame, expected_rows: int) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input workbook is missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    ids = [sample_id(value) for value in frame["index"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Input workbook has duplicate MathVista indices")


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_file = args.result_file.resolve()
    output_dir = args.output_dir.resolve()
    if not result_file.is_file():
        raise FileNotFoundError(result_file)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "judgments.jsonl"
    workbook_path = output_dir / "MathVista_MINI_strict_gt_judge.xlsx"
    score_path = output_dir / "MathVista_MINI_strict_gt_judge_score.csv"
    summary_path = output_dir / "summary.json"
    source_sha256 = sha256_file(result_file)

    if summary_path.is_file() and not args.overwrite and not args.validate_only:
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing_summary.get("status") == "complete"
            and existing_summary.get("source_sha256") == source_sha256
            and existing_summary.get("protocol_version") == PROTOCOL_VERSION
        ):
            print(json.dumps(existing_summary, ensure_ascii=False, indent=2))
            return existing_summary
        raise RuntimeError(f"Existing output does not match this input/protocol: {summary_path}")

    frame = pd.read_excel(result_file).sort_values(by="index").reset_index(drop=True)
    validate_frame(frame, args.expected_rows)
    source_rows = frame.to_dict(orient="records")
    parser_decisions = {sample_id(row["index"]): deterministic_parse(row) for row in source_rows}
    fallback_count = sum(not decision.resolved for decision in parser_decisions.values())

    validation = {
        "status": "validated",
        "protocol_version": PROTOCOL_VERSION,
        "result_file": str(result_file),
        "source_sha256": source_sha256,
        "rows": len(frame),
        "official_parser_resolved": len(frame) - fallback_count,
        "judge_fallback": fallback_count,
        "method": args.method,
        "ratio": args.ratio,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return validation

    existing: dict[str, dict[str, Any]] = {}
    if judgments_path.is_file() and not args.overwrite:
        for line in judgments_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source_sha256") != source_sha256:
                raise RuntimeError(f"Stale judgment found in {judgments_path}")
            existing[str(record["sample_id"])] = record

    records: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for row in source_rows:
        row_id = sample_id(row["index"])
        decision = parser_decisions[row_id]
        if decision.resolved:
            records[row_id] = {
                "sample_id": row_id,
                "source_sha256": source_sha256,
                "protocol_version": PROTOCOL_VERSION,
                "canonical_source": "official_parser",
                "canonical_correct": bool(decision.correct),
                "extracted_answer": decision.extracted_answer,
                "parser_reason": decision.reason,
                "judge_model": None,
                "judge_raw_output": None,
                "judge_prompt_sha256": None,
                "candidate_sha256": hashlib.sha256(
                    clean_value(row["prediction"]).encode("utf-8")
                ).hexdigest(),
            }
        elif row_id in existing:
            records[row_id] = existing[row_id]
        else:
            pending.append(row)

    if pending:
        print(
            f"[judge] method={args.method} ratio={args.ratio} "
            f"pending={len(pending)} concurrency={args.concurrency}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    request_verdict,
                    row,
                    base_url=args.base_url,
                    model=args.judge_model,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                ): row
                for row in pending
            }
            completed = 0
            for future in as_completed(futures):
                row = futures[future]
                row_id = sample_id(row["index"])
                verdict, raw_output, user_prompt_sha256 = future.result()
                records[row_id] = {
                    "sample_id": row_id,
                    "source_sha256": source_sha256,
                    "protocol_version": PROTOCOL_VERSION,
                    "canonical_source": "qwen36_strict_gt_judge",
                    "canonical_correct": verdict == "CORRECT",
                    "extracted_answer": None,
                    "parser_reason": parser_decisions[row_id].reason,
                    "judge_model": args.judge_model,
                    "judge_raw_output": raw_output,
                    "judge_prompt_sha256": user_prompt_sha256,
                    "candidate_sha256": hashlib.sha256(
                        clean_value(row["prediction"]).encode("utf-8")
                    ).hexdigest(),
                }
                completed += 1
                if completed % 20 == 0:
                    atomic_jsonl(judgments_path, [records[key] for key in sorted(records, key=int)])
                    print(f"[judge] completed={completed}/{len(pending)}", flush=True)

    if len(records) != len(frame):
        raise RuntimeError(f"Expected {len(frame)} decisions, found {len(records)}")
    ordered_records = [records[sample_id(row["index"])] for row in source_rows]
    atomic_jsonl(judgments_path, ordered_records)

    output_frame = frame.copy()
    output_frame["canonical_protocol_version"] = PROTOCOL_VERSION
    output_frame["canonical_source"] = [record["canonical_source"] for record in ordered_records]
    output_frame["canonical_correct"] = [record["canonical_correct"] for record in ordered_records]
    output_frame["canonical_extracted_answer"] = [record["extracted_answer"] for record in ordered_records]
    output_frame["canonical_parser_reason"] = [record["parser_reason"] for record in ordered_records]
    output_frame["strict_gt_judge_verdict"] = [
        ("CORRECT" if record["canonical_correct"] else "INCORRECT")
        if record["canonical_source"] == "qwen36_strict_gt_judge"
        else "NOT_CALLED"
        for record in ordered_records
    ]
    output_frame["strict_gt_judge_model"] = [record["judge_model"] for record in ordered_records]
    output_frame["strict_gt_judge_raw_output"] = [record["judge_raw_output"] for record in ordered_records]
    output_frame["res"] = [
        reference_result(row) if record["canonical_correct"] else "__STRICT_JUDGE_INCORRECT__"
        for row, record in zip(source_rows, ordered_records)
    ]
    output_frame["log"] = [
        "Prefetch succeed"
        if record["canonical_source"] == "official_parser"
        else f"{args.judge_model} strict GT judge"
        for record in ordered_records
    ]

    temporary_workbook = workbook_path.with_name(f".{workbook_path.name}.tmp.{os.getpid()}.xlsx")
    output_frame.to_excel(temporary_workbook, index=False)
    temporary_workbook.replace(workbook_path)
    score_rows = aggregate_score(output_frame)
    atomic_csv(score_path, score_rows)
    overall = next(row for row in score_rows if row["Task&Skill"] == "Overall")

    legacy_workbook = result_file.with_name(result_file.stem + "_qwen7judge.xlsx")
    old_to_new = None
    if legacy_workbook.is_file():
        legacy = pd.read_excel(legacy_workbook).sort_values(by="index").reset_index(drop=True)
        if len(legacy) == len(output_frame):
            old_hits = [legacy_correct(row) for row in legacy.to_dict(orient="records")]
            new_hits = [bool(value) for value in output_frame["canonical_correct"]]
            old_to_new = {
                "old_correct_to_new_wrong": sum(old and not new for old, new in zip(old_hits, new_hits)),
                "old_wrong_to_new_correct": sum(not old and new for old, new in zip(old_hits, new_hits)),
            }

    root = Path(__file__).resolve().parents[2]
    summary = {
        **validation,
        "status": "complete",
        "judge_model": args.judge_model,
        "judge_output_constraint": {"choice": ["CORRECT", "INCORRECT"]},
        "judge_system_prompt_sha256": PROMPT_SHA256,
        "question_visible_to_judge": True,
        "ground_truth_visible_to_judge": True,
        "image_visible_to_judge": False,
        "strict_correct_fallback": sum(
            record["canonical_correct"]
            for record in ordered_records
            if record["canonical_source"] == "qwen36_strict_gt_judge"
        ),
        "strict_incorrect_fallback": sum(
            not record["canonical_correct"]
            for record in ordered_records
            if record["canonical_source"] == "qwen36_strict_gt_judge"
        ),
        "hit": int(overall["hit"]),
        "score_percent": float(overall["acc"]),
        "old_qwen7_score_percent": read_old_score(result_file),
        "old_to_new": old_to_new,
        "workbook": str(workbook_path),
        "score_file": str(score_path),
        "judgments": str(judgments_path),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": git_commit(root),
        "command": [sys.executable, *sys.argv],
        "completed_at_unix": time.time(),
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
