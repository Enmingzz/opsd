#!/usr/bin/env python3
"""Strict MathVerse-VO scoring with parser-first exact match and Qwen fallback.

The saved VLMEvalKit ``prediction`` is already the output of the generation
parser.  This scorer conservatively re-applies closed ``<answer>`` extraction
when a fallback prediction still contains tags, accepts only a trimmed exact
match, and sends every remaining candidate to a constrained Qwen3.6-27B
semantic-equivalence judge.  Canonical questions and answers are reloaded from
the registered TSV so spreadsheet formula conversion cannot corrupt labels.
"""

from __future__ import annotations

import argparse
import ast
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
from pathlib import Path
from typing import Any

import pandas as pd


PROTOCOL_VERSION = "mathverse-vo-strict-gt-v1.0"
DEFAULT_MODEL = "Qwen/Qwen3.6-27B-FP8"
DEFAULT_DATASET_TSV = Path("/scratch/enmingzz/vlmevalkit_data/MathVerse_MINI_Vision_Only.tsv")
DEFAULT_FULL_DATASET_TSV = Path("/scratch/enmingzz/vlmevalkit_data/MathVerse_MINI.tsv")
EXPECTED_DATASET_MD5 = "68a11d4680014ac881fa37adeadea3a4"

SYSTEM_PROMPT = (
    "You are a strict benchmark answer-equivalence judge, not a problem solver. "
    "The candidate is the answer produced by the evaluated model's parser. Decide "
    "whether that candidate explicitly represents the same final answer as the "
    "reference. Use the question and choices only to interpret answer formats, such "
    "as mapping an option value to its letter. Never fill in, repair, complete, or "
    "reinterpret a missing, ambiguous, contradictory, or wrong candidate using the "
    "reference. Equivalent numeric, unit, and algebraic forms are acceptable. For "
    "multiple choice, either the correct option letter or the exact content of that "
    "option is acceptable. A candidate with no explicit final answer, multiple "
    "unresolved alternatives, or an unfinished response is INCORRECT. Output exactly "
    "CORRECT or INCORRECT and nothing else."
)
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
OPTION_LINE_RE = re.compile(r"^\s*([A-H])\s*[:.)]\s*(\S.*)\s*$", re.IGNORECASE)
REQUIRED_SOURCE_COLUMNS = {"index", "prediction"}
REQUIRED_CANONICAL_COLUMNS = {
    "index",
    "problem_index",
    "problem_version",
    "question_type",
    "question_for_eval",
    "answer",
    "metadata",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--dataset-tsv", type=Path, default=DEFAULT_DATASET_TSV)
    parser.add_argument("--full-dataset-tsv", type=Path, default=DEFAULT_FULL_DATASET_TSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--ratio", choices=("noprune", "r010", "r020", "r030"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--expected-rows", type=int, default=788)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
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
        raise ValueError(f"Cannot write an empty CSV: {path}")
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
    return str(value).strip()


def clean_value(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def parser_candidate(row: dict[str, Any]) -> tuple[str, str]:
    """Return the standard saved parser answer, with closed-tag recovery only."""

    candidate = clean_value(row.get("prediction")).strip()
    matches = ANSWER_TAG_RE.findall(candidate)
    if matches:
        candidate = "".join(matches)
        source = "closed_answer_tag_from_saved_prediction"
    else:
        source = "saved_vlmevalkit_prediction"
    return candidate.replace("\n", " ").strip(), source


def is_trimmed_exact_match(candidate: str, reference: str) -> bool:
    """Match VLMEvalKit's conservative MathVerse prefetch semantics."""

    return candidate.strip() == reference.strip()


def extract_choice_block(text: str) -> str:
    """Extract and deduplicate canonical option lines from MathVerse metadata."""

    option_lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for line in clean_value(text).splitlines():
        match = OPTION_LINE_RE.match(line)
        if not match:
            continue
        letter = match.group(1).upper()
        value = match.group(2).strip()
        key = (letter, value)
        if key not in seen:
            seen.add(key)
            option_lines.append(f"{letter}: {value}")
    return "\n".join(option_lines)


def load_tsv(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=lambda name: name in columns,
    )
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def load_canonical_metadata(
    dataset_tsv: Path,
    full_dataset_tsv: Path,
    expected_rows: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    if file_digest(dataset_tsv, "md5") != EXPECTED_DATASET_MD5:
        raise ValueError(f"Unexpected canonical MathVerse-VO dataset MD5: {dataset_tsv}")

    canonical = load_tsv(dataset_tsv, REQUIRED_CANONICAL_COLUMNS)
    if len(canonical) != expected_rows:
        raise ValueError(f"Expected {expected_rows} canonical rows, found {len(canonical)}")
    canonical_ids = [sample_id(value) for value in canonical["index"]]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("Canonical MathVerse-VO TSV has duplicate indices")
    if set(canonical["problem_version"]) != {"Vision Only"}:
        raise ValueError("Canonical dataset is not exclusively Vision Only")

    full_columns = {"problem_index", "question_for_eval", "question_type"}
    full = load_tsv(full_dataset_tsv, full_columns)
    fallback_choices: dict[str, str] = {}
    for row in full.to_dict(orient="records"):
        if clean_value(row["question_type"]) != "multi-choice":
            continue
        block = extract_choice_block(row["question_for_eval"])
        problem = sample_id(row["problem_index"])
        if len(block) > len(fallback_choices.get(problem, "")):
            fallback_choices[problem] = block

    records: dict[str, dict[str, Any]] = {}
    missing_choices: list[str] = []
    fallback_used: list[str] = []
    for row in canonical.to_dict(orient="records"):
        row_id = sample_id(row["index"])
        choices = extract_choice_block(row["question_for_eval"])
        if clean_value(row["question_type"]) == "multi-choice" and not choices:
            choices = fallback_choices.get(sample_id(row["problem_index"]), "")
            if choices:
                fallback_used.append(row_id)
            else:
                missing_choices.append(row_id)
        row["choices_for_judge"] = choices
        row["choices_fallback_used"] = row_id in fallback_used
        records[row_id] = row

    if missing_choices:
        raise ValueError(f"Missing canonical options for MCQ indices: {missing_choices}")

    diagnostics = {
        "dataset_tsv": str(dataset_tsv.resolve()),
        "dataset_md5": file_digest(dataset_tsv, "md5"),
        "dataset_sha256": file_digest(dataset_tsv),
        "full_dataset_tsv": str(full_dataset_tsv.resolve()),
        "full_dataset_sha256": file_digest(full_dataset_tsv),
        "rows": len(canonical),
        "multiple_choice_rows": int((canonical["question_type"] == "multi-choice").sum()),
        "free_form_rows": int((canonical["question_type"] == "free-form").sum()),
        "choice_fallback_indices": fallback_used,
    }
    return records, fallback_choices, diagnostics


def build_judge_user_prompt(canonical: dict[str, Any], candidate: str) -> str:
    choices = canonical.get("choices_for_judge") or "N/A (free-form question)"
    rendered_candidate = candidate if candidate else "<EMPTY>"
    return (
        "QUESTION (canonical evaluation metadata):\n"
        f"{clean_value(canonical['question_for_eval'])}\n\n"
        "CANONICAL ANSWER CHOICES:\n"
        f"{choices}\n\n"
        "REFERENCE ANSWER:\n"
        f"{clean_value(canonical['answer'])}\n\n"
        "CANDIDATE ANSWER EXTRACTED BY THE MODEL PARSER:\n"
        f"{rendered_candidate}"
    )


def parse_verdict(text: str) -> str:
    normalized = text.strip().upper()
    matches = re.findall(r"(?<![A-Z])(?:INCORRECT|CORRECT)(?![A-Z])", normalized)
    if len(set(matches)) != 1:
        raise ValueError(f"Expected one unambiguous verdict, got {text!r}")
    return matches[0]


def build_request_payload(canonical: dict[str, Any], candidate: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(canonical, candidate)},
        ],
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"choice": ["CORRECT", "INCORRECT"]},
    }


def request_verdict(
    canonical: dict[str, Any],
    candidate: str,
    *,
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> tuple[str, str, str, str]:
    user_prompt = build_judge_user_prompt(canonical, candidate)
    payload = build_request_payload(canonical, candidate, model)
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
            return (
                verdict,
                raw_output,
                user_prompt,
                hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
            )
        except Exception as exception:
            error = repr(exception)
            time.sleep(min(4.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Qwen judge failed after {max_retries} attempts: {error}")


def parse_metadata(value: Any) -> dict[str, Any]:
    text = clean_value(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def aggregate_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[("Overall", "Overall")].append(record)
        metadata = parse_metadata(record["metadata"])
        for key in ("subject", "subfield", "source"):
            value = clean_value(metadata.get(key)).strip()
            if value:
                groups[(key, value)].append(record)

    output: list[dict[str, Any]] = []
    for (category, value), rows in groups.items():
        hit = sum(bool(row["canonical_correct"]) for row in rows)
        exact = sum(row["canonical_source"] == "trimmed_exact_match" for row in rows)
        output.append(
            {
                "category": category,
                "value": value,
                "rows": len(rows),
                "hit": hit,
                "accuracy_percent": hit / len(rows) * 100,
                "exact_match_count": exact,
                "judge_fallback_count": len(rows) - exact,
            }
        )
    return output


def read_legacy_score(result_file: Path) -> float | None:
    score_file = result_file.with_name(result_file.stem + "_qwen7judge_score.csv")
    if not score_file.is_file():
        return None
    frame = pd.read_csv(score_file)
    if "Overall" in frame.columns and len(frame):
        return float(frame.iloc[0]["Overall"])
    return None


def legacy_labels(result_file: Path) -> dict[str, bool]:
    workbook = result_file.with_name(result_file.stem + "_qwen7judge_score.xlsx")
    if not workbook.is_file():
        return {}
    frame = pd.read_excel(workbook)
    if "score" not in frame.columns or "index" not in frame.columns:
        return {}
    return {sample_id(row["index"]): bool(row["score"]) for row in frame.to_dict(orient="records")}


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_source(frame: pd.DataFrame, expected_rows: int) -> None:
    missing = REQUIRED_SOURCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Source workbook is missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} source rows, found {len(frame)}")
    ids = [sample_id(value) for value in frame["index"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Source workbook has duplicate MathVerse indices")


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_file = args.result_file.resolve()
    output_dir = args.output_dir.resolve()
    if not result_file.is_file():
        raise FileNotFoundError(result_file)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "judgments.jsonl"
    scored_rows_path = output_dir / "scored_rows.csv"
    score_path = output_dir / "MathVerse_MINI_Vision_Only_strict_gt_score.csv"
    summary_path = output_dir / "summary.json"

    source_sha256 = file_digest(result_file)
    canonical, _, dataset_diagnostics = load_canonical_metadata(
        args.dataset_tsv.resolve(),
        args.full_dataset_tsv.resolve(),
        args.expected_rows,
    )
    canonical_sha256 = dataset_diagnostics["dataset_sha256"]

    if summary_path.is_file() and not args.overwrite and not args.validate_only:
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing_summary.get("status") == "complete"
            and existing_summary.get("source_sha256") == source_sha256
            and existing_summary.get("canonical_dataset_sha256") == canonical_sha256
            and existing_summary.get("protocol_version") == PROTOCOL_VERSION
        ):
            print(json.dumps(existing_summary, ensure_ascii=False, indent=2))
            return existing_summary
        raise RuntimeError(f"Existing output does not match this input/protocol: {summary_path}")

    frame = pd.read_excel(result_file).sort_values(by="index").reset_index(drop=True)
    validate_source(frame, args.expected_rows)
    source_rows = frame.to_dict(orient="records")
    source_ids = [sample_id(row["index"]) for row in source_rows]
    missing_ids = sorted(set(source_ids) - set(canonical), key=int)
    extra_ids = sorted(set(canonical) - set(source_ids), key=int)
    if missing_ids or extra_ids:
        raise ValueError(f"Source/canonical index mismatch: missing={missing_ids}, extra={extra_ids}")

    candidates: dict[str, tuple[str, str]] = {
        sample_id(row["index"]): parser_candidate(row) for row in source_rows
    }
    exact_ids = {
        row_id
        for row_id, (candidate, _) in candidates.items()
        if is_trimmed_exact_match(candidate, clean_value(canonical[row_id]["answer"]))
    }
    validation = {
        "status": "validated",
        "protocol_version": PROTOCOL_VERSION,
        "result_file": str(result_file),
        "source_sha256": source_sha256,
        "canonical_dataset_sha256": canonical_sha256,
        "rows": len(frame),
        "trimmed_exact_matches": len(exact_ids),
        "judge_fallback": len(frame) - len(exact_ids),
        "empty_candidates": sum(not candidate for candidate, _ in candidates.values()),
        "method": args.method,
        "ratio": args.ratio,
        "dataset": dataset_diagnostics,
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
            if (
                record.get("source_sha256") != source_sha256
                or record.get("canonical_dataset_sha256") != canonical_sha256
                or record.get("protocol_version") != PROTOCOL_VERSION
                or record.get("judge_system_prompt_sha256") != SYSTEM_PROMPT_SHA256
            ):
                raise RuntimeError(f"Stale judgment found in {judgments_path}")
            existing[str(record["sample_id"])] = record

    records: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for row_id in source_ids:
        candidate, candidate_source = candidates[row_id]
        common = {
            "sample_id": row_id,
            "method": args.method,
            "ratio": args.ratio,
            "source_sha256": source_sha256,
            "canonical_dataset_sha256": canonical_sha256,
            "protocol_version": PROTOCOL_VERSION,
            "judge_system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "question_type": canonical[row_id]["question_type"],
            "question_for_eval": canonical[row_id]["question_for_eval"],
            "choices_for_judge": canonical[row_id]["choices_for_judge"],
            "choices_fallback_used": canonical[row_id]["choices_fallback_used"],
            "canonical_answer": canonical[row_id]["answer"],
            "metadata": canonical[row_id]["metadata"],
            "parser_candidate": candidate,
            "parser_candidate_source": candidate_source,
            "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        }
        if row_id in exact_ids:
            records[row_id] = {
                **common,
                "canonical_source": "trimmed_exact_match",
                "canonical_correct": True,
                "judge_model": None,
                "judge_raw_output": None,
                "judge_user_prompt": None,
                "judge_prompt_sha256": None,
            }
        elif row_id in existing:
            records[row_id] = existing[row_id]
        else:
            pending.append(row_id)

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
                    canonical[row_id],
                    candidates[row_id][0],
                    base_url=args.base_url,
                    model=args.judge_model,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                ): row_id
                for row_id in pending
            }
            completed = 0
            for future in as_completed(futures):
                row_id = futures[future]
                verdict, raw_output, user_prompt, user_prompt_sha256 = future.result()
                candidate, candidate_source = candidates[row_id]
                records[row_id] = {
                    "sample_id": row_id,
                    "method": args.method,
                    "ratio": args.ratio,
                    "source_sha256": source_sha256,
                    "canonical_dataset_sha256": canonical_sha256,
                    "protocol_version": PROTOCOL_VERSION,
                    "judge_system_prompt_sha256": SYSTEM_PROMPT_SHA256,
                    "question_type": canonical[row_id]["question_type"],
                    "question_for_eval": canonical[row_id]["question_for_eval"],
                    "choices_for_judge": canonical[row_id]["choices_for_judge"],
                    "choices_fallback_used": canonical[row_id]["choices_fallback_used"],
                    "canonical_answer": canonical[row_id]["answer"],
                    "metadata": canonical[row_id]["metadata"],
                    "parser_candidate": candidate,
                    "parser_candidate_source": candidate_source,
                    "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
                    "canonical_source": "qwen36_strict_gt_judge",
                    "canonical_correct": verdict == "CORRECT",
                    "judge_model": args.judge_model,
                    "judge_raw_output": raw_output,
                    "judge_user_prompt": user_prompt,
                    "judge_prompt_sha256": user_prompt_sha256,
                }
                completed += 1
                if completed % 20 == 0:
                    ordered_partial = [records[key] for key in source_ids if key in records]
                    atomic_jsonl(judgments_path, ordered_partial)
                    print(f"[judge] completed={completed}/{len(pending)}", flush=True)

    if len(records) != len(frame):
        raise RuntimeError(f"Expected {len(frame)} decisions, found {len(records)}")
    ordered_records = [records[row_id] for row_id in source_ids]
    atomic_jsonl(judgments_path, ordered_records)

    scored_rows = [
        {
            "sample_id": record["sample_id"],
            "method": record["method"],
            "ratio": record["ratio"],
            "question_type": record["question_type"],
            "canonical_answer": record["canonical_answer"],
            "parser_candidate": record["parser_candidate"],
            "exact_match": record["canonical_source"] == "trimmed_exact_match",
            "decision_source": record["canonical_source"],
            "correct": record["canonical_correct"],
            "judge_raw_output": record["judge_raw_output"],
            "choices_fallback_used": record["choices_fallback_used"],
        }
        for record in ordered_records
    ]
    atomic_csv(scored_rows_path, scored_rows)
    score_rows = aggregate_scores(ordered_records)
    atomic_csv(score_path, score_rows)
    overall = next(row for row in score_rows if row["category"] == "Overall")

    old_labels = legacy_labels(result_file)
    old_to_new = None
    if old_labels and set(old_labels) == set(source_ids):
        old_to_new = {
            "old_correct_to_new_wrong": sum(
                old_labels[row_id] and not records[row_id]["canonical_correct"] for row_id in source_ids
            ),
            "old_wrong_to_new_correct": sum(
                not old_labels[row_id] and records[row_id]["canonical_correct"] for row_id in source_ids
            ),
        }

    root = Path(__file__).resolve().parents[2]
    summary = {
        **validation,
        "status": "complete",
        "judge_model": args.judge_model,
        "judge_output_constraint": {"choice": ["CORRECT", "INCORRECT"]},
        "judge_system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "question_visible_to_judge": True,
        "choices_visible_to_judge": True,
        "ground_truth_visible_to_judge": True,
        "image_visible_to_judge": False,
        "judge_correct_fallback": sum(
            record["canonical_correct"]
            for record in ordered_records
            if record["canonical_source"] == "qwen36_strict_gt_judge"
        ),
        "judge_incorrect_fallback": sum(
            not record["canonical_correct"]
            for record in ordered_records
            if record["canonical_source"] == "qwen36_strict_gt_judge"
        ),
        "hit": int(overall["hit"]),
        "score_percent": float(overall["accuracy_percent"]),
        "old_qwen7_score_percent": read_legacy_score(result_file),
        "old_to_new": old_to_new,
        "score_file": str(score_path),
        "scored_rows": str(scored_rows_path),
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
