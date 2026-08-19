#!/usr/bin/env python3
"""Build an ordered 20K LLaVA-CoT dataset from an existing base and new parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


SPACE_RE = re.compile(r"\s+")
IMAGE_TAG_RE = re.compile(r"<image(?:\s+\d+)?>", re.IGNORECASE)
HINT_RE = re.compile(r"^\s*hint:.*?\bquestion:\s*", re.IGNORECASE | re.DOTALL)
TARGET_RE = re.compile(
    r"^\s*<think>\s*(?P<reasoning>.*?)\s*</think>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
TRAILING_INSTRUCTIONS = (
    "answer with the option's letter from the given choices directly.",
    "please answer yes or no.",
    "please directly answer the question and provide the correct option letter, e.g., a, b, c, d.",
    "please answer the question and provide the final answer at the end.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument(
        "--continuation-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Ordered continuation part; may be repeated.",
    )
    parser.add_argument("--holdout-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-audit", type=Path)
    parser.add_argument("--processor-check", type=Path)
    parser.add_argument(
        "--exclude-continuation-sample-id",
        action="append",
        default=[],
        help="Explicit continuation sample ID to omit; may be repeated and is recorded in the manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-base-rows", type=int, default=10_000)
    parser.add_argument("--expected-continuation-rows", type=int, default=10_000)
    parser.add_argument("--max-qwen-target-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-image-tokens", type=int, default=1152)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(value: Any, *, relaxed: bool = False) -> str:
    text = str(value or "").strip().lower()
    if relaxed:
        text = IMAGE_TAG_RE.sub(" ", text)
        text = HINT_RE.sub("", text)
        for suffix in TRAILING_INSTRUCTIONS:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
    return SPACE_RE.sub(" ", text).strip()


def read_jsonl_exact(path: Path) -> tuple[list[bytes], list[dict[str, Any]]]:
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"JSONL must end with a newline: {path}")
    lines = [line for line in payload.splitlines(keepends=True) if line.strip()]
    rows = [json.loads(line) for line in lines]
    return lines, rows


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    max_target_tokens: int,
    max_image_tokens: int,
) -> dict[str, Any]:
    sample_ids: set[str] = set()
    source_ids: set[tuple[str, str]] = set()
    strict_questions: set[str] = set()
    relaxed_questions: set[str] = set()
    image_paths: set[str] = set()
    source_counts: Counter[str] = Counter()
    target_lengths: list[int] = []
    image_lengths: list[int] = []

    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row.get("id", ""))).strip()
        source = str(row.get("source", "")).strip()
        source_id_value = str(row.get("source_id", "")).strip()
        source_id = (source, source_id_value)
        question = normalize_question(row.get("question", ""))
        relaxed_question = normalize_question(row.get("question", ""), relaxed=True)
        image = str(row.get("image", "")).strip()
        target = str(row.get("answer", row.get("ground_truth", "")) or "")
        match = TARGET_RE.match(target)
        target_tokens = int(row.get("qwen_target_token_len", -1))
        image_tokens = int(row.get("qwen_prompt_image_tokens", -1))

        if not sample_id or not source or not source_id_value or not question or not image:
            raise ValueError(f"{label}[{index}] has an empty required identity field")
        if not match or not match.group("reasoning").strip() or not match.group("answer").strip():
            raise ValueError(f"{label}[{index}] has an invalid <think>/<answer> target")
        if not 0 <= target_tokens <= max_target_tokens:
            raise ValueError(f"{label}[{index}] target token count is {target_tokens}")
        if not 0 <= image_tokens <= max_image_tokens:
            raise ValueError(f"{label}[{index}] image token count is {image_tokens}")
        if sample_id in sample_ids:
            raise ValueError(f"{label} contains duplicate sample_id={sample_id}")
        if source_id in source_ids:
            raise ValueError(f"{label} contains duplicate source identity={source_id}")

        sample_ids.add(sample_id)
        source_ids.add(source_id)
        strict_questions.add(question)
        relaxed_questions.add(relaxed_question)
        image_paths.add(image)
        source_counts[source] += 1
        target_lengths.append(target_tokens)
        image_lengths.append(image_tokens)

    def lengths(values: list[int]) -> dict[str, float | int]:
        return {
            "min": min(values),
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "max": max(values),
        }

    return {
        "rows": len(rows),
        "sample_ids": sample_ids,
        "source_ids": source_ids,
        "strict_questions": strict_questions,
        "relaxed_questions": relaxed_questions,
        "image_paths": image_paths,
        "source_counts": dict(sorted(source_counts.items())),
        "qwen_target_token_len": lengths(target_lengths),
        "qwen_prompt_image_tokens": lengths(image_lengths),
    }


def overlap(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    return {
        "sample_id": len(left["sample_ids"] & right["sample_ids"]),
        "source_id": len(left["source_ids"] & right["source_ids"]),
        "normalized_question": len(left["strict_questions"] & right["strict_questions"]),
        "relaxed_question": len(left["relaxed_questions"] & right["relaxed_questions"]),
        "image_path": len(left["image_paths"] & right["image_paths"]),
    }


def public_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if key not in {"sample_ids", "source_ids", "strict_questions", "relaxed_questions", "image_paths"}
    }


def compact(value: Any, limit: int = 160) -> str:
    text = SPACE_RE.sub(" ", str(value or "")).strip().replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> int:
    args = parse_args()
    paths = [args.base_jsonl, *args.continuation_jsonl, args.holdout_jsonl]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_lines, base_rows = read_jsonl_exact(args.base_jsonl)
    raw_continuation_parts = [read_jsonl_exact(path) for path in args.continuation_jsonl]
    excluded_continuation_ids = set(args.exclude_continuation_sample_id)
    found_excluded_ids: set[str] = set()
    continuation_parts: list[tuple[list[bytes], list[dict[str, Any]]]] = []
    for lines, rows in raw_continuation_parts:
        kept_lines: list[bytes] = []
        kept_rows: list[dict[str, Any]] = []
        for line, row in zip(lines, rows):
            sample_id = str(row.get("sample_id", row.get("id", "")))
            if sample_id in excluded_continuation_ids:
                found_excluded_ids.add(sample_id)
                continue
            kept_lines.append(line)
            kept_rows.append(row)
        continuation_parts.append((kept_lines, kept_rows))
    missing_exclusions = excluded_continuation_ids - found_excluded_ids
    if missing_exclusions:
        raise ValueError(f"continuation exclusions were not found: {sorted(missing_exclusions)}")
    continuation_lines = [line for lines, _ in continuation_parts for line in lines]
    continuation_rows = [row for _, rows in continuation_parts for row in rows]
    _, holdout_rows = read_jsonl_exact(args.holdout_jsonl)

    if len(base_rows) != args.expected_base_rows:
        raise ValueError(f"expected {args.expected_base_rows} base rows, got {len(base_rows)}")
    if len(continuation_rows) != args.expected_continuation_rows:
        raise ValueError(
            f"expected {args.expected_continuation_rows} continuation rows, got {len(continuation_rows)}"
        )

    validate_kwargs = {
        "max_target_tokens": args.max_qwen_target_tokens,
        "max_image_tokens": args.max_prompt_image_tokens,
    }
    base_stats = validate_rows(base_rows, label="base", **validate_kwargs)
    continuation_stats = validate_rows(continuation_rows, label="continuation", **validate_kwargs)
    holdout_stats = validate_rows(holdout_rows, label="holdout", **validate_kwargs)
    combined_stats = validate_rows(base_rows + continuation_rows, label="combined", **validate_kwargs)

    overlap_audit = {
        "base_vs_continuation": overlap(base_stats, continuation_stats),
        "base_vs_holdout": overlap(base_stats, holdout_stats),
        "continuation_vs_holdout": overlap(continuation_stats, holdout_stats),
    }
    if any(value for item in overlap_audit.values() for value in item.values()):
        raise AssertionError(f"identity overlap detected: {overlap_audit}")

    continuation_name = "additional10k_decontam_qwentok512_imgtok1152_seed42.jsonl"
    combined_name = "train20k_old10k_then_additional10k_qwentok512_imgtok1152_seed42.jsonl"
    continuation_path = args.output_dir / continuation_name
    combined_path = args.output_dir / combined_name
    continuation_payload = b"".join(continuation_lines)
    combined_payload = b"".join(base_lines + continuation_lines)
    atomic_write(continuation_path, continuation_payload)
    atomic_write(combined_path, combined_payload)

    base_payload = b"".join(base_lines)
    if combined_payload[: len(base_payload)] != base_payload:
        raise AssertionError("combined dataset does not preserve the base segment byte-for-byte")
    if combined_payload[len(base_payload) :] != continuation_payload:
        raise AssertionError("combined dataset does not preserve the continuation segment byte-for-byte")

    external_checks: dict[str, Any] = {}
    if args.benchmark_audit is not None:
        audit = json.loads(args.benchmark_audit.read_text(encoding="utf-8"))
        audit_train_sets = audit.get("train_sets", [])
        if len(audit_train_sets) != 1:
            raise ValueError("benchmark audit must describe exactly one training artifact")
        if audit_train_sets[0].get("file_sha256") != sha256_file(combined_path):
            raise ValueError("benchmark audit hash does not match the ordered 20K")
        match_keys = (
            "strict_question_match_rows",
            "relaxed_question_match_rows",
            "encoded_image_match_rows",
            "pixel_image_match_rows",
            "perceptual_image_match_rows",
            "image_and_question_match_rows",
            "image_payload_errors",
        )
        totals = {
            key: sum(int(benchmark.get(key, 0)) for benchmark in audit.get("benchmarks", []))
            for key in match_keys
        }
        if any(totals.values()) or int(audit_train_sets[0].get("missing_image_count", -1)) != 0:
            raise ValueError(f"benchmark audit is not clean: {totals}")
        external_checks["benchmark_audit"] = {
            "status": "passed",
            "path": str(args.benchmark_audit.resolve()),
            "sha256": sha256_file(args.benchmark_audit),
            "benchmark_count": len(audit.get("benchmarks", [])),
            "missing_training_images": 0,
            **totals,
        }
    if args.processor_check is not None:
        processor_check = json.loads(args.processor_check.read_text(encoding="utf-8"))
        if processor_check.get("status") != "passed" or int(processor_check.get("mismatch_count", -1)) != 0:
            raise ValueError("Qwen processor check did not pass")
        if Path(processor_check.get("dataset", "")).resolve() != combined_path.resolve():
            raise ValueError("Qwen processor check references a different dataset")
        external_checks["qwen_processor_check"] = {
            "status": "passed",
            "path": str(args.processor_check.resolve()),
            "sha256": sha256_file(args.processor_check),
            "checked_samples": int(processor_check["checked_samples"]),
            "mismatch_count": 0,
            "max_recorded_image_tokens": int(processor_check["max_recorded_image_tokens"]),
            "max_actual_image_tokens": int(processor_check["max_actual_image_tokens"]),
        }

    manifest = {
        "schema_version": "lcot_ordered_20k_v1",
        "status": "passed",
        "ordering": "existing approved 10K first, disjoint additional 10K second",
        "continuation_start_zero_based": len(base_rows),
        "continuation_start_one_based": len(base_rows) + 1,
        "base": {
            "path": str(args.base_jsonl.resolve()),
            "sha256": sha256_file(args.base_jsonl),
            **public_stats(base_stats),
        },
        "continuation_parts": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "input_rows": len(raw_rows),
                "selected_rows": len(rows),
            }
            for path, (_, raw_rows), (_, rows) in zip(
                args.continuation_jsonl, raw_continuation_parts, continuation_parts
            )
        ],
        "continuation_excluded_sample_ids": sorted(excluded_continuation_ids),
        "holdout": {
            "path": str(args.holdout_jsonl.resolve()),
            "sha256": sha256_file(args.holdout_jsonl),
            "rows": len(holdout_rows),
        },
        "additional10k": {
            "path": str(continuation_path.resolve()),
            "sha256": sha256_file(continuation_path),
            **public_stats(continuation_stats),
        },
        "ordered20k": {
            "path": str(combined_path.resolve()),
            "sha256": sha256_file(combined_path),
            **public_stats(combined_stats),
        },
        "overlap_audit": overlap_audit,
        "caps": {
            "max_qwen_target_tokens": args.max_qwen_target_tokens,
            "max_prompt_image_tokens": args.max_prompt_image_tokens,
        },
        "byte_preservation": {
            "base_segment_matches_input": True,
            "continuation_segment_matches_output": True,
            "base_segment_sha256": sha256_bytes(base_payload),
            "continuation_segment_sha256": sha256_bytes(continuation_payload),
        },
        "external_checks": external_checks,
        "training_guidance": {
            "continue_only": str(continuation_path.resolve()),
            "full_ordered_curriculum": str(combined_path.resolve()),
            "note": "Use additional10k directly to avoid replaying the approved first 10K.",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())

    boundary_indices = [0, len(base_rows) - 1, len(base_rows), len(base_rows) + 1, len(base_rows + continuation_rows) - 1]
    all_rows = base_rows + continuation_rows
    preview = [
        "# Ordered LLaVA-CoT 20K Boundary Preview",
        "",
        "| 1-based row | segment | sample_id | source | question | target |",
        "|---:|---|---|---|---|---|",
    ]
    for index in boundary_indices:
        row = all_rows[index]
        segment = "approved_old10k" if index < len(base_rows) else "additional10k"
        preview.append(
            f"| {index + 1} | {segment} | {compact(row.get('sample_id'))} | "
            f"{compact(row.get('source'))} | {compact(row.get('question'))} | "
            f"{compact(row.get('answer', row.get('ground_truth', '')), 240)} |"
        )
    preview_path = args.output_dir / "boundary_preview.md"
    atomic_write(preview_path, ("\n".join(preview) + "\n").encode())

    readme = f"""# Ordered LLaVA-CoT 20K

This artifact preserves the approved historical 10K as rows 1-10,000 and appends a
strictly disjoint additional 10K as rows 10,001-20,000.

## Training files

- Continue without replaying old data: `{continuation_name}`
- Full ordered dataset: `{combined_name}`
- Continuation begins at one-based row `10,001` in the ordered dataset.

Use the standalone additional 10K file for a continuation run. This avoids relying on
sampler offset semantics and guarantees that the approved historical 10K is not replayed.

## Checks

- Valid `<think>...</think><answer>...</answer>` target for every row.
- Qwen target length <= {args.max_qwen_target_tokens} tokens.
- Recorded image-prompt length <= {args.max_prompt_image_tokens} tokens.
- No sample/source/question/image-path identity overlap between old and new splits.
- No identity overlap with the strict 1K validation holdout.
- The old 10K prefix and new 10K suffix are preserved byte-for-byte.
{f"- Independent {external_checks['benchmark_audit']['benchmark_count']}-benchmark leakage audit passed with zero matches." if 'benchmark_audit' in external_checks else ''}
{f"- Qwen processor check passed on {external_checks['qwen_processor_check']['checked_samples']} samples with zero token-count mismatches." if 'qwen_processor_check' in external_checks else ''}

See `manifest.json` for hashes and `boundary_preview.md` for the ordering boundary.
"""
    atomic_write(args.output_dir / "README.md", readme.encode())

    print(
        json.dumps(
            {
                "status": "passed",
                "additional10k": str(continuation_path.resolve()),
                "additional10k_sha256": manifest["additional10k"]["sha256"],
                "ordered20k": str(combined_path.resolve()),
                "ordered20k_sha256": manifest["ordered20k"]["sha256"],
                "continuation_start_one_based": len(base_rows) + 1,
                "overlap_audit": overlap_audit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
