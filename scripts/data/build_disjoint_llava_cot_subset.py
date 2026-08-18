#!/usr/bin/env python3
"""Select a deterministic LLaVA-CoT continuation subset disjoint from prior splits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SPACE_RE = re.compile(r"\s+")
IMAGE_TAG_RE = re.compile(r"<image(?:\s+\d+)?>", re.IGNORECASE)
HINT_RE = re.compile(r"^\s*hint:.*?\bquestion:\s*", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(
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
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        action="append",
        required=True,
        help="NAME=JSONL. May be repeated; every identity in every split is excluded.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/scratch/enmingzz/openmmreasoner_llava_cot_image_root"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--stem", default="additional5k_decontam_qwentok512_imgtok1152_seed42")
    parser.add_argument("--max-qwen-target-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-image-tokens", type=int, default=1152)
    parser.add_argument("--preview-n", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=JSONL, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise ValueError(f"empty exclusion name in {value!r}")
    return name.strip(), Path(raw_path).expanduser().resolve()


def normalize_question(value: Any, *, relaxed: bool = False) -> str:
    text = str(value or "").strip().lower()
    if relaxed:
        text = IMAGE_TAG_RE.sub(" ", text)
        text = HINT_RE.sub("", text)
        for suffix in TRAILING_INSTRUCTIONS:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
    return SPACE_RE.sub(" ", text).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image(row: dict[str, Any], image_root: Path) -> Path:
    path = Path(str(row.get("image", "")))
    if not path.is_absolute():
        path = image_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def image_fingerprints(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    encoded = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
        digest.update(rgb.tobytes())
    return encoded, digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def describe(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    percentile = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {
        "count": len(ordered),
        "mean": float(statistics.mean(ordered)),
        "median": float(statistics.median(ordered)),
        "min": int(ordered[0]),
        "p90": int(percentile(0.90)),
        "p95": int(percentile(0.95)),
        "p99": int(percentile(0.99)),
        "max": int(ordered[-1]),
    }


def main() -> int:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = args.candidate_jsonl.expanduser().resolve()
    candidate_manifest_path = args.candidate_manifest.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    if not candidate_path.is_file() or not candidate_manifest_path.is_file():
        raise FileNotFoundError("candidate JSONL or manifest is missing")
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("status") != "passed":
        raise RuntimeError("candidate manifest is not marked passed")
    if candidate_manifest.get("output_sha256") != sha256_file(candidate_path):
        raise RuntimeError("candidate JSONL hash does not match its manifest")

    exclusions = [parse_named_path(value) for value in args.exclude]
    if len({name for name, _ in exclusions}) != len(exclusions):
        raise ValueError("exclusion names must be unique")
    for _, path in exclusions:
        if not path.is_file():
            raise FileNotFoundError(path)

    image_cache: dict[Path, tuple[str, str]] = {}

    def fingerprints(row: dict[str, Any]) -> tuple[str, str, str]:
        path = resolve_image(row, image_root)
        if path not in image_cache:
            image_cache[path] = image_fingerprints(path)
        encoded, pixel = image_cache[path]
        return str(path), encoded, pixel

    excluded_sample_ids: set[str] = set()
    excluded_source_ids: set[tuple[str, str]] = set()
    excluded_questions: set[str] = set()
    excluded_relaxed_questions: set[str] = set()
    excluded_image_paths: set[str] = set()
    excluded_encoded_images: set[str] = set()
    excluded_pixel_images: set[str] = set()
    exclusion_docs: list[dict[str, Any]] = []
    for name, path in exclusions:
        rows = read_jsonl(path)
        for row in rows:
            excluded_sample_ids.add(str(row.get("sample_id", row.get("id", ""))))
            excluded_source_ids.add((str(row.get("source", "")), str(row.get("source_id", ""))))
            excluded_questions.add(normalize_question(row.get("question", "")))
            excluded_relaxed_questions.add(normalize_question(row.get("question", ""), relaxed=True))
            image_path, encoded, pixel = fingerprints(row)
            excluded_image_paths.add(image_path)
            excluded_encoded_images.add(encoded)
            excluded_pixel_images.add(pixel)
        exclusion_docs.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        )

    selected: list[dict[str, Any]] = []
    selected_sample_ids: set[str] = set()
    selected_source_ids: set[tuple[str, str]] = set()
    selected_image_questions: set[tuple[str, str]] = set()
    rejection_counts: Counter[str] = Counter()
    candidates_examined = 0
    for row in read_jsonl(candidate_path):
        candidates_examined += 1
        sample_id = str(row.get("sample_id", row.get("id", "")))
        source_id = (str(row.get("source", "")), str(row.get("source_id", "")))
        question = normalize_question(row.get("question", ""))
        relaxed_question = normalize_question(row.get("question", ""), relaxed=True)
        target = str(row.get("answer", row.get("ground_truth", "")) or "")
        tag_match = TAG_RE.match(target)
        reason = None
        if not tag_match or not tag_match.group("reasoning").strip() or not tag_match.group("answer").strip():
            reason = "invalid_reasoning_answer_tags"
        elif int(row.get("qwen_target_token_len", args.max_qwen_target_tokens + 1)) > args.max_qwen_target_tokens:
            reason = "target_token_len_gt_cap"
        elif int(row.get("qwen_prompt_image_tokens", args.max_prompt_image_tokens + 1)) > args.max_prompt_image_tokens:
            reason = "prompt_image_tokens_gt_cap"
        elif bool(row.get("is_ocr_heavy")) or bool(row.get("is_chart_heavy")) or bool(row.get("is_document_heavy")):
            reason = "ocr_chart_document_metadata"
        elif sample_id in excluded_sample_ids:
            reason = "excluded_sample_id"
        elif source_id in excluded_source_ids:
            reason = "excluded_source_id"
        elif question in excluded_questions:
            reason = "excluded_normalized_question"
        elif relaxed_question in excluded_relaxed_questions:
            reason = "excluded_relaxed_question"

        image_path = encoded = pixel = ""
        if reason is None:
            try:
                image_path, encoded, pixel = fingerprints(row)
            except Exception:
                reason = "missing_or_invalid_image"
        if reason is None and image_path in excluded_image_paths:
            reason = "excluded_image_path"
        elif reason is None and encoded in excluded_encoded_images:
            reason = "excluded_image_encoded_sha256"
        elif reason is None and pixel in excluded_pixel_images:
            reason = "excluded_image_pixel_sha256"
        elif reason is None and sample_id in selected_sample_ids:
            reason = "duplicate_sample_id"
        elif reason is None and source_id in selected_source_ids:
            reason = "duplicate_source_id"
        elif reason is None and (pixel, question) in selected_image_questions:
            reason = "duplicate_image_question"
        if reason is not None:
            rejection_counts[reason] += 1
            continue

        selected.append(dict(row))
        selected_sample_ids.add(sample_id)
        selected_source_ids.add(source_id)
        selected_image_questions.add((pixel, question))
        if len(selected) == args.n:
            break

    if len(selected) != args.n:
        raise RuntimeError(
            f"selected only {len(selected)} rows after examining {candidates_examined}; "
            "build a larger deterministic candidate pool"
        )

    output_path = args.output_dir / f"{args.stem}.jsonl"
    manifest_path = args.output_dir / f"{args.stem}_stats.json"
    preview_path = args.output_dir / f"{args.stem}_preview{args.preview_n}.md"
    write_jsonl(output_path, selected)

    # Recompute identities from the final artifact so the manifest is not based only on selection state.
    output_rows = read_jsonl(output_path)
    overlap_audit: dict[str, dict[str, int]] = {}
    output_sample_ids = {str(row.get("sample_id", row.get("id", ""))) for row in output_rows}
    output_source_ids = {(str(row.get("source", "")), str(row.get("source_id", ""))) for row in output_rows}
    output_questions = {normalize_question(row.get("question", "")) for row in output_rows}
    output_relaxed = {normalize_question(row.get("question", ""), relaxed=True) for row in output_rows}
    output_paths: set[str] = set()
    output_encoded: set[str] = set()
    output_pixel: set[str] = set()
    for row in output_rows:
        image_path, encoded, pixel = fingerprints(row)
        output_paths.add(image_path)
        output_encoded.add(encoded)
        output_pixel.add(pixel)
    for name, path in exclusions:
        rows = read_jsonl(path)
        sample_ids = {str(row.get("sample_id", row.get("id", ""))) for row in rows}
        source_ids = {(str(row.get("source", "")), str(row.get("source_id", ""))) for row in rows}
        questions = {normalize_question(row.get("question", "")) for row in rows}
        relaxed = {normalize_question(row.get("question", ""), relaxed=True) for row in rows}
        paths: set[str] = set()
        encoded_images: set[str] = set()
        pixel_images: set[str] = set()
        for row in rows:
            image_path, encoded, pixel = fingerprints(row)
            paths.add(image_path)
            encoded_images.add(encoded)
            pixel_images.add(pixel)
        overlap_audit[name] = {
            "sample_id": len(output_sample_ids & sample_ids),
            "source_id": len(output_source_ids & source_ids),
            "normalized_question": len(output_questions & questions),
            "relaxed_question": len(output_relaxed & relaxed),
            "image_path": len(output_paths & paths),
            "image_encoded_sha256": len(output_encoded & encoded_images),
            "image_pixel_sha256": len(output_pixel & pixel_images),
        }
    if any(value for audit in overlap_audit.values() for value in audit.values()):
        raise AssertionError(f"exclusion overlap survived selection: {overlap_audit}")

    candidate_caps = candidate_manifest.get("caps", {})
    manifest = {
        "schema_version": "lcot_additional_disjoint_subset_v1",
        "status": "passed",
        "selection_rule": (
            "candidate priority inherited from the original seed-42 decontamination builder; "
            "select first rows strictly disjoint from all named exclusions"
        ),
        "candidate_jsonl": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "output_jsonl": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "selected_rows": len(output_rows),
        "unique_sample_ids": len(output_sample_ids),
        "unique_source_ids": len(output_source_ids),
        "unique_image_question_pairs": len(selected_image_questions),
        "candidates_examined": candidates_examined,
        "exclusions": exclusion_docs,
        "strict_exclusions": [
            "sample_id",
            "source+source_id",
            "normalized_question",
            "relaxed_question",
            "resolved_image_path",
            "encoded_image_sha256",
            "decoded_pixel_sha256",
        ],
        "overlap_audit": overlap_audit,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "source_counts": dict(sorted(Counter(str(row.get("source", "")) for row in output_rows).items())),
        "qwen_target_token_len": describe([int(row["qwen_target_token_len"]) for row in output_rows]),
        "qwen_prompt_image_tokens": describe([int(row["qwen_prompt_image_tokens"]) for row in output_rows]),
        "caps": {
            "max_qwen_target_tokens": args.max_qwen_target_tokens,
            "max_prompt_image_tokens": args.max_prompt_image_tokens,
            "min_pixels": int(candidate_caps.get("min_pixels", 1080 * 28 * 28)),
            "max_pixels": int(candidate_caps.get("max_pixels", 1080 * 28 * 28)),
        },
        "benchmark_decontamination": candidate_manifest.get("benchmark_decontamination", {}),
        "source_dataset": candidate_manifest.get("source_dataset"),
        "source_dataset_revision": candidate_manifest.get("source_dataset_revision"),
        "image_root": str(image_root),
        "image_files_hashed": len(image_cache),
    }
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    def truncate(value: Any, limit: int = 300) -> str:
        text = str(value or "").replace("\n", "<br>").replace("|", "\\|")
        return text if len(text) <= limit else text[: limit - 3] + "..."

    preview = [
        "# Additional Disjoint LLaVA-CoT Preview",
        "",
        "| # | sample_id | source | image tokens | target tokens | image | question | target |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    for index, row in enumerate(output_rows[: args.preview_n], 1):
        preview.append(
            f"| {index} | {truncate(row.get('sample_id'))} | {truncate(row.get('source'))} | "
            f"{row.get('qwen_prompt_image_tokens')} | {row.get('qwen_target_token_len')} | "
            f"{truncate(row.get('image'), 160)} | {truncate(row.get('question'))} | "
            f"{truncate(row.get('answer'))} |"
        )
    atomic_text(preview_path, "\n".join(preview) + "\n")
    atomic_text(
        args.output_dir / "README.md",
        "\n".join(
            (
                "# Additional LLaVA-CoT 5K, Disjoint",
                "",
                "This continuation subset uses the same deterministic seed-42 filtering and benchmark",
                "decontamination rules as the approved 10K. It excludes both the historical 10K training",
                "artifact and the strict 1K validation holdout by sample, source, question and image identity.",
                "",
                f"- JSONL: `{output_path.name}`",
                f"- SHA-256: `{manifest['output_sha256']}`",
                f"- Rows: `{len(output_rows)}`",
                f"- Manifest: `{manifest_path.name}`",
                f"- Preview: `{preview_path.name}`",
                "",
                "Use this 5K by itself when continuing from a checkpoint trained on the prior 10K.",
                "Do not concatenate and replay the old 10K unless the experiment explicitly calls for it.",
            )
        )
        + "\n",
    )
    print(json.dumps({
        "status": "passed",
        "output_jsonl": str(output_path.resolve()),
        "output_sha256": manifest["output_sha256"],
        "selected_rows": len(output_rows),
        "candidates_examined": candidates_examined,
        "overlap_audit": overlap_audit,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
