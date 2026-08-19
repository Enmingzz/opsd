#!/usr/bin/env python3
"""Build a deterministic, benchmark-decontaminated LLaVA-CoT subset.

The source JSONL is never modified.  The output is selected only after removing
exact/relaxed question matches and exact/near-duplicate benchmark images.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import heapq
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_BENCHMARKS = (
    "MME=MME.tsv",
    "MME_CoT_TEST=MME_CoT_TEST.tsv",
    "MMStar=MMStar.tsv",
    "MathVista_MINI=MathVista_MINI.tsv",
    "MathVerse_MINI=MathVerse_MINI.tsv",
    "MathVerse_MINI_Vision_Only=MathVerse_MINI_Vision_Only.tsv",
    "MathVision_MINI=MathVision_MINI.tsv",
    "MathVision_FULL=MathVision.tsv",
    "POPE=POPE.tsv",
    "POPE_local=POPE_local.tsv",
    "MMMU_Pro_4c=MMMU_Pro_4c.tsv",
    "MMMU_Pro_10c_COT=MMMU_Pro_10c_COT.tsv",
    "MMMU_Pro_V_COT=MMMU_Pro_V_COT.tsv",
    "CV-Bench-2D=CV-Bench-2D.tsv",
    "CV-Bench-3D=CV-Bench-3D.tsv",
)

SPACE_RE = re.compile(r"\s+")
IMAGE_TAG_RE = re.compile(r"<image(?:\s+\d+)?>", re.IGNORECASE)
HINT_RE = re.compile(r"^\s*hint:.*?\bquestion:\s*", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(
    r"^\s*<think>\s*(?P<reasoning>.*?)\s*</think>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
TRAILING_EVAL_INSTRUCTIONS = (
    "answer with the option's letter from the given choices directly.",
    "please answer yes or no.",
    "please directly answer the question and provide the correct option letter, e.g., a, b, c, d.",
    "please answer the question and provide the final answer at the end.",
)
CONFLICTING_TRAIN_INSTRUCTIONS = (
    "answer the question using a single word or phrase",
    "answer with the option's letter from the given choices directly",
    "answer with the option letter from the given choices directly",
    "respond with only the answer",
    "provide only the final answer",
)
OCR_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwhat does (?:the|this|that|a|an)?\s*[^?]{0,80}\b(?:say|read)\b",
        r"\bwhat (?:is|was) written\b",
        r"\bread (?:the|this|that)\s+(?:text|sign|label|document|receipt|invoice|form)\b",
        r"\btranscribe\b",
        r"\btext (?:on|in) (?:the|this|that)\b",
        r"\bwhich words? (?:are|is|appear|appears)\b",
        r"\b(?:receipt|invoice|document|spreadsheet|form)\b",
        r"\baccording to (?:the|this) (?:chart|graph|table)\b",
        r"\b(?:bar|line|pie) chart\b",
        r"\b(?:rightmost|leftmost) bar\b",
        r"\bvalue of the bar\b",
    )
)
SOURCE_PATH_EXCLUDES = (
    "docvqa",
    "textvqa",
    "ocr",
    "stvqa",
    "infovqa",
    "receipt",
    "invoice",
    "document",
    "chartqa",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        default=Path(
            "/project/6101803/enmingzz/data/openmmreasoner_sft_874k/llava_cot/processed_opsd/"
            "train_main_no_ocr_chart_qwentok512_opsd_format.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/openmmreasoner_llava_cot_train10k_decontam_v1_seed42"),
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=Path("/scratch/enmingzz/openmmreasoner_llava_cot_image_root"),
    )
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("/scratch/enmingzz/vlmevalkit_data"),
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        help="NAME=TSV; repeat to override the default benchmark snapshot set.",
    )
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_dataset", default="OpenMMReasoner/OpenMMReasoner-SFT-874K:llava_cot")
    parser.add_argument("--source_dataset_revision", default="c44cf46e41bd24351ae7e7771a44fcd2dbf75f0a")
    parser.add_argument("--candidate_pool_size", type=int, default=50_000)
    parser.add_argument("--max_qwen_target_tokens", type=int, default=512)
    parser.add_argument("--max_prompt_image_tokens", type=int, default=1152)
    parser.add_argument("--min_pixels", type=int, default=1080 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1080 * 28 * 28)
    parser.add_argument("--perceptual_hamming_threshold", type=int, default=2)
    parser.add_argument("--perceptual_aspect_tolerance", type=float, default=0.02)
    parser.add_argument("--max_question_marks", type=int, default=1)
    parser.add_argument("--max_input_rows", type=int, default=0, help="0 scans the full source JSONL.")
    parser.add_argument("--preview_n", type=int, default=20)
    parser.add_argument("--progress_every", type=int, default=25_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_question(text: Any, relaxed: bool = False) -> str:
    value = str(text or "").strip().lower()
    if relaxed:
        value = IMAGE_TAG_RE.sub(" ", value)
        value = HINT_RE.sub("", value)
        for suffix in TRAILING_EVAL_INSTRUCTIONS:
            if value.endswith(suffix):
                value = value[: -len(suffix)]
    return SPACE_RE.sub(" ", value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_priority(seed: int, identity: str) -> int:
    payload = f"{seed}\0{identity}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"image aspect ratio exceeds 200: {width}x{height}")
    out_h = max(factor, round(height / factor) * factor)
    out_w = max(factor, round(width / factor) * factor)
    if out_h * out_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        out_h = max(factor, math.floor(height / beta / factor) * factor)
        out_w = max(factor, math.floor(width / beta / factor) * factor)
    elif out_h * out_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        out_h = math.ceil(height * beta / factor) * factor
        out_w = math.ceil(width * beta / factor) * factor
    return int(out_h), int(out_w)


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


def image_fingerprints(data: bytes) -> dict[str, Any]:
    encoded = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
        digest.update(rgb.tobytes())
        return {
            "encoded_sha256": encoded,
            "pixel_sha256": digest.hexdigest(),
            "dhash64": dhash64(rgb),
            "width": rgb.width,
            "height": rgb.height,
            "aspect": rgb.width / rgb.height,
        }


def parse_named_path(value: str, root: Path) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, raw = value.split("=", 1)
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return name, path


def resolve_image_paths(raw_value: str, benchmark_path: Path, benchmark_name: str) -> Iterable[Path]:
    if not raw_value or raw_value.lower() == "nan":
        return
    try:
        parsed = ast.literal_eval(raw_value)
        values = parsed if isinstance(parsed, list) else [parsed]
    except (SyntaxError, ValueError):
        values = [raw_value]
    for value in values:
        path = Path(str(value))
        candidates = [path]
        if not path.is_absolute():
            candidates.extend(
                (
                    benchmark_path.parent / path,
                    benchmark_path.parent / "images" / benchmark_name / path,
                )
            )
        for candidate in candidates:
            if candidate.is_file():
                yield candidate
                break


def benchmark_image_payloads(
    row: dict[str, str], benchmark_path: Path, benchmark_name: str
) -> Iterable[tuple[str, bytes]]:
    encoded = str(row.get("image", "") or "")
    if encoded and encoded.lower() != "nan":
        if len(encoded) > 128:
            try:
                payload = base64.b64decode(encoded, validate=False)
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                yield "embedded", payload
            except Exception:
                for path in resolve_image_paths(encoded, benchmark_path, benchmark_name):
                    yield str(path), path.read_bytes()
        else:
            for path in resolve_image_paths(encoded, benchmark_path, benchmark_name):
                yield str(path), path.read_bytes()
    for path in resolve_image_paths(str(row.get("image_path", "") or ""), benchmark_path, benchmark_name):
        yield str(path), path.read_bytes()


def _dhash_buckets(value: int) -> Iterable[tuple[int, int]]:
    for index in range(4):
        yield index, (value >> (index * 16)) & 0xFFFF


class BenchmarkRegistry:
    def __init__(self) -> None:
        self.strict_questions: set[str] = set()
        self.relaxed_questions: set[str] = set()
        self.encoded: dict[str, set[str]] = defaultdict(set)
        self.pixel: dict[str, set[str]] = defaultdict(set)
        self.dhash_records: list[dict[str, Any]] = []
        self.dhash_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.benchmark_stats: list[dict[str, Any]] = []

    def add_image(self, fp: dict[str, Any], benchmark: str) -> None:
        self.encoded[fp["encoded_sha256"]].add(benchmark)
        self.pixel[fp["pixel_sha256"]].add(benchmark)
        record_index = len(self.dhash_records)
        record = {**fp, "benchmarks": [benchmark]}
        self.dhash_records.append(record)
        for bucket in _dhash_buckets(int(fp["dhash64"])):
            self.dhash_buckets[bucket].append(record_index)

    def image_match(
        self, fp: dict[str, Any], hamming_threshold: int, aspect_tolerance: float
    ) -> tuple[str | None, list[str], int | None]:
        encoded_hits = self.encoded.get(fp["encoded_sha256"])
        if encoded_hits:
            return "encoded_sha256", sorted(encoded_hits), 0
        pixel_hits = self.pixel.get(fp["pixel_sha256"])
        if pixel_hits:
            return "pixel_sha256", sorted(pixel_hits), 0
        candidates: set[int] = set()
        for bucket in _dhash_buckets(int(fp["dhash64"])):
            candidates.update(self.dhash_buckets.get(bucket, ()))
        for index in candidates:
            other = self.dhash_records[index]
            if abs(float(fp["aspect"]) - float(other["aspect"])) > aspect_tolerance:
                continue
            distance = (int(fp["dhash64"]) ^ int(other["dhash64"])).bit_count()
            if distance <= hamming_threshold:
                return "dhash_near_duplicate", list(other["benchmarks"]), distance
        return None, [], None


def build_benchmark_registry(named_paths: list[tuple[str, Path]]) -> BenchmarkRegistry:
    registry = BenchmarkRegistry()
    csv.field_size_limit(2**31 - 1)
    seen_payloads: set[str] = set()
    for name, path in named_paths:
        rows = images = image_errors = 0
        strict_before = len(registry.strict_questions)
        relaxed_before = len(registry.relaxed_questions)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows += 1
                question = row.get("question", row.get("problem", row.get("query", "")))
                strict = normalize_question(question)
                relaxed = normalize_question(question, relaxed=True)
                if strict:
                    registry.strict_questions.add(strict)
                if relaxed:
                    registry.relaxed_questions.add(relaxed)
                for _, payload in benchmark_image_payloads(row, path, name):
                    payload_key = hashlib.sha256(payload).hexdigest()
                    if payload_key in seen_payloads:
                        registry.encoded[payload_key].add(name)
                        continue
                    seen_payloads.add(payload_key)
                    try:
                        fp = image_fingerprints(payload)
                    except Exception:
                        image_errors += 1
                        continue
                    registry.add_image(fp, name)
                    images += 1
        registry.benchmark_stats.append(
            {
                "benchmark": name,
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "rows": rows,
                "new_strict_questions": len(registry.strict_questions) - strict_before,
                "new_relaxed_questions": len(registry.relaxed_questions) - relaxed_before,
                "new_unique_images": images,
                "image_errors": image_errors,
            }
        )
        print(f"benchmark={name} rows={rows} new_images={images} image_errors={image_errors}", flush=True)
    return registry


def row_prefilter_reason(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    question = str(row.get("question", "") or "").strip()
    target = str(row.get("answer", row.get("ground_truth", "")) or "").strip()
    match = TAG_RE.match(target)
    if not match:
        return "invalid_reasoning_answer_tags"
    if not match.group("reasoning").strip():
        return "empty_reasoning"
    if not match.group("answer").strip():
        return "empty_answer"
    target_tokens = row.get("qwen_target_token_len")
    if target_tokens is None:
        return "missing_qwen_target_token_len"
    if int(target_tokens) > args.max_qwen_target_tokens:
        return "target_token_len_gt_cap"
    source_path = f"{row.get('source', '')} {row.get('image', '')}".lower()
    if any(term in source_path for term in SOURCE_PATH_EXCLUDES):
        return "ocr_chart_document_source"
    normalized = normalize_question(question)
    if any(term in normalized for term in CONFLICTING_TRAIN_INSTRUCTIONS):
        return "conflicting_direct_answer_instruction"
    if question.count("?") > args.max_question_marks:
        return "composite_multi_question"
    if any(pattern.search(question) for pattern in OCR_PATTERNS):
        return "ocr_chart_document_question"
    return None


def collect_candidate_pool(args: argparse.Namespace, registry: BenchmarkRegistry) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    heap: list[tuple[int, int, dict[str, Any]]] = []
    rejection_counts: Counter[str] = Counter()
    rejection_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_source_ids: set[tuple[str, str]] = set()
    input_rows = 0
    eligible_before_priority = 0
    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            input_rows += 1
            if args.max_input_rows and input_rows > args.max_input_rows:
                input_rows -= 1
                break
            row = json.loads(line)
            reason = row_prefilter_reason(row, args)
            if reason is None:
                strict = normalize_question(row.get("question", ""))
                relaxed = normalize_question(row.get("question", ""), relaxed=True)
                if strict in registry.strict_questions:
                    reason = "benchmark_strict_question_match"
                elif relaxed in registry.relaxed_questions:
                    reason = "benchmark_relaxed_question_match"
            source_key = (str(row.get("source", "")), str(row.get("source_id", "")))
            if reason is None and source_key[1] and source_key in seen_source_ids:
                reason = "duplicate_source_id"
            if reason:
                rejection_counts[reason] += 1
                if len(rejection_examples[reason]) < 20:
                    rejection_examples[reason].append(
                        {
                            "line_number": line_number,
                            "sample_id": row.get("sample_id"),
                            "source": row.get("source"),
                            "source_id": row.get("source_id"),
                            "image": row.get("image"),
                            "question": row.get("question"),
                        }
                    )
                continue
            if source_key[1]:
                seen_source_ids.add(source_key)
            eligible_before_priority += 1
            identity = f"{row.get('sample_id', '')}\0{row.get('source_id', '')}\0{line_number}"
            priority = deterministic_priority(args.seed, identity)
            item = (-priority, line_number, row)
            if len(heap) < args.candidate_pool_size:
                heapq.heappush(heap, item)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, item)
            if args.progress_every and input_rows % args.progress_every == 0:
                print(
                    f"source_progress rows={input_rows} eligible={eligible_before_priority} "
                    f"candidate_pool={len(heap)}",
                    flush=True,
                )
    candidates = [(-negative_priority, line_number, row) for negative_priority, line_number, row in heap]
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in candidates], {
        "input_rows": input_rows,
        "eligible_before_random_priority": eligible_before_priority,
        "candidate_pool_rows": len(candidates),
        "prefilter_rejection_counts": dict(rejection_counts),
        "prefilter_rejection_examples": dict(rejection_examples),
    }


def describe(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    percentile = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def select_rows(
    args: argparse.Namespace, candidates: list[dict[str, Any]], registry: BenchmarkRegistry
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    rejection_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_image_question: set[tuple[str, str]] = set()
    seen_sample_ids: set[str] = set()
    image_cache: dict[str, tuple[dict[str, Any], int, int, int]] = {}
    for row in candidates:
        image_value = str(row.get("image", ""))
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = args.image_root / image_path
        reason = None
        detail: dict[str, Any] = {}
        try:
            cache_key = str(image_path)
            if cache_key not in image_cache:
                data = image_path.read_bytes()
                fp = image_fingerprints(data)
                resized_h, resized_w = smart_resize(
                    int(fp["height"]),
                    int(fp["width"]),
                    factor=28,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                )
                image_tokens = (resized_h // 28) * (resized_w // 28)
                image_cache[cache_key] = (fp, resized_h, resized_w, image_tokens)
            fp, resized_h, resized_w, image_tokens = image_cache[cache_key]
            match_type, benchmarks, hamming = registry.image_match(
                fp,
                hamming_threshold=args.perceptual_hamming_threshold,
                aspect_tolerance=args.perceptual_aspect_tolerance,
            )
            if match_type:
                reason = f"benchmark_image_{match_type}"
                detail = {"matched_benchmarks": benchmarks, "perceptual_hamming_distance": hamming}
            elif image_tokens > args.max_prompt_image_tokens:
                reason = "prompt_image_tokens_gt_cap"
            else:
                pair = (fp["pixel_sha256"], normalize_question(row.get("question", "")))
                if pair in seen_image_question:
                    reason = "duplicate_image_question"
                elif str(row.get("sample_id", "")) in seen_sample_ids:
                    reason = "duplicate_sample_id"
        except Exception as exc:
            reason = "missing_or_invalid_image"
            detail = {"error": repr(exc)}
        if reason:
            rejection_counts[reason] += 1
            if len(rejection_examples[reason]) < 100:
                rejection_examples[reason].append(
                    {
                        "sample_id": row.get("sample_id"),
                        "source": row.get("source"),
                        "source_id": row.get("source_id"),
                        "image": image_value,
                        "question": row.get("question"),
                        **detail,
                    }
                )
            continue
        out = dict(row)
        out.update(
            {
                "qwen_prompt_image_tokens": int(image_tokens),
                "qwen_resized_height": int(resized_h),
                "qwen_resized_width": int(resized_w),
                "image_width": int(fp["width"]),
                "image_height": int(fp["height"]),
                "image_area": int(fp["width"]) * int(fp["height"]),
                "image_token_metadata_mode": "estimate_smart_resize",
                "image_token_filter_max_prompt_image_tokens": args.max_prompt_image_tokens,
                "image_token_filter_min_pixels": args.min_pixels,
                "image_token_filter_max_pixels": args.max_pixels,
                "decontamination_version": "exact_pixel_dhash_question_v1",
                "decontamination_seed": args.seed,
            }
        )
        selected.append(out)
        seen_image_question.add((fp["pixel_sha256"], normalize_question(row.get("question", ""))))
        seen_sample_ids.add(str(row.get("sample_id", "")))
        if len(selected) == args.n:
            break
    if len(selected) != args.n:
        raise RuntimeError(
            f"selected only {len(selected)} rows from {len(candidates)} candidates; "
            "increase --candidate_pool_size"
        )
    return selected, {
        "selection_rejection_counts": dict(rejection_counts),
        "selection_rejection_examples": dict(rejection_examples),
        "candidate_rows_examined": sum(rejection_counts.values()) + len(selected),
        "unique_images_opened": len(image_cache),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate(value: Any, limit: int = 300) -> str:
    text = str(value or "").replace("\n", "<br>").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    if args.n <= 0 or args.candidate_pool_size < args.n:
        raise ValueError("require 0 < n <= candidate_pool_size")
    if not 0 <= args.perceptual_hamming_threshold <= 3:
        raise ValueError("perceptual_hamming_threshold must be in [0, 3]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    named_benchmarks = [
        parse_named_path(value, args.benchmark_root)
        for value in (args.benchmark or DEFAULT_BENCHMARKS)
    ]
    missing = [str(path) for _, path in named_benchmarks if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing benchmark snapshots: {missing}")
    registry = build_benchmark_registry(named_benchmarks)
    candidates, prefilter_stats = collect_candidate_pool(args, registry)
    selected, selection_stats = select_rows(args, candidates, registry)

    stem = f"train{args.n // 1000}k_decontam_qwentok{args.max_qwen_target_tokens}_imgtok{args.max_prompt_image_tokens}_seed{args.seed}"
    train_path = args.output_dir / f"{stem}.jsonl"
    stats_path = args.output_dir / f"{stem}_stats.json"
    preview_path = args.output_dir / f"{stem}_preview{args.preview_n}.md"
    excluded_path = args.output_dir / "decontamination_excluded_examples.jsonl"
    readme_path = args.output_dir / "README.md"
    write_jsonl(train_path, selected)

    excluded_rows = []
    for stage in (prefilter_stats["prefilter_rejection_examples"], selection_stats["selection_rejection_examples"]):
        for reason, rows in stage.items():
            excluded_rows.extend({"exclusion_reason": reason, **row} for row in rows)
    write_jsonl(excluded_path, excluded_rows)

    source_counts = Counter(str(row.get("source", "")) for row in selected)
    stats_doc = {
        "status": "passed",
        "decontamination_version": "exact_pixel_dhash_question_v1",
        "input_jsonl": str(args.input_jsonl.resolve()),
        "input_sha256": sha256_file(args.input_jsonl),
        "source_dataset": args.source_dataset,
        "source_dataset_revision": args.source_dataset_revision,
        "output_jsonl": str(train_path.resolve()),
        "output_sha256": sha256_file(train_path),
        "image_root": str(args.image_root.resolve()),
        "seed": args.seed,
        "selected_rows": len(selected),
        "unique_sample_ids": len({str(row.get("sample_id", "")) for row in selected}),
        "unique_source_ids": len({(str(row.get("source", "")), str(row.get("source_id", ""))) for row in selected}),
        "unique_image_question_pairs": len(
            {(str(row.get("image", "")), normalize_question(row.get("question", ""))) for row in selected}
        ),
        "source_counts": dict(sorted(source_counts.items())),
        "qwen_target_token_len": describe([int(row["qwen_target_token_len"]) for row in selected]),
        "qwen_prompt_image_tokens": describe([int(row["qwen_prompt_image_tokens"]) for row in selected]),
        "caps": {
            "max_qwen_target_tokens": args.max_qwen_target_tokens,
            "max_prompt_image_tokens": args.max_prompt_image_tokens,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "internal_deduplication": ["source+source_id", "sample_id", "decoded_image+normalized_question"],
        "benchmark_decontamination": {
            "strict_question": True,
            "relaxed_question": True,
            "encoded_sha256": True,
            "decoded_rgb_sha256": True,
            "dhash64_hamming_threshold": args.perceptual_hamming_threshold,
            "dhash_aspect_tolerance": args.perceptual_aspect_tolerance,
            "benchmarks": registry.benchmark_stats,
            "registry_unique_strict_questions": len(registry.strict_questions),
            "registry_unique_relaxed_questions": len(registry.relaxed_questions),
            "registry_unique_encoded_images": len(registry.encoded),
            "registry_unique_pixel_images": len(registry.pixel),
        },
        **prefilter_stats,
        **selection_stats,
        "limitations": [
            "No finite automated audit can rule out semantic paraphrases or arbitrary image crops.",
            "dHash is a conservative near-duplicate screen, not a semantic image retrieval model.",
            "The guarantee applies only to the benchmark snapshot files and SHA-256 hashes recorded here.",
        ],
    }
    stats_path.write_text(json.dumps(stats_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Decontaminated LLaVA-CoT 10K Preview",
        "",
        "| # | sample_id | source | image tokens | target tokens | image | question | target |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    for index, row in enumerate(selected[: args.preview_n], 1):
        lines.append(
            f"| {index} | {truncate(row.get('sample_id'))} | {truncate(row.get('source'))} | "
            f"{row.get('qwen_prompt_image_tokens')} | {row.get('qwen_target_token_len')} | "
            f"{truncate(row.get('image'), 160)} | {truncate(row.get('question'))} | "
            f"{truncate(row.get('answer'))} |"
        )
    preview_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme_path.write_text(
        "\n".join(
            (
                "# OpenMMReasoner LLaVA-CoT 10K Decontaminated v1",
                "",
                "This is the only LLaVA-CoT training subset approved for new scope experiments.",
                "It was sampled deterministically after internal deduplication and benchmark decontamination.",
                "",
                f"- Training JSONL: `{train_path.name}`",
                f"- SHA-256: `{stats_doc['output_sha256']}`",
                f"- Seed: `{args.seed}`",
                f"- Rows: `{len(selected)}`",
                f"- Benchmark snapshots: `{len(named_benchmarks)}`",
                f"- Target token cap: `{args.max_qwen_target_tokens}`",
                f"- Image token cap: `{args.max_prompt_image_tokens}`",
                "",
                "The stats JSON records every benchmark file hash and exclusion count. The audit rules",
                "eliminate exact question/image overlap and conservative dHash near duplicates. They do",
                "not constitute a guarantee against arbitrary semantic paraphrases or transformed crops.",
                "",
                "The previous seed-42 10K contains POPE-overlapping images and must not be used for",
                "new paper experiments.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output_jsonl": str(train_path),
                "output_sha256": stats_doc["output_sha256"],
                "selected_rows": len(selected),
                "selection_rejection_counts": selection_stats["selection_rejection_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
