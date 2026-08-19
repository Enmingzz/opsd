#!/usr/bin/env python3
"""Build a deterministic, numeric-answer MMStar option-free audit cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COHORT = ROOT / "manual_review" / "mmstar_open_ended_clean100_seed42"
DEFAULT_OUTPUT_DIR = ROOT / "manual_review" / "mmstar_numeric_open_ended_clean100_seed42"
NUMERIC_VALUE = re.compile(
    r"""(?ix)^\s*
    (?P<qualifier>about|approximately|around|roughly|nearly|over|under|at\s+least|at\s+most)?\s*
    [-+]?\s*(?P<prefix>[$£€¥])?\s*
    (?P<number>(?:\d+(?:[,.]\d+)*(?:\s*/\s*\d+)?|\.\d+))
    (?P<unit>\s*(?:%|°|degrees?|percent|pixels?|px|cm|mm|m|km|inches?|in|feet|foot|ft|
        yards?|yd|miles?|mi|grams?|g|kg|kilograms?|lbs?|pounds?|seconds?|secs?|minutes?|
        mins?|hours?|hrs?|days?|weeks?|months?|years?|dollars?|cents?|yuan|meters?|liters?|
        litres?|ml|mph|km/h|m/s|am|pm|a\.m\.|p\.m\.))?
    (?:\s*(?:wide|long|high|tall))?\s*[.!]?\s*$"""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, default=SOURCE_COHORT / "candidate_pool.jsonl")
    parser.add_argument(
        "--semantic-audit",
        type=Path,
        default=SOURCE_COHORT / "semantic_option_independence_audit.jsonl",
    )
    parser.add_argument(
        "--manual-exclusions",
        type=Path,
        default=SOURCE_COHORT / "manual_visual_exclusions.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_prepare_module() -> Any:
    path = Path(__file__).with_name("prepare_mmstar_manual_review_clean100.py")
    spec = importlib.util.spec_from_file_location("prepare_mmstar_clean100", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_signature(value: str) -> tuple[str, str] | None:
    match = NUMERIC_VALUE.fullmatch(str(value))
    if match is None:
        return None
    return (
        (match.group("prefix") or "").strip().lower(),
        (match.group("unit") or "").strip().lower(),
    )


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def rejection_reasons(
    row: dict[str, Any],
    audits: dict[str, dict[str, Any]],
    manually_excluded: set[str],
) -> list[str]:
    sample_id = str(row["sample_id"])
    reasons: list[str] = []
    if sample_id in manually_excluded:
        reasons.append("prior_manual_visual_exclusion")
    if not audits.get(sample_id, {}).get("unanimous_pass"):
        reasons.append("semantic_option_independence_not_unanimous")
    if numeric_signature(str(row.get("answer", ""))) is None:
        reasons.append("reference_is_not_strict_numeric")
    options = row.get("options", {})
    if set(options) != set("ABCD"):
        reasons.append("not_exactly_four_source_options")
    option_signatures = [numeric_signature(str(options.get(letter, ""))) for letter in "ABCD"]
    if any(signature is None for signature in option_signatures):
        reasons.append("not_all_source_options_are_strict_numeric")
    elif len(set(option_signatures)) != 1:
        reasons.append("source_option_unit_signatures_differ")
    elif numeric_signature(str(row.get("answer", ""))) != option_signatures[0]:
        reasons.append("reference_unit_signature_differs_from_options")
    return reasons


def main() -> int:
    args = parse_args()
    prepare = load_prepare_module()
    candidate_path = args.candidate_pool.expanduser().resolve()
    audit_path = args.semantic_audit.expanduser().resolve()
    manual_exclusions_path = args.manual_exclusions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    for path in (candidate_path, audit_path, manual_exclusions_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates = prepare.read_jsonl(candidate_path)
    audits = {str(row["sample_id"]): row for row in prepare.read_jsonl(audit_path)}
    manual_records = json.loads(manual_exclusions_path.read_text(encoding="utf-8"))
    manually_excluded = {str(row["sample_id"]) for row in manual_records}

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        reasons = rejection_reasons(candidate, audits, manually_excluded)
        if reasons:
            rejected.append({
                "sample_id": str(candidate["sample_id"]),
                "question": candidate["question"],
                "answer": candidate["answer"],
                "reasons": reasons,
            })
            continue
        answer_signature = numeric_signature(str(candidate["answer"]))
        eligible.append({
            **candidate,
            "numeric_filter": {
                "strict_numeric_reference": True,
                "all_four_source_options_strict_numeric": True,
                "consistent_unit_signature": True,
                "unit_signature": list(answer_signature or ("", "")),
            },
        })

    ordered = sorted(eligible, key=lambda row: stable_key(args.seed, str(row["sample_id"])))
    selected: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    duplicate_image_skips: list[str] = []
    for row in ordered:
        fingerprint = str(row["source_image_fingerprint"])
        if fingerprint in seen_images:
            duplicate_image_skips.append(str(row["sample_id"]))
            continue
        selected.append(row)
        seen_images.add(fingerprint)
        if len(selected) == args.n:
            break
    if len(selected) != args.n:
        raise RuntimeError(
            f"Requested {args.n} unique-image numeric rows, but only {len(selected)} are available."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_pool_path = output_dir / "candidate_pool.jsonl"
    base_selection_path = output_dir / "numeric_base_selection.jsonl"
    rejected_path = output_dir / "numeric_filter_rejections.jsonl"
    prepare.atomic_write_jsonl(numeric_pool_path, eligible)
    prepare.atomic_write_jsonl(base_selection_path, selected)
    prepare.atomic_write_jsonl(rejected_path, rejected)

    manifest = prepare.build_final(
        output_dir=output_dir,
        audit_path=audit_path,
        n=args.n,
        seed=args.seed,
        overwrite=args.overwrite,
        manual_exclusions_path=None,
        base_selection_path=base_selection_path,
    )
    samples = prepare.read_jsonl(output_dir / "samples.jsonl")
    numeric_failures: list[str] = []
    for row in samples:
        signatures = [numeric_signature(str(row["answer"]))]
        signatures.extend(numeric_signature(str(row["options"][letter])) for letter in "ABCD")
        if any(signature is None for signature in signatures):
            numeric_failures.append(f"{row['sample_id']}: nonnumeric answer or option")
        elif len(set(signatures)) != 1:
            numeric_failures.append(f"{row['sample_id']}: inconsistent unit signatures")
    report = {
        "status": "failed" if numeric_failures else "passed",
        "selected_count": len(samples),
        "strict_numeric_reference_count": sum(
            numeric_signature(str(row["answer"])) is not None for row in samples
        ),
        "all_four_options_strict_numeric_count": sum(
            all(numeric_signature(str(row["options"][letter])) is not None for letter in "ABCD")
            for row in samples
        ),
        "consistent_unit_signature_count": len(samples) - len(numeric_failures),
        "unique_image_count": len({row["image_sha256"] for row in samples}),
        "eligible_numeric_pool_count": len(eligible),
        "duplicate_image_skips_before_selection_complete": duplicate_image_skips,
        "l2_category_counts": dict(sorted(Counter(row["l2_category"] for row in samples).items())),
        "source_benchmark_counts": dict(sorted(Counter(row["bench"] for row in samples).items())),
        "failures": numeric_failures,
    }
    report_path = output_dir / "numeric_validation_report.json"
    prepare.atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if numeric_failures:
        raise RuntimeError(f"Numeric validation failed; see {report_path}")

    manifest.update({
        "selection": (
            "strict numeric reference + all four source options numeric + consistent unit signature + "
            "hard filter + unanimous three-prompt Qwen3.6-27B option-independence audit + "
            "prior manual visual exclusions + deterministic seed-42 sampling"
        ),
        "eligible_numeric_pool_count": len(eligible),
        "strict_numeric_reference_count": len(samples),
        "all_four_source_options_strict_numeric_count": len(samples),
        "consistent_unit_signature_count": len(samples),
        "numeric_validation_report": str(report_path),
        "numeric_filter_rejections": str(rejected_path),
        "supersedes": str(SOURCE_COHORT),
    })
    prepare.atomic_write_text(
        output_dir / "selection_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    readme = f"""# MMStar Numeric Option-Free Clean 100

This is the replacement MMStar diagnostic cohort for new option-free pass@k and pruning experiments.
All {len(samples)} selected rows have a strict numeric free-text reference, four numeric source
options with a consistent unit signature, one unique image, and a unanimous three-prompt semantic
audit confirming that the question remains understandable after choices are removed.

The earlier `{SOURCE_COHORT.name}` cohort remains on disk only for historical result provenance. It
must not be reused for new comparisons because 71/100 rows had nonnumeric open-ended references and
manual review identified answer-granularity ambiguity.

Files:

- `samples.jsonl`: model-facing records.
- `manual_review.html`, `manual_review.md`, `manual_review.csv`: human-audit views.
- `numeric_validation_report.json`: fail-closed numeric checks.
- `selection_manifest.json`: selection provenance.
- `numeric_filter_rejections.jsonl`: all rejected candidates and reasons.
"""
    prepare.atomic_write_text(output_dir / "README.md", readme)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
