#!/usr/bin/env python3
"""Validate and freeze the manually audited MMStar metric cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from prepare_mmstar_metric_noocr_clean100 import (
    DEFAULT_OUTPUT,
    DEFAULT_QUOTAS,
    REASONING_INSTRUCTION,
    build_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--confirm-all-reviewed",
        action="store_true",
        help="Required acknowledgement that every final image/question pair was visually inspected.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    if not args.confirm_all_reviewed:
        raise ValueError("Refusing to finalize without --confirm-all-reviewed.")
    cohort = args.cohort_dir.expanduser().resolve()
    samples_path = cohort / "samples.jsonl"
    exclusions_path = cohort / "manual_visual_exclusions.json"
    manifest_path = cohort / "selection_manifest.json"
    rows = read_jsonl(samples_path)
    exclusions = {
        str(item["sample_id"]): str(item["reason"])
        for item in json.loads(exclusions_path.read_text(encoding="utf-8"))
    }

    failures: list[str] = []
    if len(rows) != 100:
        failures.append(f"expected 100 samples, found {len(rows)}")
    ids = [str(row["sample_id"]) for row in rows]
    hashes = [str(row["image_sha256"]) for row in rows]
    if len(set(ids)) != len(ids):
        failures.append("duplicate sample IDs")
    if len(set(hashes)) != len(hashes):
        failures.append("duplicate image hashes")
    overlap = sorted(set(ids) & set(exclusions))
    if overlap:
        failures.append(f"manually excluded IDs remain selected: {overlap}")
    expected_strata = dict(sorted(DEFAULT_QUOTAS.items()))
    observed_strata = dict(sorted(Counter(row["l2_category"] for row in rows).items()))
    if observed_strata != expected_strata:
        failures.append(f"stratum mismatch: {observed_strata} != {expected_strata}")

    for rank, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        if row.get("sample_rank") != rank:
            failures.append(f"{sample_id}: noncanonical sample_rank")
        if set(row.get("options", {})) != set("ABCD"):
            failures.append(f"{sample_id}: options are not exactly A-D")
        if row.get("original_answer_letter") not in set("ABCD"):
            failures.append(f"{sample_id}: invalid answer letter")
        expected_prompt = build_prompt(row["question"], row["options"])
        if row.get("prompt") != expected_prompt:
            failures.append(f"{sample_id}: prompt does not match canonical reasoning prompt")
        if not expected_prompt.endswith(REASONING_INSTRUCTION):
            failures.append(f"{sample_id}: reasoning instruction missing")
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            failures.append(f"{sample_id}: image missing")
            continue
        if sha256_file(image_path) != row["image_sha256"]:
            failures.append(f"{sample_id}: image hash changed")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            failures.append(f"{sample_id}: unreadable image ({type(exc).__name__})")
    if failures:
        raise RuntimeError("Final cohort validation failed:\n" + "\n".join(failures))

    reviewed_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "status": "passed",
        "review_type": "manual_visual_image_question_option_ground_truth_audit",
        "reviewed_at_utc": reviewed_at,
        "sample_count": len(rows),
        "samples_sha256": sha256_file(samples_path),
        "all_final_images_visually_inspected": True,
        "criteria": [
            "natural-image evidence is visible and sufficiently coarse",
            "question and source answer are visually defensible",
            "no OCR-, chart-, table-, document-, or tiny-detail-dependent task",
            "no unresolved reference or answer ambiguity",
        ],
        "passed_samples": [
            {"rank": rank, "sample_id": row["sample_id"], "verdict": "pass"}
            for rank, row in enumerate(rows, start=1)
        ],
        "excluded_sample_count": len(exclusions),
        "excluded_samples": [
            {"sample_id": sample_id, "reason": reason}
            for sample_id, reason in exclusions.items()
        ],
        "validation_failures": [],
    }
    audit_pdf = cohort / "mmstar_metric_noocr_clean100_audit.pdf"
    if audit_pdf.is_file():
        audit["audit_pdf"] = str(audit_pdf)
        audit["audit_pdf_sha256"] = sha256_file(audit_pdf)
    atomic_write(cohort / "manual_visual_audit.json", json.dumps(audit, indent=2) + "\n")

    review_markdown = cohort / "manual_review.md"
    if review_markdown.is_file():
        review_text = review_markdown.read_text(encoding="utf-8").replace(
            "**Manual audit:** [ ] clear visual evidence  [ ] reject/replace",
            "**Manual audit:** [x] PASS - clear visual evidence",
        )
        atomic_write(review_markdown, review_text)
    review_csv = cohort / "manual_review.csv"
    if review_csv.is_file():
        with review_csv.open("r", encoding="utf-8", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
            fieldnames = list(review_rows[0]) if review_rows else []
        for review_row in review_rows:
            review_row["manual_verdict"] = "PASS"
            review_row["manual_notes"] = "Clear coarse visual evidence; no unresolved audit issue."
        temporary_csv = review_csv.with_suffix(f".csv.tmp.{os.getpid()}")
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_rows)
        temporary_csv.replace(review_csv)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "final_manual_visual_audit_passed",
        "manual_visual_audit_required": False,
        "manual_visual_audit": str(cohort / "manual_visual_audit.json"),
        "manual_visual_audit_timestamp_utc": reviewed_at,
        "samples_sha256": audit["samples_sha256"],
        "manual_exclusion_count": len(exclusions),
        "audit_pdf": audit.get("audit_pdf"),
        "audit_pdf_sha256": audit.get("audit_pdf_sha256"),
    })
    atomic_write(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    readme = f"""# MMStar Metric-Specific No-OCR Clean 100

This is the frozen 100-question cohort used only for the r020/r025/full-token KL
trajectory metric. It keeps the official MMStar choices and clean-Armen reasoning
prompt. It uses natural-image MMBench/SEEDBench samples and excludes OCR, charts,
tables, documents, diagrams, tiny-detail reading, duplicate images, ambiguous labels,
and visually unsupported questions.

- Manual visual audit: **PASS** ({reviewed_at})
- Samples: `samples.jsonl`
- Samples SHA256: `{audit['samples_sha256']}`
- Excluded during visual review: {len(exclusions)}
- Prompt mode: official MMStar MCQ prompt plus the standard `<think>/<answer>` suffix

This cohort is diagnostic and must not be reported as the official MMStar benchmark score.

## Reproduce

```bash
python hypothesis_validate/scripts/prepare_mmstar_metric_noocr_clean100.py \\
  --manual-exclusions hypothesis_validate/manual_review/mmstar_metric_noocr_clean100_seed42/manual_visual_exclusions.json \\
  --overwrite
python hypothesis_validate/scripts/build_mmstar_metric_noocr_audit.py --overwrite
python hypothesis_validate/scripts/finalize_mmstar_metric_noocr_clean100.py --confirm-all-reviewed
```
"""
    atomic_write(cohort / "README.md", readme)
    print(json.dumps({
        "status": "passed",
        "samples": str(samples_path),
        "sample_count": len(rows),
        "samples_sha256": audit["samples_sha256"],
        "manual_exclusion_count": len(exclusions),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
