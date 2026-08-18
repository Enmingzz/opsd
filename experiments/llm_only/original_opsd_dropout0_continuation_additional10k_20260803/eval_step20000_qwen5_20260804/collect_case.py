#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_ROWS = {
    "MME": 2374,
    "MMStar": 1500,
    "MathVista_MINI": 1000,
    "MathVerse_MINI_Vision_Only": 788,
    "MMMU_Pro_4c": 1730,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(EXPECTED_ROWS), required=True)
    parser.add_argument("--ratio", choices=["noprune", "r010", "r020", "r030"], required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def only(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {len(paths)}: {paths}")
    return paths[0]


def first_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def score(work_dir: Path, dataset: str, result: Path) -> tuple[float, Path, str, dict[str, object]]:
    extra: dict[str, object] = {}
    if dataset == "MME":
        path = only(list((work_dir / "posthoc_qwen7yn").glob("*_qwen_unknown_judge_summary.csv")), "MME summary")
        return float(first_csv_row(path)["primary_score"]), path, "MME points", extra
    if dataset == "MMStar":
        path = result.with_name("Qwen_MMStar_acc.csv")
        return 100.0 * float(first_csv_row(path)["Overall"]), path, "percent", extra
    if dataset == "MathVista_MINI":
        path = result.with_name("Qwen_MathVista_MINI_qwen7judge_score.csv")
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        row = next(item for item in rows if item["Task&Skill"] == "Overall")
        return float(row["acc"]), path, "percent", extra
    if dataset == "MathVerse_MINI_Vision_Only":
        path = result.with_name("Qwen_MathVerse_MINI_Vision_Only_qwen7judge_score.csv")
        return float(first_csv_row(path)["Overall"]), path, "percent", extra

    official = result.with_name("Qwen_MMMU_Pro_4c_acc.csv")
    official_score = 100.0 * float(first_csv_row(official)["Overall"])
    summary_path = result.with_name("Qwen_MMMU_Pro_4c_qwen7extract_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "passed" or int(summary["rows"]) != EXPECTED_ROWS[dataset]:
        raise RuntimeError(f"Invalid MMMU Qwen summary: {summary}")
    extra.update({
        "official_score": official_score,
        "official_score_file": str(official),
        "qwen_unknown_rows": int(summary["qwen_unknown_rows"]),
        "qwen_ground_truth_visible": bool(summary["ground_truth_visible_to_extractor"]),
    })
    return float(summary["qwen_score_percent"]), Path(summary["score_file"]), "percent", extra


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    validation_path = work_dir / "raw_prediction_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation["status"] != "passed" or int(validation["rows"]) != EXPECTED_ROWS[args.dataset]:
        raise RuntimeError(f"Raw-prediction validation failed: {validation}")
    result = Path(validation["result_file"]).resolve()
    sidecar = Path(validation["sidecar_file"]).resolve()
    if not result.is_file() or not sidecar.is_file():
        raise RuntimeError("Validated raw output artifacts are missing")

    value, score_file, unit, extra = score(work_dir, args.dataset, result)
    payload: dict[str, object] = {
        "status": "passed",
        "method": "original_opsd_dropout0_step20000_merged",
        "training_method": "opsd_nogt_random_uniform_exact_resume_20k",
        "parameter_scope": "language_decoder_only",
        "lora_dropout": 0.0,
        "checkpoint_load_form": "merged_full_checkpoint",
        "dataset": args.dataset,
        "ratio": args.ratio,
        "rows": EXPECTED_ROWS[args.dataset],
        "score": value,
        "score_unit": unit,
        "postprocess": "qwen_assisted",
        "slurm_job_id": args.job_id,
        "score_file": str(score_file),
        "raw_workbook": str(result),
        "raw_prediction_sidecar": str(sidecar),
        "raw_prediction_validation": str(validation_path),
        "run_manifest": str(work_dir / "run_manifest.json"),
    }
    payload.update(extra)
    output = work_dir / "case_validation_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
