#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = {
    "direct_inv_d25",
    "softmax_t005_d25",
    "softmax_t010_d25",
    "softmax_t005_d40",
    "softmax_t010_d40",
}
EXPECTED_ROWS = {
    "MME": 2374,
    "MMStar": 1500,
    "MathVista_MINI": 1000,
    "MMMU_Pro_4c": 1730,
}


def only(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def first_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def score(work: Path, dataset: str, workbook: Path) -> tuple[float, Path, str, dict[str, object]]:
    extra: dict[str, object] = {}
    if dataset == "MME":
        path = only(list((work / "posthoc_qwen7yn").glob("*_qwen_unknown_judge_summary.csv")), "MME score")
        return float(first_row(path)["primary_score"]), path, "MME points", extra
    if dataset == "MMStar":
        path = workbook.with_name("Qwen_MMStar_acc.csv")
        return 100.0 * float(first_row(path)["Overall"]), path, "percent", extra
    if dataset == "MathVista_MINI":
        path = workbook.with_name("Qwen_MathVista_MINI_qwen7judge_score.csv")
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        overall = next(row for row in rows if row["Task&Skill"] == "Overall")
        return float(overall["acc"]), path, "percent", extra

    official = workbook.with_name("Qwen_MMMU_Pro_4c_acc.csv")
    official_score = 100.0 * float(first_row(official)["Overall"])
    summary_path = workbook.with_name("Qwen_MMMU_Pro_4c_qwen7extract_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "passed" and int(summary["rows"]) == EXPECTED_ROWS[dataset]
    extra.update({
        "official_score": official_score,
        "official_score_file": str(official),
        "qwen_unknown_rows": int(summary["qwen_unknown_rows"]),
        "qwen_ground_truth_visible": bool(summary["ground_truth_visible_to_extractor"]),
    })
    return float(summary["qwen_score_percent"]), Path(summary["score_file"]), "percent", extra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--dataset", choices=sorted(EXPECTED_ROWS), required=True)
    parser.add_argument("--ratio", choices=["noprune", "r010", "r020", "r030"], required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    validation_path = work / "raw_prediction_validation.json"
    validation = json.loads(validation_path.read_text())
    assert validation["status"] == "passed"
    assert int(validation["rows"]) == EXPECTED_ROWS[args.dataset]
    workbook = Path(validation["result_file"]).resolve()
    sidecar = Path(validation["sidecar_file"]).resolve()
    workbook.relative_to(work)
    sidecar.relative_to(work)
    assert workbook.is_file() and sidecar.is_file()
    value, score_path, unit, extra = score(work, args.dataset, workbook)
    payload: dict[str, object] = {
        "status": "passed", "method": args.method,
        "training_method": args.method, "parameter_scope": "language_decoder_only",
        "lora_dropout": 0.0, "checkpoint_step": 10240,
        "checkpoint_load_form": "merged_full_checkpoint", "dataset": args.dataset,
        "ratio": args.ratio, "rows": EXPECTED_ROWS[args.dataset], "score": value,
        "score_unit": unit, "postprocess": "qwen_assisted",
        "slurm_job_id": args.job_id, "score_file": str(score_path),
        "raw_workbook": str(workbook), "raw_prediction_sidecar": str(sidecar),
        "raw_prediction_validation": str(validation_path),
        "run_manifest": str(work / "run_manifest.json"),
    }
    payload.update(extra)
    (work / "case_validation_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
