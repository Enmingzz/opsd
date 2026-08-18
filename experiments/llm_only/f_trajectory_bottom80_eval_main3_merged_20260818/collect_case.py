#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_ROWS = {"MME": 2374, "MMStar": 1500, "MathVista_MINI": 1000}


def only(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def first_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def score(work: Path, dataset: str, workbook: Path) -> tuple[float, Path, str]:
    if dataset == "MME":
        path = only(list((work / "posthoc_qwen7yn").glob("*_qwen_unknown_judge_summary.csv")), "MME score")
        return float(first_row(path)["primary_score"]), path, "MME points"
    if dataset == "MMStar":
        path = workbook.with_name("Qwen_MMStar_acc.csv")
        if not path.is_file():
            path = only(list((work / "Qwen").glob("**/Qwen_MMStar_acc.csv")), "MMStar score")
        return 100.0 * float(first_row(path)["Overall"]), path, "percent"
    path = workbook.with_name("Qwen_MathVista_MINI_qwen7judge_score.csv")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    row = next(item for item in rows if item["Task&Skill"] == "Overall")
    return float(row["acc"]), path, "percent (legacy; strict-GT follows)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(EXPECTED_ROWS), required=True)
    parser.add_argument("--ratio", choices=("noprune", "r010", "r020", "r030"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--method-long", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    validation = json.loads((work / "raw_prediction_validation.json").read_text())
    assert validation["status"] == "passed"
    assert int(validation["rows"]) == EXPECTED_ROWS[args.dataset]
    workbook = Path(validation["result_file"]).resolve()
    sidecar = Path(validation["sidecar_file"]).resolve()
    workbook.relative_to(work)
    sidecar.relative_to(work)
    assert workbook.is_file() and sidecar.is_file()
    value, score_path, unit = score(work, args.dataset, workbook)
    payload = {
        "status": "passed",
        "method": args.method,
        "method_long": args.method_long,
        "checkpoint_step": 10240,
        "checkpoint_load_form": "merged_full_checkpoint",
        "dataset": args.dataset,
        "ratio": args.ratio,
        "rows": EXPECTED_ROWS[args.dataset],
        "score": value,
        "score_unit": unit,
        "slurm_job_id": args.job_id,
        "score_file": str(score_path),
        "raw_workbook": str(workbook),
        "raw_prediction_sidecar": str(sidecar),
    }
    (work / "case_validation_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
