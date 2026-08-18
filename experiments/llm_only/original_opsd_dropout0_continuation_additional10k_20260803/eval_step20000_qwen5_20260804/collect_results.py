#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path("/scratch/enmingzz/outputs/llm_only/eval/original_opsd_dropout0_step20000_merged_qwen5_20260804")
CAMPAIGN = "original_opsd_dropout0_step20000_merged_reasoning_qwen5_cleanarmen_20260804"


def main() -> int:
    manifest_path = ROOT / "submission.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = list(csv.DictReader(manifest_path.open(encoding="utf-8"), delimiter="\t"))
    rows: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for item in manifest:
        pattern = f"original_opsd_dropout0_step20000_merged_{item['ratio']}_*_{item['job_id']}"
        matches = list((OUT_ROOT / "eval_vlmevalkit_trainenv" / CAMPAIGN).glob(pattern))
        if len(matches) != 1 or not matches[0].joinpath("case_validation_summary.json").is_file():
            missing.append({"job_id": item["job_id"], "task": item["task"], "ratio": item["ratio"]})
            continue
        rows.append(json.loads(matches[0].joinpath("case_validation_summary.json").read_text()))

    rows.sort(key=lambda row: (str(row["dataset"]), str(row["ratio"])))
    payload = {"expected": 20, "completed": len(rows), "missing": missing, "results": rows}
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "ratio", "score", "score_unit", "rows", "slurm_job_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps(payload, indent=2))
    return 0 if len(rows) == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
