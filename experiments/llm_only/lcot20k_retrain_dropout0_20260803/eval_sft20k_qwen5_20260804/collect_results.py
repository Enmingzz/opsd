#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path("/scratch/enmingzz/outputs/llm_only/eval/sft20k_dropout0_step20000_merged_qwen5_20260804")


def main() -> int:
    jobs = list(csv.DictReader((ROOT / "submission.tsv").open(encoding="utf-8"), delimiter="\t"))
    rows: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for item in jobs:
        candidates = list(OUT_ROOT.rglob(f"*_{item['job_id']}/case_validation_summary.json"))
        if len(candidates) != 1:
            missing.append(item)
            continue
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        rows.append(payload)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["ratio"])))
    (ROOT / "results.json").write_text(json.dumps({"rows": rows, "missing": missing}, indent=2) + "\n")
    fields = ["dataset", "ratio", "score", "score_unit", "rows", "slurm_job_id"]
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    print(json.dumps({"completed": len(rows), "missing": len(missing)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
