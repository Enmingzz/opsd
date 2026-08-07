#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/project/6101803/enmingzz/opsd/experiments/llm_only/jsd_weighted_opsd_five_eval_main4_20260807")
OUT = Path("/scratch/enmingzz/outputs/llm_only/eval/jsd_weighted_opsd_five_step10240_merged_20260807")
ACTIVE_DATASETS = {"MME", "MMStar", "MathVista_MINI"}
ACTIVE_METHODS = {"direct_inv_d25", "softmax_t005_d25", "softmax_t010_d25"}
EXPECTED_CASES = 36


def main() -> int:
    rows = []
    for path in OUT.glob("eval_vlmevalkit_trainenv/**/case_validation_summary.json"):
        payload = json.loads(path.read_text())
        if payload.get("dataset") not in ACTIVE_DATASETS or payload.get("method") not in ACTIVE_METHODS:
            continue
        payload["summary_path"] = str(path)
        rows.append(payload)
    rows.sort(key=lambda row: (row["method"], row["dataset"], row["ratio"]))
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "complete" if len(rows) == EXPECTED_CASES and all(row["status"] == "passed" for row in rows) else "incomplete",
        "collected_at": datetime.now().astimezone().isoformat(),
        "expected_cases": EXPECTED_CASES,
        "completed_cases": len(rows),
        "active_datasets": sorted(ACTIVE_DATASETS),
        "active_methods": sorted(ACTIVE_METHODS),
        "intentionally_cancelled_dataset": "MMMU_Pro_4c",
        "intentionally_cancelled_methods": ["softmax_t005_d40", "softmax_t010_d40"],
        "results": rows,
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "expected_cases", "completed_cases")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
