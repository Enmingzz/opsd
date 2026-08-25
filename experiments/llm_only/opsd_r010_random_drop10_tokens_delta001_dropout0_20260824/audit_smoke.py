#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.run_dir / "rank0_native_budget_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4
    for row in rows:
        valid = int(row["native_token_partition_valid_tokens"])
        eligible = int(row["native_token_partition_eligible_tokens"])
        dropped = int(row["native_token_partition_random_dropped_tokens"])
        kept = int(row["native_token_partition_selected_tokens"])
        assert row["loss_type"] == "opsd_nogt_token_random_drop20_forward_kl"
        assert row["native_budget_weighting_mode"] == "token_random_drop20"
        assert row["sampled_b"] == 0.1
        assert math.isclose(row["sampled_b_plus"], 0.11, abs_tol=1e-12)
        assert row["native_token_partition_top_fraction"] == 0.1
        assert row["native_token_partition_min_teacher_kl"] == 0.0
        assert row["native_token_partition_below_kl_floor_tokens"] == 0
        assert row["native_token_partition_excluded_low_kl_tokens"] == 0
        assert valid == eligible == dropped + kept
        assert dropped == math.ceil(0.10 * valid)
        assert kept > 0
        assert row["native_probe_grad_enabled"] is False
        assert row["native_weight_detached"] is True
        assert math.isfinite(row["weighted_kl_loss"])
        assert math.isfinite(row["unweighted_kl_loss"])
    payload = {
        "status": "passed",
        "samples": len(rows),
        "mean_actual_dropped_fraction": sum(
            row["native_token_partition_random_dropped_tokens"]
            / row["native_token_partition_valid_tokens"]
            for row in rows
        ) / len(rows),
        "mean_retained_to_full_loss_ratio": sum(
            row["native_weighted_to_unweighted_kl_ratio"] for row in rows
        ) / len(rows),
    }
    (args.run_dir / "smoke_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
