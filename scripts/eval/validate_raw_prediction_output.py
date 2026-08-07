#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = (
    "prediction",
    "parsed_prediction",
    "raw_prediction",
    "generated_token_len",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed unless a VLMEvalKit result retains raw and parsed predictions."
    )
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--sidecar-file", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args()


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def normalize_index(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return text_value(value)


def main() -> int:
    args = parse_args()
    result_file = args.result_file.resolve()
    if not result_file.is_file():
        raise SystemExit(f"Missing result file: {result_file}")

    sidecar_file = (
        args.sidecar_file.resolve()
        if args.sidecar_file
        else result_file.with_name(f"{result_file.stem}_raw_predictions.jsonl")
    )
    if not sidecar_file.is_file():
        raise SystemExit(f"Missing raw-prediction sidecar: {sidecar_file}")

    # Preserve literal answer strings such as "NA" instead of coercing them to NaN.
    table = pd.read_excel(result_file, keep_default_na=False)
    if len(table) != args.expected_rows:
        raise SystemExit(
            f"Unexpected row count in {result_file}: {len(table)} != {args.expected_rows}"
        )
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in table.columns]
    if missing_columns:
        raise SystemExit(f"Missing result columns: {missing_columns}")

    raw_values = [text_value(value) for value in table["raw_prediction"]]
    parsed_values = [text_value(value) for value in table["parsed_prediction"]]
    prediction_values = [text_value(value) for value in table["prediction"]]
    empty_raw_rows = [index for index, value in enumerate(raw_values) if not value.strip()]
    if empty_raw_rows:
        raise SystemExit(
            f"Found {len(empty_raw_rows)} empty raw predictions; first rows={empty_raw_rows[:10]}"
        )
    parser_mismatches = [
        index
        for index, (prediction, parsed) in enumerate(zip(prediction_values, parsed_values))
        if prediction != parsed
    ]
    if parser_mismatches:
        raise SystemExit(
            "prediction and parsed_prediction differ at rows "
            f"{parser_mismatches[:10]} (total={len(parser_mismatches)})"
        )

    token_lengths = pd.to_numeric(table["generated_token_len"], errors="coerce")
    invalid_token_lengths = token_lengths.isna() | (token_lengths < 0)
    if bool(invalid_token_lengths.any()):
        bad_rows = list(table.index[invalid_token_lengths])[:10]
        raise SystemExit(f"Invalid generated_token_len at rows {bad_rows}")

    sidecar_rows = []
    with sidecar_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {sidecar_file}:{line_number}: {exc}") from exc
            sidecar_rows.append(row)
    if len(sidecar_rows) != args.expected_rows:
        raise SystemExit(
            f"Unexpected sidecar row count: {len(sidecar_rows)} != {args.expected_rows}"
        )

    sidecar_raw = [text_value(row.get("raw_prediction")) for row in sidecar_rows]
    sidecar_parsed = [text_value(row.get("parsed_prediction")) for row in sidecar_rows]
    raw_mismatches = [
        index
        for index, (xlsx_value, jsonl_value) in enumerate(zip(raw_values, sidecar_raw))
        if xlsx_value != jsonl_value
    ]
    parsed_mismatches = [
        index
        for index, (xlsx_value, jsonl_value) in enumerate(zip(parsed_values, sidecar_parsed))
        if xlsx_value != jsonl_value
    ]
    if raw_mismatches or parsed_mismatches:
        raise SystemExit(
            "Raw sidecar does not match the result workbook: "
            f"raw rows={raw_mismatches[:10]}, parsed rows={parsed_mismatches[:10]}"
        )

    index_checked = False
    if "index" in table.columns and all("index" in row for row in sidecar_rows):
        index_checked = True
        xlsx_indices = [normalize_index(value) for value in table["index"]]
        sidecar_indices = [normalize_index(row["index"]) for row in sidecar_rows]
        if xlsx_indices != sidecar_indices:
            raise SystemExit("Sidecar sample indices do not match workbook sample indices")

    complete_think_answer = sum(
        "<think>" in value
        and "</think>" in value
        and "<answer>" in value
        and "</answer>" in value
        for value in raw_values
    )
    report = {
        "status": "passed",
        "result_file": str(result_file),
        "sidecar_file": str(sidecar_file),
        "rows": len(table),
        "expected_rows": args.expected_rows,
        "raw_prediction_nonempty_rows": len(raw_values),
        "prediction_matches_parsed_rows": len(table),
        "sidecar_matches_workbook_rows": len(table),
        "sample_index_order_checked": index_checked,
        "complete_think_answer_rows": complete_think_answer,
        "complete_think_answer_fraction": complete_think_answer / len(table),
        "mean_generated_token_len": float(token_lengths.mean()),
        "max_generated_token_len": int(token_lengths.max()),
    }
    report_out = (
        args.report_out.resolve()
        if args.report_out
        else result_file.with_name(f"{result_file.stem}_raw_validation.json")
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
