#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--full_predictions", required=True)
    p.add_argument("--pruned_base_predictions", required=True)
    p.add_argument("--distilled_predictions", required=True)
    p.add_argument("--output_dir", required=True)
    return p


def read_jsonl(path: str) -> dict[str, dict[str, Any]]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("sample_id"))] = row
    return rows


def correct(row: dict[str, Any] | None) -> bool | None:
    if not row:
        return None
    if "multiple_choice_accuracy" in row:
        return bool(row["multiple_choice_accuracy"])
    if "exact_match" in row:
        return bool(row["exact_match"])
    return None


def accuracy(rows: dict[str, dict[str, Any]]) -> float | None:
    vals = [correct(row) for row in rows.values()]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = read_jsonl(args.full_predictions)
    pruned = read_jsonl(args.pruned_base_predictions)
    distilled = read_jsonl(args.distilled_predictions)

    sample_ids = sorted(set(full) | set(pruned) | set(distilled))
    changed = []
    fixed = []
    broken = []
    teacher_matches = []
    for sample_id in sample_ids:
        f = full.get(sample_id)
        p = pruned.get(sample_id)
        d = distilled.get(sample_id)
        pc = correct(p)
        dc = correct(d)
        fc = correct(f)
        if pc != dc:
            item = {
                "sample_id": sample_id,
                "full_correct": fc,
                "pruned_correct": pc,
                "distilled_correct": dc,
                "question": (d or p or f or {}).get("question", ""),
                "gold_answer": (d or p or f or {}).get("gold_answer", ""),
                "full_prediction": (f or {}).get("prediction", ""),
                "pruned_prediction": (p or {}).get("prediction", ""),
                "distilled_prediction": (d or {}).get("prediction", ""),
            }
            changed.append(item)
            if pc is False and dc is True:
                fixed.append(item)
            if pc is True and dc is False:
                broken.append(item)
        if f and d and str(f.get("prediction", "")).strip() == str(d.get("prediction", "")).strip():
            teacher_matches.append(sample_id)

    full_acc = accuracy(full)
    pruned_acc = accuracy(pruned)
    distilled_acc = accuracy(distilled)
    gap_recovery = None
    if full_acc is not None and pruned_acc is not None and distilled_acc is not None and abs(full_acc - pruned_acc) > 1e-12:
        gap_recovery = (distilled_acc - pruned_acc) / (full_acc - pruned_acc)

    summary = {
        "full_token_accuracy": full_acc,
        "pruned_base_accuracy": pruned_acc,
        "distilled_student_accuracy": distilled_acc,
        "gap_recovery": gap_recovery,
        "num_samples": len(sample_ids),
        "num_changed": len(changed),
        "num_fixed_by_distillation": len(fixed),
        "num_broken_by_distillation": len(broken),
        "num_distilled_matches_full_token_teacher": len(teacher_matches),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "changed_samples.jsonl").open("w", encoding="utf-8") as f:
        for row in changed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = [
        "# OPSD Eval Comparison",
        "",
        f"- Full-token accuracy: {full_acc}",
        f"- Pruned-base accuracy: {pruned_acc}",
        f"- Distilled-student accuracy: {distilled_acc}",
        f"- Gap recovery: {gap_recovery}",
        f"- Samples fixed by distillation: {len(fixed)}",
        f"- Samples broken by distillation: {len(broken)}",
        f"- Distilled predictions matching full-token teacher text: {len(teacher_matches)}",
        "",
        "See `changed_samples.jsonl` for per-sample changes.",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
