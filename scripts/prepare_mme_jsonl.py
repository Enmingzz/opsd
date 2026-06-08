#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="lmms-lab/MME")
    p.add_argument("--split", default="test")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_image_root", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    return p


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "sample"


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower().replace(".", "")
    if text in {"yes", "y"}:
        return "yes"
    if text in {"no", "n"}:
        return "no"
    return str(value).strip()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from datasets import load_dataset

    ds = load_dataset(args.dataset_path, split=args.split)
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(int(args.limit), len(ds))))

    image_root = Path(args.output_image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            category = str(row.get("category", "unknown"))
            question_id = str(row.get("question_id", idx))
            answer_index = idx
            sample_id = f"{safe_name(category)}_{safe_name(question_id)}_{answer_index:06d}"
            rel_image = Path(safe_name(category)) / f"{safe_name(question_id)}.jpg"
            image_path = image_root / rel_image
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = row["image"].convert("RGB")
            if not image_path.exists():
                image.save(image_path, quality=95)

            record = {
                "sample_id": sample_id,
                "image": str(rel_image),
                "question": str(row.get("question", "")).strip(),
                "answer": normalize_answer(row.get("answer", "")),
                "source": "mme",
                "category": category,
                "question_id": question_id,
                "mme_question_id": question_id,
                "mme_pair_id": question_id,
                "mme_answer_index": answer_index,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "dataset_path": args.dataset_path,
                "split": args.split,
                "output_jsonl": str(output_path),
                "output_image_root": str(image_root),
                "num_written": written,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
