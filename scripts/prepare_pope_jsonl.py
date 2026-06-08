#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--splits", default="adversarial,popular,random")
    p.add_argument("--limit_per_split", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_path", default="lmms-lab/POPE")
    p.add_argument("--dataset_name", default="Full")
    return p


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from datasets import load_dataset

    out_path = Path(args.output_jsonl)
    image_root = Path(args.image_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total_written = 0
    per_split: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for split in split_list(args.splits):
            ds = load_dataset(args.dataset_path, args.dataset_name, split=split, token=True)
            indices = list(range(len(ds)))
            if args.limit_per_split and args.limit_per_split > 0:
                rng.shuffle(indices)
                indices = sorted(indices[: args.limit_per_split])
            written = 0
            for idx in indices:
                row: dict[str, Any] = ds[idx]
                image_source = str(row["image_source"])
                image_path = image_root / f"{image_source}.jpg"
                if not image_path.exists():
                    row["image"].convert("RGB").save(image_path)
                question = str(row["question"]).strip()
                record = {
                    "sample_id": f"pope_{split}_{row['question_id']}",
                    "image": image_path.name,
                    "question": f"{question}\nAnswer the question using a single word or phrase.",
                    "answer": str(row["answer"]).strip().lower(),
                    "source": "pope",
                    "category": split,
                    "question_id": str(row["question_id"]),
                    "pope_id": str(row["id"]),
                    "image_source": image_source,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                total_written += 1
            per_split[split] = written

    print(json.dumps({"output_jsonl": str(out_path), "image_root": str(image_root), "total_written": total_written, "per_split": per_split}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
