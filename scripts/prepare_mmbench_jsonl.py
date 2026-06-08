#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--dataset_path", default="lmms-lab/MMBench")
    p.add_argument("--dataset_name", default="en")
    p.add_argument("--split", default="dev")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


def valid_option(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from datasets import load_dataset

    out_path = Path(args.output_jsonl)
    image_root = Path(args.image_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_path, args.dataset_name, split=args.split, token=True)
    indices = list(range(len(ds)))
    if args.limit and args.limit > 0:
        rng = random.Random(args.seed)
        rng.shuffle(indices)
        indices = sorted(indices[: args.limit])

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx in indices:
            row: dict[str, Any] = ds[idx]
            mmbench_index = int(row["index"])
            image_path = image_root / f"mmbench_{args.dataset_name}_{args.split}_{mmbench_index}.jpg"
            if not image_path.exists():
                row["image"].convert("RGB").save(image_path)

            question_parts = []
            hint = str(row.get("hint", "")).strip()
            if valid_option(hint):
                question_parts.append(hint)
            question_parts.append(str(row["question"]).strip())
            choices = []
            for letter in "ABCD":
                if valid_option(row.get(letter)):
                    choices.append(f"{letter}. {str(row[letter]).strip()}")
            record = {
                "sample_id": f"mmbench_{args.dataset_name}_{args.split}_{mmbench_index}",
                "image": image_path.name,
                "question": "\n".join(question_parts),
                "choices": choices,
                "answer": str(row["answer"]).strip().upper(),
                "source": "mmbench",
                "category": str(row.get("category", "")),
                "l2_category": str(row.get("L2-category", "")),
                "mmbench_index": mmbench_index,
                "mmbench_source": str(row.get("source", "")),
                "split": str(row.get("split", args.split)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({"output_jsonl": str(out_path), "image_root": str(image_root), "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
