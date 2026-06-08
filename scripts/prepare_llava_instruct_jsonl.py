#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--llava_json", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_jsonl", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.0)
    p.add_argument("--output_train_jsonl", default="")
    p.add_argument("--output_val_jsonl", default="")
    return p


def clean_question(text: str) -> str:
    text = re.sub(r"<image>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def role_of(message: dict[str, Any]) -> str:
    return str(message.get("from", message.get("role", ""))).strip().lower()


def value_of(message: dict[str, Any]) -> str:
    return str(message.get("value", message.get("content", ""))).strip()


def extract_qa(record: dict[str, Any]) -> tuple[str, str] | None:
    conversations = record.get("conversations", record.get("messages"))
    if not isinstance(conversations, list):
        return None
    question = None
    answer = None
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = role_of(message)
        value = value_of(message)
        if question is None and role in {"human", "user"} and value:
            question = clean_question(value)
            continue
        if answer is None and role in {"gpt", "assistant"} and value:
            answer = value
        if question and answer:
            break
    if not question or not answer:
        return None
    return question, answer


def image_path_for(record: dict[str, Any]) -> str | None:
    image = record.get("image", record.get("image_path", record.get("file_name")))
    if isinstance(image, list):
        return None
    if not isinstance(image, str) or not image.strip():
        return None
    return image.strip()


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    random.seed(args.seed)
    image_root = Path(args.image_root)

    with open(args.llava_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected LLaVA input JSON to be a list of records.")

    rows: list[dict[str, Any]] = []
    missing_images = 0
    skipped_bad = 0
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            skipped_bad += 1
            continue
        image = image_path_for(record)
        qa = extract_qa(record)
        if image is None or qa is None:
            skipped_bad += 1
            continue
        resolved_image = Path(image) if Path(image).is_absolute() else image_root / image
        if not resolved_image.exists():
            missing_images += 1
            continue
        if not os.access(resolved_image, os.R_OK):
            missing_images += 1
            continue
        question, answer = qa
        base_id = str(record.get("id", record.get("sample_id", idx)))
        rows.append(
            {
                "sample_id": f"{base_id}_{idx}",
                "source_id": base_id,
                "image": image,
                "question": question,
                "answer": answer,
                "source": "llava_instruct",
            }
        )

    random.shuffle(rows)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    val_size = int(round(len(rows) * max(0.0, min(1.0, float(args.val_ratio)))))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rows)
    write_jsonl(args.output_train_jsonl, train_rows)
    write_jsonl(args.output_val_jsonl, val_rows)

    stats = {
        "total_loaded": len(data),
        "total_written": len(rows),
        "missing_images": missing_images,
        "skipped_bad_records": skipped_bad,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }
    print(json.dumps(stats, indent=2))
    if not args.output_jsonl and not args.output_train_jsonl and not args.output_val_jsonl:
        print("No output path was provided.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
