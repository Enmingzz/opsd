#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMAGE_KEYS = [f"image_{idx}" for idx in range(1, 8)]


def lmu_data_root() -> Path:
    value = os.environ.get("LMUData") or os.environ.get("LMU_DATA_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return Path.home().joinpath("LMUData").resolve()


def safe_stem(value: Any, fallback: str) -> str:
    raw = str(value or fallback)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return stem or fallback


def parse_list_field(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [stripped]
    return [value]


def save_image(image: Image.Image, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")


def build_rows(
    *,
    dataset_name: str,
    config_name: str,
    split: str,
    output_root: Path,
    vlmeval_name: str,
    max_samples: int,
    overwrite_images: bool,
    streaming: bool,
) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    image_root = output_root / "images" / vlmeval_name
    rows: list[dict[str, Any]] = []
    total = max_samples if max_samples > 0 else None

    for pos, record in enumerate(tqdm(iter(dataset), total=total, desc=vlmeval_name)):
        if max_samples > 0 and pos >= max_samples:
            break

        sample_id = str(record.get("id") or pos)
        options = [str(item).strip() for item in parse_list_field(record.get("options"))]
        if len(options) < 2 or len(options) > len(LETTERS):
            raise ValueError(f"{sample_id} has unsupported option count {len(options)}: {options!r}")

        image_paths: list[str] = []
        for image_idx, key in enumerate(IMAGE_KEYS, start=1):
            image = record.get(key)
            if image is None:
                continue
            image_path = image_root / f"{safe_stem(sample_id, str(pos))}_{image_idx}.png"
            save_image(image, image_path, overwrite=overwrite_images)
            image_paths.append(str(image_path))
        if not image_paths:
            raise ValueError(f"{sample_id} has no images")

        row: dict[str, Any] = {
            "index": sample_id,
            "id": sample_id,
            "question": str(record.get("question", "")).strip(),
            "image_path": repr(image_paths),
            "answer": str(record.get("answer", "")).strip().upper(),
            "category": str(record.get("subject", "")).strip(),
            "subject": str(record.get("subject", "")).strip(),
            "topic_difficulty": str(record.get("topic_difficulty", "")).strip(),
            "img_type": json.dumps(parse_list_field(record.get("img_type")), ensure_ascii=False),
        }
        for idx, option in enumerate(options):
            row[LETTERS[idx]] = option
        explanation = record.get("explanation")
        if explanation is not None:
            row["explanation"] = str(explanation)
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows generated")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MMMU/MMMU_Pro standard (4 options) as a VLMEvalKit MMMU_Pro_4c TSV."
    )
    parser.add_argument("--dataset-name", default="MMMU/MMMU_Pro")
    parser.add_argument("--config-name", default="standard (4 options)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default=None, help="Defaults to $LMUData or ~/LMUData")
    parser.add_argument("--vlmeval-name", default="MMMU_Pro_4c")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full split")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--hard-exit",
        action="store_true",
        help="Use os._exit(0) after flushing output; useful on environments with pyarrow finalizer crashes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else lmu_data_root()
    output_root.mkdir(parents=True, exist_ok=True)
    data = build_rows(
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        split=args.split,
        output_root=output_root,
        vlmeval_name=args.vlmeval_name,
        max_samples=args.max_samples,
        overwrite_images=args.overwrite,
        streaming=not args.no_streaming,
    )
    tsv_path = output_root / f"{args.vlmeval_name}.tsv"
    if tsv_path.exists() and not args.overwrite:
        raise FileExistsError(f"{tsv_path} exists; pass --overwrite to replace it")
    data.to_csv(tsv_path, sep="\t", index=False)
    summary = {
        "vlmeval_dataset": args.vlmeval_name,
        "rows": int(len(data)),
        "tsv": str(tsv_path),
        "image_root": str(output_root / "images" / args.vlmeval_name),
        "subjects": sorted(set(str(x) for x in data["subject"])),
        "answer_counts": data["answer"].value_counts().sort_index().to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
