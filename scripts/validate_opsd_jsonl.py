#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DISALLOWED_QWEN25_BOOTSTRAP = "/scratch/enmingzz/temp/qwen25_bootstrap"
sys.path = [path for path in sys.path if not path or not path.startswith(DISALLOWED_QWEN25_BOOTSTRAP)]

from PIL import Image


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--image_root", default="")
    p.add_argument("--limit", type=int, default=0)
    return p


def resolve_image(path_value: Any, image_root: str) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute() and image_root:
        path = Path(image_root) / path
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    seen: set[str] = set()
    errors: list[dict[str, Any]] = []
    count = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if args.limit and count >= args.limit:
                break
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append({"line": line_no, "error": f"invalid json: {exc}"})
                continue
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                errors.append({"line": line_no, "error": "missing sample_id"})
            elif sample_id in seen:
                errors.append({"line": line_no, "sample_id": sample_id, "error": "duplicate sample_id"})
            seen.add(sample_id)

            if not str(row.get("question", "")).strip():
                errors.append({"line": line_no, "sample_id": sample_id, "error": "empty question"})
            if "answer" not in row:
                errors.append({"line": line_no, "sample_id": sample_id, "warning": "missing optional answer"})

            image_path = resolve_image(row.get("image"), args.image_root)
            if image_path is None or not image_path.exists():
                errors.append({"line": line_no, "sample_id": sample_id, "error": "missing image"})
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                errors.append({"line": line_no, "sample_id": sample_id, "error": f"PIL open failed: {exc}"})

    fatal = [e for e in errors if "error" in e]
    print(json.dumps({"num_records": count, "num_errors": len(fatal), "num_warnings": len(errors) - len(fatal)}, indent=2))
    for item in errors[:50]:
        print(json.dumps(item, ensure_ascii=False))
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
