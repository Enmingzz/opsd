#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import huggingface_hub
import PIL
import PIL._imaging
import pyarrow
import tokenizers
import torch
import transformers


EXPECTED = {
    "PIL": "9.5.0.post2",
    "huggingface_hub": "0.34.3",
    "pyarrow": "23.0.1",
    "tokenizers": "0.22.2",
    "torch": "2.9.1",
    "transformers": "4.57.0",
}
EXPECTED_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
WHEEL_NAME = (
    "Pillow_SIMD-9.5.0.post2+computecanada-"
    "cp311-cp311-linux_x86_64.whl"
)
WHEEL_SHA256 = "df0d6f51f815ef94d1d513ebc751b34e93f5b68345b41b9016c315239c8c4f24"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pillow_files(wheel: Path) -> dict[str, int]:
    site_packages = Path(PIL.__file__).resolve().parent.parent
    matched = mismatched = missing = 0
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("PIL/") and not name.endswith("/")
        ]
        for name in names:
            installed = site_packages / name
            if not installed.is_file():
                missing += 1
            elif hashlib.sha256(archive.read(name)).digest() == hashlib.sha256(
                installed.read_bytes()
            ).digest():
                matched += 1
            else:
                mismatched += 1
    return {"matched": matched, "mismatched": mismatched, "missing": missing}


def cached_model_revision() -> str | None:
    model_key = "models--Qwen--Qwen2.5-VL-7B-Instruct"
    candidates = []
    if os.environ.get("TRANSFORMERS_CACHE"):
        candidates.append(Path(os.environ["TRANSFORMERS_CACHE"]) / model_key)
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "hub" / model_key)
    for root in candidates:
        ref = root / "refs" / "main"
        if ref.is_file():
            return ref.read_text().strip()
    return None


def main() -> int:
    bundle = Path(__file__).resolve().parent
    wheel = bundle / "wheels" / WHEEL_NAME
    versions = {
        "PIL": PIL.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "pyarrow": pyarrow.__version__,
        "tokenizers": tokenizers.__version__,
        "torch": torch.__version__.split("+")[0],
        "transformers": transformers.__version__,
    }
    wheel_hash = sha256(wheel) if wheel.is_file() else None
    pillow_files = verify_pillow_files(wheel) if wheel.is_file() else None
    model_revision = cached_model_revision()

    checks = {
        "versions": versions == EXPECTED,
        "wheel_sha256": wheel_hash == WHEEL_SHA256,
        "pillow_files": pillow_files == {
            "matched": 102,
            "mismatched": 0,
            "missing": 0,
        },
        "model_revision": model_revision in (None, EXPECTED_MODEL_REVISION),
    }
    result = {
        "checks": checks,
        "expected_versions": EXPECTED,
        "actual_versions": versions,
        "python": sys.version,
        "pillow_path": str(Path(PIL.__file__).resolve()),
        "pillow_imaging_path": str(Path(PIL._imaging.__file__).resolve()),
        "pillow_wheel_sha256": wheel_hash,
        "pillow_file_comparison": pillow_files,
        "cached_model_revision": model_revision,
        "expected_model_revision": EXPECTED_MODEL_REVISION,
        "transformers_path": str(Path(transformers.__file__).resolve()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
