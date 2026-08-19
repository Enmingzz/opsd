#!/usr/bin/env python3
"""Validate and record the checkpoint loading mode for an evaluation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NEW_EVAL_LOAD_MODES = (
    "merged_full_checkpoint",
    "unmerged_peft_adapter",
    "standalone_base",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument(
        "--load-mode",
        choices=NEW_EVAL_LOAD_MODES,
        default="merged_full_checkpoint",
        help="Defaults to merged for trained-checkpoint evaluations.",
    )
    parser.add_argument("--allow-unmerged", action="store_true")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--method", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--ratio", default="")
    return parser.parse_args()


def env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y"}


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")


def find_model_weights(model_path: Path) -> list[Path]:
    index = model_path / "model.safetensors.index.json"
    single = model_path / "model.safetensors"
    shards = sorted(model_path.glob("model-*.safetensors"))
    return [path for path in (index, single, *shards) if path.is_file()]


def find_adapter_weights(adapter_path: Path) -> list[Path]:
    return [
        path
        for path in (
            adapter_path / "adapter_model.safetensors",
            adapter_path / "adapter_model.bin",
        )
        if path.is_file()
    ]


def validate(args: argparse.Namespace) -> dict[str, Any]:
    model_text = args.model_path.strip()
    adapter_text = args.adapter_path.strip()
    if not model_text:
        raise ValueError("model_path must not be empty")

    evidence: list[str] = []
    merge_metadata: dict[str, Any] | None = None

    if args.load_mode == "merged_full_checkpoint":
        if adapter_text:
            raise ValueError("Merged evaluation requires an empty adapter_path")
        model_path = Path(model_text).expanduser()
        if not model_path.is_dir():
            raise ValueError(f"Merged model_path must be a local directory: {model_path}")
        require_file(model_path / "config.json", "merged config.json")
        require_file(model_path / "merge_metadata.json", "merge provenance")
        weights = find_model_weights(model_path)
        if not weights:
            raise ValueError(f"No full-model safetensor weights found in {model_path}")
        merge_metadata = json.loads((model_path / "merge_metadata.json").read_text(encoding="utf-8"))
        for key in ("base_model", "adapter_path"):
            if not str(merge_metadata.get(key, "")).strip():
                raise ValueError(f"merge_metadata.json is missing {key}: {model_path}")
        evidence.extend(("adapter_path_empty", "local_full_model_weights", "merge_metadata"))

    elif args.load_mode == "unmerged_peft_adapter":
        if not (args.allow_unmerged or env_true("ALLOW_UNMERGED_EVAL")):
            raise ValueError(
                "Unmerged adapter evaluation is disabled by default; pass --allow-unmerged "
                "or set ALLOW_UNMERGED_EVAL=1 for an explicitly requested comparison"
            )
        if not adapter_text:
            raise ValueError("Unmerged evaluation requires a non-empty adapter_path")
        adapter_path = Path(adapter_text).expanduser()
        if not adapter_path.is_dir():
            raise ValueError(f"adapter_path is not a local directory: {adapter_path}")
        require_file(adapter_path / "adapter_config.json", "adapter_config.json")
        if not find_adapter_weights(adapter_path):
            raise ValueError(f"No PEFT adapter weights found in {adapter_path}")
        evidence.extend(("explicit_unmerged_override", "peft_adapter_config", "peft_adapter_weights"))

    else:
        if adapter_text:
            raise ValueError("standalone_base evaluation requires an empty adapter_path")
        evidence.append("adapter_path_empty")
        local_path = Path(model_text).expanduser()
        if local_path.exists():
            if not local_path.is_dir():
                raise ValueError(f"Local standalone model_path is not a directory: {local_path}")
            require_file(local_path / "config.json", "standalone config.json")
            if not find_model_weights(local_path):
                raise ValueError(f"No full-model safetensor weights found in {local_path}")
            evidence.append("local_full_model_weights")
        else:
            evidence.append("remote_or_cached_model_id")

    return {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_load_form": args.load_mode,
        "model_path": model_text,
        "adapter_path": adapter_text,
        "method": args.method,
        "dataset": args.dataset,
        "ratio": args.ratio,
        "validation_evidence": evidence,
        "merge_metadata": merge_metadata,
    }


def main() -> None:
    args = parse_args()
    result = validate(args)
    rendered = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
