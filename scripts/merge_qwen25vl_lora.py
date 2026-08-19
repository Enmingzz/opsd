#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opsd.scripts.train_qwen25vl_prune_distill import (
    bootstrap_qwen25,
    import_qwen25_modules,
    resolve_attn_implementation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument(
        "--safe_merge",
        action="store_true",
        help="Use PEFT's checked dtype-preserving LoRA merge path.",
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_model = Path(args.base_model)
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output_dir is not empty: {output_dir}; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_qwen25()
    AutoProcessor, model_cls = import_qwen25_modules()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    attn_impl = resolve_attn_implementation(args.attn_implementation)

    kwargs = {"torch_dtype": dtype, "attn_implementation": attn_impl}
    if args.device_map and str(args.device_map).lower() != "none" and torch.cuda.is_available():
        kwargs["device_map"] = args.device_map

    try:
        model = model_cls.from_pretrained(str(base_model), **kwargs)
    except Exception:
        if attn_impl == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
            model = model_cls.from_pretrained(str(base_model), **kwargs)
        else:
            raise

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload(safe_merge=args.safe_merge)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    processor = AutoProcessor.from_pretrained(str(base_model))
    processor.save_pretrained(str(output_dir))

    metadata = {
        "base_model": str(base_model),
        "adapter_path": str(adapter_path),
        "output_dir": str(output_dir),
        "dtype": str(dtype),
        "attn_implementation_requested": args.attn_implementation,
        "attn_implementation_used": kwargs.get("attn_implementation", attn_impl),
        "merge_semantics": (
            "peft_merge_and_unload_safe" if args.safe_merge
            else "peft_merge_and_unload_default"
        ),
        "merged_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": args.note,
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
