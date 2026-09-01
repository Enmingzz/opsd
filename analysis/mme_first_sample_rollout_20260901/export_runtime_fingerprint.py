#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


THINKING_PROMPT = (
    "First output the thinking process in <think> </think> tags and then output "
    "the final answer in <answer> </answer> tags."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an exact preprocessing and VisionZip-selection fingerprint."
    )
    parser.add_argument(
        "--mme-tsv",
        type=Path,
        default=Path("/scratch/enmingzz/vlmevalkit_data/MME.tsv"),
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/scratch/enmingzz/.cache/huggingface/"
            "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
            "cc594898137f460bfe9f0759e9844b3ce807cfb5"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visionzip-prune-ratio", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256_bytes(header + b"\0" + contiguous.tobytes(order="C"))


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_npy_dump(array: np.ndarray, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        temporary = Path(handle.name)
    temporary.replace(path)
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "content_sha256": array_content_sha256(array),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_row(path: Path, row_number: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            if number == row_number:
                return row
    raise IndexError(f"MME row {row_number} not found in {path}")


def module_info(module: Any) -> dict[str, Any]:
    return {
        "version": getattr(module, "__version__", None),
        "path": str(Path(module.__file__).resolve()),
    }


def index_to_coordinate(index: int, grid_thw: list[int], merge_size: int) -> list[int]:
    grid_t, grid_h, grid_w = grid_thw
    merged_h = grid_h // merge_size
    merged_w = grid_w // merge_size
    plane = merged_h * merged_w
    t = index // plane
    remainder = index % plane
    return [t, remainder // merged_w, remainder % merged_w]


def normalize_rollout(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"prediction": str(result), "raw_prediction": str(result)}
    extra = result.get("extra_records", {})
    return {
        "prediction": result.get("prediction"),
        "raw_prediction": extra.get("raw_prediction", result.get("prediction")),
        "parsed_prediction": extra.get("parsed_prediction", result.get("prediction")),
        "generated_token_len": extra.get("generated_token_len"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLMEVAL_SAVE_RAW_PREDICTION"] = "1"

    import accelerate
    import flash_attn
    import huggingface_hub
    import peft
    import PIL
    import pyarrow
    import qwen_vl_utils
    import tokenizers
    import transformers
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

    row = read_row(args.mme_tsv, args.row)
    image_bytes = base64.b64decode(row["image"])
    image_path = args.output_dir / f"MME_index_{int(row['index']):04d}.jpg"
    image_path.write_bytes(image_bytes)

    torch.manual_seed(0)
    chat = Qwen2VLChat(
        model_path=str(args.model_path),
        min_pixels=1280 * 28 * 28,
        max_pixels=4096 * 28 * 28,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        use_kv_cache=True,
        enable_visionzip=True,
        visionzip_ratio=args.visionzip_prune_ratio,
        num_return_sequences=1,
        use_custom_prompt=False,
        enable_thinking=True,
        adapter_path="",
    )

    captured_processor: dict[str, torch.Tensor] = {}
    captured_visual: dict[str, torch.Tensor] = {}
    captured_topk: list[dict[str, torch.Tensor | int]] = []

    processor_class = type(chat.processor)
    original_processor_call = processor_class.__call__
    original_topk = torch.topk

    def processor_call(instance: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        output = original_processor_call(instance, *call_args, **call_kwargs)
        if instance is chat.processor and "pixel_values" in output:
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    captured_processor[key] = value.detach().cpu().contiguous().clone()
        return output

    def visual_hook(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) < 3 or output[1] is None:
            return
        captured_visual["image_embeds"] = output[0].detach().cpu().contiguous().clone()
        captured_visual["attention_scores"] = output[1].detach().float().cpu().contiguous().clone()
        captured_visual["attention_keys"] = output[2].detach().float().cpu().contiguous().clone()

    def topk_capture(input_tensor: torch.Tensor, k: int, *call_args: Any, **call_kwargs: Any) -> Any:
        result = original_topk(input_tensor, k, *call_args, **call_kwargs)
        if input_tensor.ndim == 1 and input_tensor.numel() >= 100 and int(k) > 1:
            captured_topk.append(
                {
                    "input": input_tensor.detach().float().cpu().contiguous().clone(),
                    "k": int(k),
                    "values": result.values.detach().float().cpu().contiguous().clone(),
                    "indices": result.indices.detach().cpu().contiguous().clone(),
                }
            )
        return result

    processor_class.__call__ = processor_call
    torch.topk = topk_capture
    hook = chat.model.model.visual.register_forward_hook(visual_hook)
    try:
        message = [
            {"type": "image", "value": str(image_path)},
            {"type": "text", "value": row["question"]},
        ]
        torch.manual_seed(0)
        rollout = chat.generate_inner(deepcopy(message), dataset="MME")
    finally:
        hook.remove()
        torch.topk = original_topk
        processor_class.__call__ = original_processor_call

    required_processor = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    missing_processor = sorted(required_processor - captured_processor.keys())
    if missing_processor:
        raise RuntimeError(f"Processor capture missing tensors: {missing_processor}")
    if not captured_visual:
        raise RuntimeError("Vision forward hook captured no output")

    image_grid = captured_processor["image_grid_thw"][0].tolist()
    merge_size = int(chat.model.config.vision_config.spatial_merge_size)
    visual_count = int(np.prod(image_grid) // (merge_size**2))
    dominant_num = int((1.0 - args.visionzip_prune_ratio) * visual_count)
    contextual_num = max(int(0.05 * visual_count), 1)
    candidates = [
        record
        for record in captured_topk
        if record["input"].numel() == visual_count and record["k"] == dominant_num
    ]
    if len(candidates) != 1:
        observed = [(record["input"].numel(), record["k"]) for record in captured_topk]
        raise RuntimeError(
            f"Expected one VisionZip top-k call ({visual_count}, {dominant_num}), "
            f"found {len(candidates)}; observed={observed}"
        )

    selection = candidates[0]
    attention_scores = selection["input"]
    dominant_ranked = selection["indices"].to(torch.long)
    select_mask = torch.zeros(visual_count, dtype=torch.bool)
    select_mask[dominant_ranked] = True
    non_dominant = torch.nonzero(~select_mask, as_tuple=True)[0]
    step = max(1, non_dominant.numel() // contextual_num)
    contextual_relative = torch.arange(0, non_dominant.numel(), step)[:contextual_num]
    contextual_original = non_dominant[contextual_relative]
    select_mask[contextual_original] = True
    retained = torch.nonzero(select_mask, as_tuple=True)[0]
    dropped = torch.nonzero(~select_mask, as_tuple=True)[0]

    attention_keys = captured_visual["attention_keys"]
    dominant_mask = torch.zeros(visual_count, dtype=torch.bool)
    dominant_mask[dominant_ranked] = True
    filtered_keys = attention_keys[:, ~dominant_mask]
    normalized_keys = filtered_keys / filtered_keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    target_keys = normalized_keys[:, contextual_relative, :]
    relative_positions = torch.arange(normalized_keys.shape[1])
    merge_source_relative = relative_positions[~torch.isin(relative_positions, contextual_relative)]
    merge_source_keys = normalized_keys[:, merge_source_relative, :]
    similarities = torch.bmm(merge_source_keys, target_keys.transpose(1, 2))
    merge_assignment = similarities.argmax(dim=2)[0]
    merge_source_original = non_dominant[merge_source_relative]
    merge_groups = []
    for target_number, anchor in enumerate(contextual_original.tolist()):
        members = merge_source_original[merge_assignment == target_number].tolist()
        merge_groups.append(
            {
                "anchor_index": int(anchor),
                "anchor_coordinate": index_to_coordinate(int(anchor), image_grid, merge_size),
                "merged_source_indices": [int(value) for value in members],
            }
        )

    arrays: dict[str, np.ndarray] = {
        "pixel_values": captured_processor["pixel_values"].numpy(),
        "image_grid_thw": captured_processor["image_grid_thw"].numpy(),
        "input_ids": captured_processor["input_ids"].numpy(),
        "attention_mask": captured_processor["attention_mask"].numpy(),
        "vision_attention_scores": attention_scores.numpy(),
        "dominant_indices_ranked": dominant_ranked.numpy(),
        "dominant_indices_model_order": dominant_ranked.sort().values.numpy(),
        "contextual_anchor_indices": contextual_original.numpy(),
        "retained_indices_model_order": retained.numpy(),
        "dropped_indices_model_order": dropped.numpy(),
        "contextual_merge_assignment": merge_assignment.numpy(),
        "contextual_merge_source_indices": merge_source_original.numpy(),
    }
    array_manifest = {
        name: atomic_npy_dump(array, args.output_dir / f"{name}.npy")
        for name, array in arrays.items()
    }

    processor_config = chat.processor.image_processor.to_dict()
    atomic_json_dump(processor_config, args.output_dir / "image_processor_config.json")
    rendered_prompt_ids = captured_processor["input_ids"][0].tolist()
    rendered_prompt = chat.processor.tokenizer.decode(
        rendered_prompt_ids, skip_special_tokens=False
    )
    (args.output_dir / "rendered_prompt.txt").write_text(rendered_prompt, encoding="utf-8")

    vlm_root = Path(os.environ.get("VLM_ROOT", ".")).resolve()
    model_source = (
        vlm_root
        / "transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"
    )
    flash_source = vlm_root / "transformers/src/transformers/integrations/flash_attention.py"
    wrapper_source = vlm_root / "vlmeval/vlm/qwen2_vl/model.py"
    pixel_values = arrays["pixel_values"]
    payload = {
        "schema_version": 1,
        "sample": {
            "dataset": "MME",
            "row_number": args.row,
            "index": int(row["index"]),
            "category": row.get("category"),
            "question": row["question"],
            "ground_truth": row.get("answer"),
            "image_path": image_path.name,
            "image_sha256": sha256_bytes(image_bytes),
        },
        "model": {
            "path": str(args.model_path.resolve()),
            "revision": args.model_path.name,
            "adapter": None,
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": module_info(torch),
            "torch_cuda": torch.version.cuda,
            "transformers": module_info(transformers),
            "tokenizers": module_info(tokenizers),
            "huggingface_hub": module_info(huggingface_hub),
            "qwen_vl_utils": module_info(qwen_vl_utils),
            "PIL": module_info(PIL),
            "flash_attn": module_info(flash_attn),
            "accelerate": module_info(accelerate),
            "peft": module_info(peft),
            "pyarrow": module_info(pyarrow),
            "vlmevalkit_commit": git_commit(vlm_root),
            "source_sha256": {
                "modeling_qwen2_5_vl.py": sha256_file(model_source),
                "flash_attention.py": sha256_file(flash_source),
                "vlmeval_qwen_model.py": sha256_file(wrapper_source),
            },
        },
        "evaluation": {
            "prompt_suffix": THINKING_PROMPT,
            "min_pixels": 1280 * 28 * 28,
            "max_pixels": 4096 * 28 * 28,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
            "seed": 0,
            "use_kv_cache": True,
            "visionzip_prune_ratio_argument": args.visionzip_prune_ratio,
            "target_retention_ratio": 1.0 - args.visionzip_prune_ratio + 0.05,
        },
        "preprocessing": {
            "processor_class": type(chat.processor).__name__,
            "image_processor_class": type(chat.processor.image_processor).__name__,
            "tokenizer_class": type(chat.processor.tokenizer).__name__,
            "image_grid_thw": image_grid,
            "spatial_merge_size": merge_size,
            "pixel_values_summary": {
                "shape": list(pixel_values.shape),
                "dtype": str(pixel_values.dtype),
                "min": float(pixel_values.min()),
                "max": float(pixel_values.max()),
                "mean": float(pixel_values.mean(dtype=np.float64)),
                "std": float(pixel_values.std(dtype=np.float64)),
            },
        },
        "visionzip": {
            "initial_visual_token_count": visual_count,
            "dominant_count": dominant_num,
            "contextual_count": contextual_num,
            "retained_count": int(retained.numel()),
            "dropped_count": int(dropped.numel()),
            "merged_grid_thw": [
                image_grid[0],
                image_grid[1] // merge_size,
                image_grid[2] // merge_size,
            ],
            "dominant_indices_ranked": dominant_ranked.tolist(),
            "dominant_indices_model_order": dominant_ranked.sort().values.tolist(),
            "contextual_anchor_indices": contextual_original.tolist(),
            "retained_indices_model_order": retained.tolist(),
            "dropped_indices_model_order": dropped.tolist(),
            "retained_coordinates": [
                index_to_coordinate(int(index), image_grid, merge_size)
                for index in retained.tolist()
            ],
            "contextual_merge_groups": merge_groups,
        },
        "rollout": normalize_rollout(rollout),
        "arrays": array_manifest,
    }
    atomic_json_dump(payload, args.output_dir / "fingerprint.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Saved runtime fingerprint to {args.output_dir}")


if __name__ == "__main__":
    main()

