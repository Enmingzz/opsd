#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OPSD_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = OPSD_ROOT.parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config.json"

PRIMARY_METRICS = (
    "kl_r025_official_to_r020",
    "kl_base_full_to_r020",
    "kl_base_full_to_r025_official",
)
SELF_CHECKPOINT_METRICS = (
    "kl_r025_official_to_r020",
    "kl_checkpoint_full_to_r020",
    "kl_checkpoint_full_to_r025_official",
)
ALL_METRICS = PRIMARY_METRICS + (
    "js_r020_r025_official",
    "kl_r025_nested_to_r020",
    "js_r020_r025_nested",
    "kl_base_full_to_r025_nested",
) + SELF_CHECKPOINT_METRICS[1:] + (
    "kl_checkpoint_full_to_r025_nested",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure configurable low/high-retention and base-full KL for one checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Checkpoint adapter directory, a path relative to checkpoint_root, or __BASE__ for step 0.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument(
        "--prefix-source-dir",
        type=Path,
        help=(
            "Optional samples/ directory from a completed matching run. When set, "
            "reuse its checkpoint-generated token IDs instead of regenerating."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--merge-student",
        action="store_true",
        help=(
            "Merge the LoRA adapter into the student for inference. A separate "
            "adapter-free base model is loaded for full-token teacher scoring."
        ),
    )
    parser.add_argument(
        "--full-reference",
        choices=("base_teacher", "self_checkpoint"),
        default="base_teacher",
        help=(
            "Full-token distribution used as q_full. base_teacher preserves the "
            "historical adapter-free OPSD teacher. self_checkpoint uses the exact "
            "same checkpoint weights as the low/high-retention student branches."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(sample_id: str) -> str:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]
    return f"{digest}.json"


def capture(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=OPSD_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def tensor_stats(values: Any) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def resolve_adapter(raw: str, checkpoint_root: Path) -> Path | None:
    if raw == "__BASE__":
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = checkpoint_root / candidate
    candidate = candidate.resolve()
    if not (candidate / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter checkpoint: {candidate}")
    if not (candidate / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter weights: {candidate}")
    return candidate


def make_sample(row: dict[str, Any], sample_cls: Any) -> Any:
    option_map = row.get("options") or {}
    option_letters = sorted(option_map)
    options = [str(option_map[key]) for key in option_letters]
    correct_letter = str(row.get("original_answer_letter") or "")
    correct_index = option_letters.index(correct_letter) if correct_letter in option_letters else 0
    return sample_cls(
        sample_id=str(row["sample_id"]),
        image=str(row["image_path"]),
        question=str(row["question"]),
        options=options,
        correct_index=correct_index,
        correct_letter=correct_letter,
        reasoning="",
        prompt=str(row["prompt"]),
        target=str(row.get("answer", "")),
        raw=row,
    )


def load_model(
    cfg: dict[str, Any],
    adapter_path: Path | None,
    *,
    merge_student: bool = False,
) -> tuple[Any, Any]:
    import torch
    from opsd.visionzip_aokvqa.qwen_wrapper import apply_lora, load_qwen_model_and_processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, processor = load_qwen_model_and_processor(
        str(cfg["base_model"]),
        bf16=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        min_pixels=int(cfg["min_pixels"]),
        max_pixels=int(cfg["max_pixels"]),
        visionzip_official=True,
    )
    if adapter_path is None and not merge_student:
        # The original run starts from a fresh no-op LoRA. LoRA B is zero, so
        # this is functionally identical to the unadapted base model while
        # preserving the same PEFT execution surface as later checkpoints.
        model = apply_lora(
            model,
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=list(range(28)),
            layers_pattern="layers",
        )
    elif adapter_path is not None:
        model = apply_lora(model, adapter_path=str(adapter_path))
    if merge_student and adapter_path is not None:
        if not hasattr(model, "merge_and_unload"):
            raise RuntimeError("Loaded PEFT model does not support merge_and_unload().")
        model = model.merge_and_unload(safe_merge=True)
    if merge_student:
        model.requires_grad_(False)
    model.eval()
    return model, processor


def load_base_teacher(cfg: dict[str, Any]) -> Any:
    """Load the immutable adapter-free teacher used with a merged student."""
    from opsd.visionzip_aokvqa.qwen_wrapper import load_qwen_model_and_processor

    teacher, _ = load_qwen_model_and_processor(
        str(cfg["base_model"]),
        bf16=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        min_pixels=int(cfg["min_pixels"]),
        max_pixels=int(cfg["max_pixels"]),
        visionzip_official=True,
    )
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def visionzip_backend_kwargs(retention_ratio: float, contextual_ratio: float = 0.05) -> dict[str, Any]:
    from opsd.visionzip_aokvqa.qwen_wrapper import disabled_pruning_kwargs

    retention = min(max(float(retention_ratio), 0.0), 1.0)
    contextual = min(float(contextual_ratio), retention)
    dominant = max(0.0, retention - contextual)
    kwargs = disabled_pruning_kwargs()
    kwargs.update({"enable_visionzip": True, "visionzip_ratio": 1.0 - dominant})
    return kwargs


def actual_visionzip_metadata(
    model: Any,
    inputs: dict[str, Any],
    prompt_len: int,
    retention_ratio: float,
    contextual_ratio: float = 0.05,
) -> dict[str, Any]:
    from opsd.pruning_distill.qwen25_pruned_forward import _unwrap_qwen_model
    from opsd.visionzip_aokvqa.qwen_wrapper import last_visionzip_pruned_inputs

    qwen = _unwrap_qwen_model(model)
    last = last_visionzip_pruned_inputs(model)
    if not isinstance(last, dict) or last.get("input_ids") is None:
        raise RuntimeError("VisionZip backend did not expose its actual pruned input IDs.")
    image_token_id = int(qwen.config.image_token_id)
    full_count = int((inputs["input_ids"] == image_token_id).sum().item())
    kept_count = int((last["input_ids"] == image_token_id).sum().item())
    backend = visionzip_backend_kwargs(retention_ratio, contextual_ratio)
    # Match the clean Armen implementation exactly. In particular, 1 - 0.8
    # can be represented just below 0.2, so floor() may differ by one token
    # from int(0.20 * N). The actual backend count is authoritative.
    backend_dominant = int((1.0 - float(backend["visionzip_ratio"])) * full_count)
    contextual_count = max(int(float(contextual_ratio) * full_count), 1)
    if backend_dominant + contextual_count != kept_count:
        raise RuntimeError(
            "Could not reconcile actual VisionZip count: "
            f"dominant={backend_dominant}, contextual={contextual_count}, kept={kept_count}."
        )
    student_prompt_len = int(prompt_len) - full_count + kept_count
    return {
        "student_prompt_len": student_prompt_len,
        "num_full_visual_tokens": full_count,
        "num_kept_visual_tokens": kept_count,
        "visionzip_dominant_tokens": backend_dominant,
        "visionzip_contextual_tokens": contextual_count,
        "visionzip_ratio_backend": float(backend["visionzip_ratio"]),
        "requested_retention_ratio": float(retention_ratio),
        "realized_retention_ratio": float(kept_count / full_count),
        "metric_source": "clean_armen_actual_pruned_input_ids",
    }


def generate_official_pruned(
    model: Any,
    processor: Any,
    prompt_inputs: dict[str, Any],
    retention_ratio: float,
    max_new_tokens: int,
    contextual_ratio: float,
) -> tuple[Any, str, dict[str, Any]]:
    import torch
    from opsd.visionzip_aokvqa.qwen_wrapper import model_input_subset

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or eos_token_id
    kwargs = {
        **model_input_subset(prompt_inputs),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        **visionzip_backend_kwargs(retention_ratio, contextual_ratio),
    }
    with torch.inference_mode():
        output_ids = model.generate(**kwargs)
    generated_ids = output_ids[:, prompt_len:]
    text = processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    metadata = actual_visionzip_metadata(
        model,
        prompt_inputs,
        prompt_len,
        retention_ratio,
        contextual_ratio,
    )
    return generated_ids, text, metadata


def score_full(model: Any, inputs: dict[str, Any], prompt_len: int, count: int) -> Any:
    import torch
    from opsd.visionzip_aokvqa.qwen_wrapper import extract_generated_logits, model_input_subset

    with torch.inference_mode():
        outputs = model(**model_input_subset(inputs), use_cache=False)
        logits = extract_generated_logits(outputs.logits, int(prompt_len), int(count)).detach()
    del outputs
    return logits


def score_official_pruned(
    model: Any,
    inputs: dict[str, Any],
    prompt_len: int,
    count: int,
    ratio: float,
    contextual_ratio: float,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from opsd.visionzip_aokvqa.qwen_wrapper import extract_generated_logits, model_input_subset

    with torch.inference_mode():
        outputs = model(
            **model_input_subset(inputs),
            use_cache=False,
            **visionzip_backend_kwargs(ratio, contextual_ratio),
        )
        metadata = actual_visionzip_metadata(
            model,
            inputs,
            prompt_len,
            ratio,
            contextual_ratio,
        )
        logits = extract_generated_logits(
            outputs.logits,
            int(metadata["student_prompt_len"]),
            int(count),
        ).detach()
    del outputs
    return logits, metadata


def build_actual_standard_mask(scores: Any, dominant_count: int, contextual_count: int, name: str) -> tuple[Any, Any]:
    import torch
    from opsd.algorithm1.src.mask_variants import VisionZipMask

    scores = scores.float().flatten()
    num_tokens = int(scores.numel())
    ranking = torch.argsort(scores, descending=True, stable=True)
    dominant = torch.topk(scores, k=int(dominant_count), largest=True, sorted=False).indices
    dominant_mask = torch.zeros(num_tokens, dtype=torch.bool, device=scores.device)
    dominant_mask[dominant] = True
    non_dominant = (~dominant_mask).nonzero(as_tuple=True)[0]
    step = max(1, int(non_dominant.numel()) // int(contextual_count))
    positions = torch.arange(0, int(non_dominant.numel()), step, device=scores.device)[: int(contextual_count)]
    contextual = non_dominant.index_select(0, positions)
    mask = VisionZipMask(
        dominant_indices=dominant.unique(sorted=True),
        contextual_indices=contextual.unique(sorted=True),
        full_visual_token_count=num_tokens,
        name=name,
    )
    if mask.count != int(dominant_count) + int(contextual_count):
        raise AssertionError("Actual VisionZip mask reconstruction changed the token count.")
    return mask, ranking


def build_nested_to_actual_budget(base: Any, ranking: Any, target_count: int) -> Any:
    import torch
    from opsd.algorithm1.src.mask_variants import VisionZipMask

    retained = set(base.retained_indices.tolist())
    additions_needed = int(target_count) - len(retained)
    if additions_needed < 0:
        raise ValueError("The high-retention target budget is smaller than the low-retention budget.")
    additions = [index for index in ranking.tolist() if index not in retained][:additions_needed]
    if len(additions) != additions_needed:
        raise RuntimeError("Could not construct the nested high-retention mask at the official token budget.")
    dominant = torch.tensor(
        sorted(set(base.dominant_indices.tolist()) | set(additions)),
        dtype=torch.long,
        device=ranking.device,
    )
    nested = VisionZipMask(
        dominant_indices=dominant,
        contextual_indices=base.contextual_indices.clone(),
        full_visual_token_count=int(base.full_visual_token_count),
        name="M25_nested_actual_budget",
    )
    if nested.count != int(target_count) or not retained.issubset(set(nested.retained_indices.tolist())):
        raise AssertionError("Invalid nested high-retention mask.")
    return nested


def summarize_checkpoint(output_dir: Path, expected: int) -> dict[str, Any]:
    sample_files = sorted((output_dir / "samples").glob("*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sample_files]
    payload: dict[str, Any] = {
        "completed_samples": len(rows),
        "expected_samples": int(expected),
        "complete": len(rows) == int(expected),
        "metrics": {},
    }
    for metric in ALL_METRICS:
        sample_means = [
            float(row["metric_stats"][metric]["mean"])
            for row in rows
            if metric in row.get("metric_stats", {})
        ]
        pooled = [
            value
            for row in rows
            if metric in row.get("metrics", {})
            for value in row["metrics"][metric]
        ]
        if sample_means:
            payload["metrics"][metric] = {
                "sample_balanced_mean": sum(sample_means) / len(sample_means),
                "token_pooled_mean": sum(pooled) / len(pooled),
            }
    atomic_write_json(output_dir / "checkpoint_summary.json", payload)
    return payload


def run_one(
    model: Any,
    processor: Any,
    sample: Any,
    cfg: dict[str, Any],
    checkpoint_label: str,
    checkpoint_step: int,
    teacher_model: Any | None = None,
    full_reference: str = "base_teacher",
    fixed_prefix_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    from opsd.algorithm1.src.fixed_context import (
        build_fixed_context,
        build_visual_token_bank,
        score_fixed_context,
    )
    from opsd.algorithm1.src.metrics import per_token_forward_kl, per_token_js
    from opsd.visionzip_aokvqa.qwen_wrapper import (
        decode_token_ids,
        encode_prompt,
        primary_device,
        teacher_adapter_disabled,
    )
    from opsd.visionzip_aokvqa.train import (
        sequence_inputs_from_prompt,
        temporary_cached_rollout,
        temporary_eval,
    )

    started = time.perf_counter()
    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root="", device=device)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    rollout_ratio = float(cfg["rollout_retention_ratio"])
    comparison_ratio = float(cfg["comparison_retention_ratio"])
    timings: dict[str, float] = {}

    generated_started = time.perf_counter()
    if fixed_prefix_payload is None:
        with temporary_cached_rollout(model):
            generated_ids, generated_text, generation_metadata = generate_official_pruned(
                model,
                processor,
                prompt_inputs,
                retention_ratio=rollout_ratio,
                max_new_tokens=int(cfg["max_new_tokens"]),
                contextual_ratio=float(cfg["contextual_ratio"]),
            )
        prefix_source = "checkpoint_student_visionzip_low_retention_greedy"
        prefix_provenance = None
    else:
        if str(fixed_prefix_payload.get("sample_id")) != str(sample.sample_id):
            raise AssertionError("Reused prefix sample ID does not match the current sample.")
        if str(fixed_prefix_payload.get("checkpoint_label")) != str(checkpoint_label):
            raise AssertionError("Reused prefix checkpoint label does not match.")
        source_pair = fixed_prefix_payload.get("ratio_pair", {})
        if not abs(float(source_pair.get("low_retention_ratio", -1.0)) - rollout_ratio) < 1e-12:
            raise AssertionError("Reused prefix retention ratio does not match.")
        source_ids = [int(value) for value in fixed_prefix_payload.get("generated_token_ids", [])]
        if not source_ids:
            raise AssertionError("Reused prefix has no generated token IDs.")
        generated_ids = torch.tensor([source_ids], dtype=torch.long, device=device)
        generated_text = str(fixed_prefix_payload.get("generated_text", ""))
        generation_metadata = dict(fixed_prefix_payload.get("generation_metadata", {}))
        prefix_source = "reused_checkpoint_student_visionzip_low_retention_greedy"
        prefix_provenance = {
            "source_schema_version": fixed_prefix_payload.get("schema_version"),
            "source_checkpoint_label": fixed_prefix_payload.get("checkpoint_label"),
            "source_generated_token_count": fixed_prefix_payload.get("generated_token_count"),
        }
    timings["generation_seconds"] = time.perf_counter() - generated_started
    if generated_ids.numel() == 0:
        raise RuntimeError("Greedy low-retention rollout generated zero response tokens.")
    count = int(generated_ids.numel())
    sequence_inputs = sequence_inputs_from_prompt(prompt_inputs, generated_ids)

    low_started = time.perf_counter()
    with temporary_eval(model):
        p20_logits, meta20 = score_official_pruned(
            model, sequence_inputs, prompt_len, count, rollout_ratio, float(cfg["contextual_ratio"])
        )
    timings["low_retention_seconds"] = time.perf_counter() - low_started

    high_started = time.perf_counter()
    with temporary_eval(model):
        p25_official_logits, meta25 = score_official_pruned(
            model, sequence_inputs, prompt_len, count, comparison_ratio, float(cfg["contextual_ratio"])
        )
    timings["high_retention_official_seconds"] = time.perf_counter() - high_started

    bank_started = time.perf_counter()
    with temporary_eval(model):
        bank = build_visual_token_bank(model, sequence_inputs, prompt_len)
        m20, ranking = build_actual_standard_mask(
            bank.attention_scores,
            dominant_count=int(meta20["visionzip_dominant_tokens"]),
            contextual_count=int(meta20["visionzip_contextual_tokens"]),
            name="M20",
        )
        m25_nested = build_nested_to_actual_budget(
            m20,
            ranking,
            target_count=int(meta25["num_kept_visual_tokens"]),
        )
        m20_context = build_fixed_context(model, bank, m20)
        m25_nested_context = build_fixed_context(model, bank, m25_nested)
        with torch.inference_mode():
            p20_fixed_logits = score_fixed_context(model, m20_context, count).detach()
            p25_nested_logits = score_fixed_context(model, m25_nested_context, count).detach()
    timings["fixed_context_seconds"] = time.perf_counter() - bank_started

    equivalence_delta = (p20_fixed_logits.float() - p20_logits.float()).abs()
    equivalence = {
        "allclose": bool(
            torch.allclose(
                p20_fixed_logits.float(),
                p20_logits.float(),
                atol=float(cfg["equivalence_atol"]),
                rtol=float(cfg["equivalence_rtol"]),
            )
        ),
        "max_abs_logit_error": float(equivalence_delta.max()),
        "mean_abs_logit_error": float(equivalence_delta.mean()),
        "atol": float(cfg["equivalence_atol"]),
        "rtol": float(cfg["equivalence_rtol"]),
    }
    if not equivalence["allclose"]:
        raise AssertionError(f"Reconstructed M20 does not match official VisionZip: {equivalence}")

    if int(meta20["num_full_visual_tokens"]) != bank.num_visual_tokens:
        raise AssertionError("Official and fixed-context full visual-token counts differ.")
    if int(meta20["num_kept_visual_tokens"]) != m20.count:
        raise AssertionError("Official and fixed-context low-retention token counts differ.")
    if not set(m20.retained_indices.tolist()).issubset(set(m25_nested.retained_indices.tolist())):
        raise AssertionError("Nested high-retention mask does not contain every low-retention token identity.")

    teacher_started = time.perf_counter()
    if teacher_model is None:
        adapter_context = contextlib.nullcontext()
        if full_reference == "base_teacher" and hasattr(model, "peft_config"):
            adapter_context = teacher_adapter_disabled(model)
        with adapter_context, temporary_eval(model):
            full_reference_logits = score_full(model, sequence_inputs, prompt_len, count)
        teacher_execution = (
            "same_checkpoint_full_visual_tokens"
            if full_reference == "self_checkpoint"
            else "shared_base_model_adapter_disabled"
        )
    else:
        teacher_device = primary_device(teacher_model)
        teacher_inputs = {
            key: value.to(teacher_device) if isinstance(value, torch.Tensor) else value
            for key, value in sequence_inputs.items()
        }
        with temporary_eval(teacher_model):
            full_reference_logits = score_full(teacher_model, teacher_inputs, prompt_len, count)
        full_reference_logits = full_reference_logits.to(device=p20_logits.device)
        del teacher_inputs
        teacher_execution = "independent_adapter_free_base_model"
    timings["base_full_seconds"] = time.perf_counter() - teacher_started

    reference_low_key = (
        "kl_checkpoint_full_to_r020"
        if full_reference == "self_checkpoint"
        else "kl_base_full_to_r020"
    )
    reference_high_key = (
        "kl_checkpoint_full_to_r025_official"
        if full_reference == "self_checkpoint"
        else "kl_base_full_to_r025_official"
    )
    reference_nested_key = (
        "kl_checkpoint_full_to_r025_nested"
        if full_reference == "self_checkpoint"
        else "kl_base_full_to_r025_nested"
    )
    metric_tensors = {
        "kl_r025_official_to_r020": per_token_forward_kl(p25_official_logits, p20_logits),
        reference_low_key: per_token_forward_kl(full_reference_logits, p20_logits),
        reference_high_key: per_token_forward_kl(full_reference_logits, p25_official_logits),
        "js_r020_r025_official": per_token_js(p20_logits, p25_official_logits),
        "kl_r025_nested_to_r020": per_token_forward_kl(p25_nested_logits, p20_logits),
        "js_r020_r025_nested": per_token_js(p20_logits, p25_nested_logits),
        reference_nested_key: per_token_forward_kl(full_reference_logits, p25_nested_logits),
    }
    for name, values in metric_tensors.items():
        if not torch.isfinite(values).all() or bool((values < 0).any()):
            raise FloatingPointError(f"Invalid metric values for {name}.")

    token_ids = [int(value) for value in generated_ids.flatten().detach().cpu().tolist()]
    token_text = [
        decode_token_ids(processor, torch.tensor([token_id], device=generated_ids.device))
        for token_id in token_ids
    ]
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    timings["total_seconds"] = time.perf_counter() - started
    return {
        "schema_version": "opsd_checkpoint_kl_trajectory_v2",
        "sample_id": sample.sample_id,
        "checkpoint_label": checkpoint_label,
        "checkpoint_step": int(checkpoint_step),
        "question": sample.question,
        "ground_truth": sample.target,
        "image_path": str(sample.image),
        "prompt": sample.prompt,
        "prefix_source": prefix_source,
        "prefix_provenance": prefix_provenance,
        "generated_text": generated_text,
        "generated_token_ids": token_ids,
        "generated_token_text": token_text,
        "generated_token_count": count,
        "hit_max_new_tokens": bool(
            count >= int(cfg["max_new_tokens"])
            and (not token_ids or eos_token_id is None or token_ids[-1] != int(eos_token_id))
        ),
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "use_cache": True,
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "retention_ratio": rollout_ratio,
        },
        "teacher": {
            "model": str(cfg["base_model"]),
            "visual_tokens": "full",
            "adapter": (
                "same_merged_checkpoint"
                if full_reference == "self_checkpoint"
                else ("none" if teacher_model is not None else "disabled")
            ),
            "weights_same_as_student": full_reference == "self_checkpoint",
            "fixed_across_checkpoints": full_reference != "self_checkpoint",
            "reference_mode": full_reference,
            "execution": teacher_execution,
        },
        "ratio_pair": {
            "low_retention_ratio": rollout_ratio,
            "high_retention_ratio": comparison_ratio,
            "contextual_ratio": float(cfg["contextual_ratio"]),
            "legacy_metric_keys": {
                "kl_r025_official_to_r020": "KL(p_student_high_official || p_student_low)",
                "kl_base_full_to_r020": "KL(p_base_full || p_student_low)",
                "kl_base_full_to_r025_official": "KL(p_base_full || p_student_high_official)"
            }
        },
        "kl_directions": {
            "kl_r025_official_to_r020": "KL(p_student_high_official || p_student_low)",
            "kl_base_full_to_r020": "KL(p_base_full || p_student_low)",
            "kl_base_full_to_r025_official": "KL(p_base_full || p_student_high_official)",
            "kl_r025_nested_to_r020": "KL(p_student_high_nested || p_student_low)",
            "kl_base_full_to_r025_nested": "KL(p_base_full || p_student_high_nested)",
            "kl_checkpoint_full_to_r020": "KL(p_checkpoint_full || p_checkpoint_low)",
            "kl_checkpoint_full_to_r025_official": "KL(p_checkpoint_full || p_checkpoint_high_official)",
            "kl_checkpoint_full_to_r025_nested": "KL(p_checkpoint_full || p_checkpoint_high_nested)",
        },
        "visual_tokens": {
            "full": int(meta20["num_full_visual_tokens"]),
            "low_official": int(meta20["num_kept_visual_tokens"]),
            "high_official": int(meta25["num_kept_visual_tokens"]),
            "high_nested": int(m25_nested.count),
            "low_realized_ratio": float(meta20["num_kept_visual_tokens"] / meta20["num_full_visual_tokens"]),
            "high_official_realized_ratio": float(meta25["num_kept_visual_tokens"] / meta25["num_full_visual_tokens"]),
            "high_nested_realized_ratio": float(m25_nested.count / bank.num_visual_tokens),
            "r020_official": int(meta20["num_kept_visual_tokens"]),
            "r025_official": int(meta25["num_kept_visual_tokens"]),
            "r025_nested": int(m25_nested.count),
            "r020_realized_ratio": float(meta20["num_kept_visual_tokens"] / meta20["num_full_visual_tokens"]),
            "r025_official_realized_ratio": float(meta25["num_kept_visual_tokens"] / meta25["num_full_visual_tokens"]),
            "r025_nested_realized_ratio": float(m25_nested.count / bank.num_visual_tokens),
        },
        "masks": {
            "low_retained_indices": m20.retained_indices.detach().cpu().tolist(),
            "high_nested_retained_indices": m25_nested.retained_indices.detach().cpu().tolist(),
            "low_subset_high_nested": True,
            "r020_retained_indices": m20.retained_indices.detach().cpu().tolist(),
            "r025_nested_retained_indices": m25_nested.retained_indices.detach().cpu().tolist(),
            "r020_subset_r025_nested": True,
            "official_high_retention_note": "Official VisionZip recomputes contextual centers; it is not forced to be nested.",
            "official_r025_note": "Legacy key: official high-retention VisionZip recomputes contextual centers and is not forced to be nested.",
        },
        "official_fixed_r020_equivalence": equivalence,
        "metrics": {name: values.detach().float().cpu().tolist() for name, values in metric_tensors.items()},
        "metric_stats": {name: tensor_stats(values) for name, values in metric_tensors.items()},
        "timings": timings,
        "generation_metadata": generation_metadata,
    }


def main() -> int:
    args = parse_args()
    cfg = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    clean_transformers = Path(cfg["clean_transformers_src"]).expanduser().resolve()
    if not clean_transformers.is_dir():
        raise FileNotFoundError(clean_transformers)
    os.environ["ARMEN_TRANSFORMERS_SRC"] = str(clean_transformers)
    os.environ["OPSD_PRUNING_METHOD"] = "visionzip"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(clean_transformers))
    sys.path.insert(0, str(PROJECT_ROOT))

    import torch
    from opsd.visionzip_aokvqa.prompting import FormattedAOKVQASample

    if not torch.cuda.is_available():
        raise RuntimeError("This worker requires one CUDA GPU.")
    seed_everything(int(cfg["seed"]))
    checkpoint_root = Path(cfg["checkpoint_root"]).expanduser().resolve()
    adapter_path = resolve_adapter(args.adapter_path, checkpoint_root)
    samples_path = (args.samples or Path(cfg["samples"])).expanduser().resolve()
    prefix_source_dir = args.prefix_source_dir.expanduser().resolve() if args.prefix_source_dir else None
    if prefix_source_dir is not None and not prefix_source_dir.is_dir():
        raise FileNotFoundError(prefix_source_dir)
    rows = read_jsonl(samples_path)
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError("No samples selected.")
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = int(args.max_new_tokens)

    checkpoint_entry = next(
        (item for item in cfg["checkpoint_steps"] if item["label"] == args.checkpoint_label),
        None,
    )
    if checkpoint_entry is None:
        raise KeyError(f"Unknown checkpoint label: {args.checkpoint_label}")
    checkpoint_step = int(checkpoint_entry["step"])
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else Path(cfg["output_root"]).expanduser().resolve() / args.checkpoint_label
    )
    samples_dir = output_dir / "samples"
    errors_dir = output_dir / "errors"
    samples_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in samples_dir.glob("*.json"):
            path.unlink()
        for path in errors_dir.glob("*.json"):
            path.unlink()

    model, processor = load_model(cfg, adapter_path, merge_student=bool(args.merge_student))
    teacher_model = (
        load_base_teacher(cfg)
        if (
            args.full_reference == "base_teacher"
            and args.merge_student
            and adapter_path is not None
        )
        else None
    )
    import transformers
    parameter_names = [name for name, parameter in model.named_parameters() if "lora_" in name]
    expected_lora_tensors = 0 if args.merge_student else 392
    if len(parameter_names) != expected_lora_tensors:
        raise RuntimeError(
            f"Expected {expected_lora_tensors} LLM-only LoRA tensors for this load form, "
            f"found {len(parameter_names)}."
        )
    model_class = model.get_base_model().__class__.__name__ if hasattr(model, "get_base_model") else model.__class__.__name__
    if "Qwen2_5_VL" not in model_class:
        raise RuntimeError(f"Expected Qwen2.5-VL, got {model_class}.")

    manifest = {
        "schema_version": "opsd_checkpoint_kl_trajectory_manifest_v1",
        "experiment_id": cfg["experiment_id"],
        "training_method": cfg.get("training_method", "opsd"),
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_step": checkpoint_step,
        "adapter_path": str(adapter_path or "__BASE__"),
        "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors") if adapter_path else None,
        "base_model": cfg["base_model"],
        "teacher": (
            "same checkpoint weights with full visual tokens"
            if args.full_reference == "self_checkpoint"
            else (
                "independent adapter-free base model with full visual tokens"
                if teacher_model is not None
                else "fixed adapter-disabled base model with full visual tokens"
            )
        ),
        "full_reference_mode": args.full_reference,
        "ema_teacher_used": False,
        "samples": str(samples_path),
        "samples_sha256": sha256_file(samples_path),
        "sample_count": len(rows),
        "prefix_source_dir": str(prefix_source_dir) if prefix_source_dir is not None else None,
        "model_class": model_class,
        "parameter_scope": "language_decoder_only",
        "trainable_lora_tensor_count": len(parameter_names),
        "load_form": "merged_bf16_in_memory" if args.merge_student else "peft_adapter_unmerged",
        "student_adapter_merged": bool(args.merge_student),
        "independent_teacher_model": teacher_model is not None,
        "clean_transformers_src": str(clean_transformers),
        "transformers_version": transformers.__version__,
        "transformers_file": str(Path(transformers.__file__).resolve()),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "git_commit": capture(["git", "rev-parse", "HEAD"]),
        "git_status_sha256": hashlib.sha256(capture(["git", "status", "--porcelain=v1"]).encode()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": capture(["hostname"]),
        "config": cfg,
        "primary_metrics": list(
            SELF_CHECKPOINT_METRICS
            if args.full_reference == "self_checkpoint"
            else PRIMARY_METRICS
        ),
        "exact_full_vocabulary_kl": True,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)

    failures = 0
    started = time.time()
    for ordinal, row in enumerate(rows, start=1):
        sample = make_sample(row, FormattedAOKVQASample)
        sample_path = samples_dir / safe_name(sample.sample_id)
        if sample_path.exists() and not args.overwrite:
            continue
        try:
            fixed_prefix_payload = None
            if prefix_source_dir is not None:
                prefix_path = prefix_source_dir / safe_name(sample.sample_id)
                if not prefix_path.is_file():
                    raise FileNotFoundError(f"Missing reusable prefix: {prefix_path}")
                fixed_prefix_payload = json.loads(prefix_path.read_text(encoding="utf-8"))
            torch.cuda.reset_peak_memory_stats()
            payload = run_one(
                model,
                processor,
                sample,
                cfg,
                args.checkpoint_label,
                checkpoint_step,
                teacher_model=teacher_model,
                full_reference=args.full_reference,
                fixed_prefix_payload=fixed_prefix_payload,
            )
            payload["peak_gpu_allocated_gib"] = float(torch.cuda.max_memory_allocated() / 2**30)
            payload["peak_gpu_reserved_gib"] = float(torch.cuda.max_memory_reserved() / 2**30)
            atomic_write_json(sample_path, payload)
            (errors_dir / safe_name(sample.sample_id)).unlink(missing_ok=True)
            means = payload["metric_stats"]
            reference_low_key = (
                "kl_checkpoint_full_to_r020"
                if args.full_reference == "self_checkpoint"
                else "kl_base_full_to_r020"
            )
            reference_high_key = (
                "kl_checkpoint_full_to_r025_official"
                if args.full_reference == "self_checkpoint"
                else "kl_base_full_to_r025_official"
            )
            print(
                f"[{ordinal}/{len(rows)}] checkpoint={args.checkpoint_label} id={sample.sample_id} "
                f"tokens={payload['generated_token_count']} "
                f"KLhigh_to_low={means['kl_r025_official_to_r020']['mean']:.6f} "
                f"KLfull_to_low={means[reference_low_key]['mean']:.6f} "
                f"KLfull_to_high={means[reference_high_key]['mean']:.6f} "
                f"sec={payload['timings']['total_seconds']:.2f}",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            atomic_write_json(
                errors_dir / safe_name(sample.sample_id),
                {
                    "sample_id": sample.sample_id,
                    "checkpoint_label": args.checkpoint_label,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"ERROR checkpoint={args.checkpoint_label} id={sample.sample_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize_checkpoint(output_dir, len(rows))
    summary.update(
        {
            "checkpoint_label": args.checkpoint_label,
            "checkpoint_step": checkpoint_step,
            "new_failures": failures,
            "wall_seconds": time.time() - started,
            "output_dir": str(output_dir),
        }
    )
    atomic_write_json(output_dir / "checkpoint_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    del model
    if teacher_model is not None:
        del teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    return 1 if failures or not summary["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
