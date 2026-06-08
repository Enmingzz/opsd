#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from opsd.visionzip_aokvqa.aokvqa import FormattedAOKVQASample, load_aokvqa_dataset
from opsd.visionzip_aokvqa.losses import (
    compute_forward_kl,
    compute_generalized_jsd,
    compute_sequence_logprob,
    compute_token_ce,
    grpo_policy_loss,
)
from opsd.visionzip_aokvqa.prompting import build_opsd_teacher_prompt, parse_final_answer
from opsd.visionzip_aokvqa.qwen_wrapper import (
    apply_lora,
    encode_prompt,
    encode_prompt_and_response,
    encode_prompt_text,
    extract_generated_logits,
    forward_pruned,
    generate_pruned,
    load_qwen_model_and_processor,
    model_input_subset,
    primary_device,
    teacher_adapter_disabled,
)
from opsd.visionzip_aokvqa.visionzip import grpo_group_advantages


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")
METHODS = ("sft", "grpo", "epic", "opsd", "opsd_fixed_teacher", "opsd_nogt", "offpolicy")
OPSD_DYNAMIC_TEACHER_ALIASES = {"", "dynamic", "dynamic_shared_current", "shared_current", "latest"}
OPSD_FIXED_TEACHER_ALIASES = {"fixed_base", "fixed_teacher", "legacy_fixed_base", "base"}
OPSD_EMA_TEACHER_ALIASES = {"ema", "ema_teacher", "ema_shared", "ema_reference"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", choices=METHODS, default=None)
    p.add_argument("--output_dir", default="")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--start_step", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow_embedding_fallback", action="store_true")
    p.add_argument("--adapter_path", default="")
    p.add_argument("--selected_ids_path", default="")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    return p


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    if args.method:
        set_nested(cfg, "training.method", args.method)
    method = get_nested(cfg, "training.method", "sft")
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg.setdefault("output_dir", str(OUTPUT_ROOT / "checkpoints" / method))
    if args.max_steps is not None:
        set_nested(cfg, "training.max_steps", args.max_steps)
    if args.start_step is not None:
        set_nested(cfg, "training.start_step", args.start_step)
    if args.limit is not None:
        set_nested(cfg, "dataset.limit", args.limit)
    if args.smoke:
        set_nested(cfg, "training.max_steps", 1)
        set_nested(cfg, "dataset.limit", 16)
        cfg["smoke"] = True
    if args.allow_embedding_fallback:
        set_nested(cfg, "pruning.allow_embedding_fallback", True)
    if args.adapter_path:
        set_nested(cfg, "training.adapter_path", args.adapter_path)
    if args.selected_ids_path:
        set_nested(cfg, "dataset.selected_ids_path", args.selected_ids_path)
    if args.gradient_accumulation_steps is not None:
        set_nested(cfg, "training.gradient_accumulation_steps", args.gradient_accumulation_steps)
    return cfg


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            timeout_minutes = int(os.environ.get("OPSD_DDP_TIMEOUT_MINUTES", "10"))
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    return distributed, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return int(rank) == 0


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def apply_selected_ids(dataset: list[FormattedAOKVQASample], ids_path: str | Path | None) -> list[FormattedAOKVQASample]:
    if not ids_path:
        return dataset
    path = Path(ids_path)
    ids = json.loads(path.read_text(encoding="utf-8"))
    wanted = [str(x) for x in ids]
    by_id = {sample.sample_id: sample for sample in dataset}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise KeyError(f"Selected training ids missing from dataset: {missing[:10]}")
    return [by_id[sample_id] for sample_id in wanted]


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_resolved_config(cfg: dict[str, Any], output_dir: Path) -> None:
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def sample_retention_ratio(cfg: dict[str, Any], rng: random.Random) -> float:
    ratios = [float(x) for x in get_nested(cfg, "pruning.train_retention_ratios", [0.1, 0.2, 0.3, 0.4])]
    weights_raw = get_nested(cfg, "pruning.train_retention_ratio_weights", None)
    if weights_raw is None:
        return float(rng.choice(ratios))
    weights = [float(x) for x in weights_raw]
    if len(weights) != len(ratios):
        raise ValueError(
            "pruning.train_retention_ratio_weights must have the same length as pruning.train_retention_ratios; "
            f"got {len(weights)} weights for {len(ratios)} ratios."
        )
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("pruning.train_retention_ratio_weights must be non-negative and sum to a positive value.")
    return float(rng.choices(ratios, weights=weights, k=1)[0])


def resolve_opsd_teacher_strategy(cfg: dict[str, Any], teacher_model: Any | None) -> str:
    raw = str(get_nested(cfg, "opsd.teacher_strategy", "") or "").strip().lower()
    use_ema = bool(get_nested(cfg, "opsd.use_ema_teacher", False))
    if teacher_model is not None and (use_ema or raw in OPSD_EMA_TEACHER_ALIASES):
        return "ema"
    if teacher_model is not None:
        return "external"
    fixed_teacher = bool(get_nested(cfg, "opsd.fixed_teacher", False))
    if fixed_teacher and not raw:
        return "fixed_base"
    if use_ema:
        return "ema"
    if raw in OPSD_EMA_TEACHER_ALIASES:
        return "ema"
    if raw in OPSD_DYNAMIC_TEACHER_ALIASES:
        return "dynamic_shared_current"
    if raw in OPSD_FIXED_TEACHER_ALIASES:
        return "fixed_base"
    raise ValueError(
        "Unsupported opsd.teacher_strategy="
        f"{raw!r}. Use dynamic_shared_current for the online shared path, ema for the official EMA reference path, "
        "or fixed_base for the legacy ablation."
    )


def trainable_parameter_names(model: Any) -> list[str]:
    return [name for name, param in unwrap_model(model).named_parameters() if param.requires_grad]


def copy_named_parameters(source_model: Any, target_model: Any, names: list[str]) -> None:
    source_params = dict(unwrap_model(source_model).named_parameters())
    target_params = dict(unwrap_model(target_model).named_parameters())
    missing = [name for name in names if name not in target_params]
    if missing:
        raise KeyError(f"EMA teacher is missing student trainable parameters: {missing[:10]}")
    with torch.no_grad():
        for name in names:
            target_params[name].data.copy_(source_params[name].detach().data)


def _configured(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def resolve_ema_update_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_decay = get_nested(cfg, "opsd.ema_decay", None)
    raw_alpha = get_nested(cfg, "opsd.ema_alpha", None)
    has_decay = _configured(raw_decay)
    has_alpha = _configured(raw_alpha)
    if has_decay and has_alpha:
        raise ValueError("Specify only one of opsd.ema_decay (official) or opsd.ema_alpha (legacy ablation).")
    if has_decay:
        decay = float(raw_decay)
        if decay <= 0.0 or decay >= 1.0:
            raise ValueError(f"opsd.ema_decay must be in (0, 1), got {decay}.")
        return {
            "mode": "official_decay",
            "decay": decay,
            "alpha": 1.0 - decay,
            "lazy_init": bool(get_nested(cfg, "opsd.ema_lazy_init", True)),
        }
    if has_alpha:
        alpha = float(raw_alpha)
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError(f"opsd.ema_alpha must be in (0, 1], got {alpha}.")
        return {
            "mode": "legacy_alpha",
            "decay": 1.0 - alpha,
            "alpha": alpha,
            "lazy_init": bool(get_nested(cfg, "opsd.ema_lazy_init", False)),
        }
    decay = float(get_nested(cfg, "opsd.ema_decay_default", 0.999))
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError(f"opsd.ema_decay_default must be in (0, 1), got {decay}.")
    return {
        "mode": "official_decay_default",
        "decay": decay,
        "alpha": 1.0 - decay,
        "lazy_init": bool(get_nested(cfg, "opsd.ema_lazy_init", True)),
    }


def update_ema_teacher(source_model: Any, target_model: Any, names: list[str], decay: float) -> None:
    source_params = dict(unwrap_model(source_model).named_parameters())
    target_params = dict(unwrap_model(target_model).named_parameters())
    decay = float(decay)
    if decay < 0.0 or decay >= 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
    with torch.no_grad():
        for name in names:
            target_params[name].data.mul_(decay).add_(source_params[name].detach().data, alpha=1.0 - decay)


def create_ema_shadow(model: Any, names: list[str]) -> dict[str, torch.Tensor]:
    params = dict(unwrap_model(model).named_parameters())
    with torch.no_grad():
        return {name: params[name].detach().clone() for name in names}


def load_ema_shadow(path: str | Path, model: Any, names: list[str]) -> dict[str, torch.Tensor] | None:
    if not path:
        return None
    ema_path = Path(path) / "ema_shadow.pt"
    if not ema_path.exists():
        return None
    params = dict(unwrap_model(model).named_parameters())
    payload = torch.load(ema_path, map_location="cpu")
    missing = [name for name in names if name not in payload]
    if missing:
        raise KeyError(f"EMA shadow checkpoint is missing parameters: {missing[:10]}")
    return {
        name: payload[name].to(device=params[name].device, dtype=params[name].dtype).clone()
        for name in names
    }


def update_ema_shadow(model: Any, shadow: dict[str, torch.Tensor], names: list[str], decay: float) -> None:
    params = dict(unwrap_model(model).named_parameters())
    decay = float(decay)
    if decay < 0.0 or decay >= 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
    with torch.no_grad():
        for name in names:
            shadow[name].mul_(decay).add_(params[name].detach().data, alpha=1.0 - decay)


@contextmanager
def swapped_ema_parameters(model: Any, shadow: dict[str, torch.Tensor]):
    params = dict(unwrap_model(model).named_parameters())
    originals = {name: params[name].detach().clone() for name in shadow}
    try:
        with torch.no_grad():
            for name, value in shadow.items():
                params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))
        yield
    finally:
        with torch.no_grad():
            for name, value in originals.items():
                params[name].data.copy_(value)


@contextmanager
def temporary_eval(model: Any):
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def save_ema_shadow(path: Path, shadow: dict[str, torch.Tensor] | None) -> None:
    if shadow is None:
        return
    torch.save({name: value.detach().cpu() for name, value in shadow.items()}, path / "ema_shadow.pt")


def sequence_inputs_from_prompt(prompt_inputs: dict[str, torch.Tensor], generated_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    generated_ids = generated_ids.reshape(1, -1).to(device=prompt_inputs["input_ids"].device)
    out = dict(prompt_inputs)
    out["input_ids"] = torch.cat([prompt_inputs["input_ids"], generated_ids], dim=1)
    ones = torch.ones_like(generated_ids, dtype=prompt_inputs["attention_mask"].dtype)
    out["attention_mask"] = torch.cat([prompt_inputs["attention_mask"], ones], dim=1)
    if "mm_token_type_ids" in prompt_inputs and prompt_inputs["mm_token_type_ids"] is not None:
        zeros = torch.zeros_like(generated_ids, dtype=prompt_inputs["mm_token_type_ids"].dtype)
        out["mm_token_type_ids"] = torch.cat([prompt_inputs["mm_token_type_ids"], zeros], dim=1)
    return out


def generation_top_k(cfg: dict[str, Any]) -> int | None:
    top_k = int(get_nested(cfg, "generation.top_k", 0) or 0)
    return top_k if top_k > 0 else None


def generate_full_teacher(model: Any, processor: Any, prompt_inputs: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, str]:
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or eos_token_id
    generate_model = unwrap_model(model)
    with torch.no_grad():
        kwargs = {
            **model_input_subset(prompt_inputs),
            "max_new_tokens": int(get_nested(cfg, "generation.max_new_tokens", 128)),
            "do_sample": True,
            "temperature": float(get_nested(cfg, "generation.temperature", 0.7)),
            "top_p": float(get_nested(cfg, "generation.top_p", 0.9)),
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        top_k = generation_top_k(cfg)
        if top_k is not None:
            kwargs["top_k"] = top_k
        output_ids = generate_model.generate(**kwargs)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    gen = output_ids[:, prompt_len:]
    text = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return gen, text


def corrected_epic_target(text: str, correct_letter: str) -> str:
    parsed = parse_final_answer(text)
    if parsed == correct_letter:
        return text.strip()
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    kept = []
    for line in lines:
        if "final" in line.lower() and "answer" in line.lower():
            continue
        kept.append(line)
    reasoning = "\n".join(kept).strip()
    if "Reasoning:" not in reasoning:
        reasoning = f"Reasoning: {reasoning or 'The correct option is supported by the image and question.'}"
    return f"{reasoning}\nFinal answer: {correct_letter}"


def build_epic_target(text: str, correct_letter: str, cfg: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    parsed = parse_final_answer(text)
    teacher_correct = parsed == correct_letter
    correct_final_answer = bool(get_nested(cfg, "epic.correct_final_answer", False))
    filter_teacher_correct = bool(get_nested(cfg, "epic.filter_teacher_correct", True))
    if teacher_correct:
        return text.strip(), {
            "teacher_parseable": parsed is not None,
            "teacher_correct": True,
            "epic_target_policy": "teacher_correct",
        }
    if filter_teacher_correct and not correct_final_answer:
        return None, {
            "teacher_parseable": parsed is not None,
            "teacher_correct": False,
            "epic_target_policy": "filtered_teacher_incorrect",
        }
    if correct_final_answer:
        return corrected_epic_target(text, correct_letter), {
            "teacher_parseable": parsed is not None,
            "teacher_correct": False,
            "epic_target_policy": "corrected_final_answer",
        }
    return None, {
        "teacher_parseable": parsed is not None,
        "teacher_correct": False,
        "epic_target_policy": "dropped_teacher_incorrect",
    }


def epic_teacher_retention_ratio(cfg: dict[str, Any], student_retention_ratio: float) -> float | str:
    """Map the student retention ratio to EPIC's easier teacher view.

    Official EPIC/TCD runs the same model twice: a more compressed student
    forward and an easier teacher forward, then distills teacher token
    distributions into the student.  Their LLaVA code expresses compression as
    a reduction ratio, where teacher_reduction = student_reduction - gap.  In
    this Qwen2.5-VL/VisionZip adaptation retention = 1 - reduction, so the
    equivalent teacher retention is student_retention + gap.
    """

    policy = str(get_nested(cfg, "epic.teacher_retention_policy", "easier_by_gap"))
    if policy == "full":
        return "full"
    if policy != "easier_by_gap":
        raise ValueError(f"Unsupported EPIC teacher_retention_policy={policy!r}.")
    gap = float(get_nested(cfg, "epic.teacher_retention_gap", 0.3))
    return min(1.0, max(float(student_retention_ratio), float(student_retention_ratio) + gap))


def sft_like_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
    target_response: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = primary_device(model)
    _, full_inputs, answer_ids = encode_prompt_and_response(
        processor,
        sample,
        target_response,
        image_root=get_nested(cfg, "dataset.image_root", ""),
        device=device,
    )
    prompt_len = int(full_inputs["input_ids"].shape[1] - answer_ids.numel())
    outputs, pruned = forward_pruned(
        model,
        full_inputs,
        retention_ratio,
        prompt_len=prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    student_prompt_len = int(pruned["metadata"]["student_prompt_len"])
    logits = extract_generated_logits(outputs.logits, student_prompt_len, int(answer_ids.numel()))
    loss = compute_token_ce(logits, answer_ids)
    return loss, {
        "loss_type": "ce",
        "generated_tokens": int(answer_ids.numel()),
        **numeric_metadata(pruned["metadata"]),
    }


def epic_tcd_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """EPIC-style token consistency distillation adapted to Qwen2.5-VL/VisionZip.

    This follows the official EPIC repo's TCD training structure rather than
    our earlier response-level generated-target baseline:
      1. run student with compressed visual tokens and supervised labels;
      2. run the same model under no_grad with an easier/full visual-token view;
      3. align logits by answer-token index;
      4. optimize alpha * forward-KL + (1 - alpha) * SFT CE.

    The teacher pass intentionally keeps the current LoRA adapter enabled,
    matching EPIC's "teacher and student share weights" self-consistency setup.
    """

    device = primary_device(model)
    _, full_inputs, answer_ids = encode_prompt_and_response(
        processor,
        sample,
        sample.target,
        image_root=get_nested(cfg, "dataset.image_root", ""),
        device=device,
    )
    prompt_len = int(full_inputs["input_ids"].shape[1] - answer_ids.numel())
    student_outputs, student_pruned = forward_pruned(
        model,
        full_inputs,
        retention_ratio,
        prompt_len=prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    student_prompt_len = int(student_pruned["metadata"]["student_prompt_len"])
    student_logits = extract_generated_logits(student_outputs.logits, student_prompt_len, int(answer_ids.numel()))
    student_loss_sft = compute_token_ce(student_logits, answer_ids)

    teacher_ratio = epic_teacher_retention_ratio(cfg, retention_ratio)
    with torch.no_grad():
        if teacher_ratio == "full" or float(teacher_ratio) >= 1.0:
            teacher_outputs = model(**model_input_subset(full_inputs), use_cache=False)
            teacher_prompt_len = prompt_len
            teacher_meta: dict[str, Any] = {
                "teacher_num_full_visual_tokens": int(student_pruned["metadata"].get("num_full_visual_tokens", 0)),
                "teacher_num_kept_visual_tokens": int(student_pruned["metadata"].get("num_full_visual_tokens", 0)),
            }
        else:
            teacher_outputs, teacher_pruned = forward_pruned(
                model,
                full_inputs,
                float(teacher_ratio),
                prompt_len=prompt_len,
                allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
            )
            teacher_prompt_len = int(teacher_pruned["metadata"]["student_prompt_len"])
            teacher_meta = {
                "teacher_num_full_visual_tokens": int(teacher_pruned["metadata"].get("num_full_visual_tokens", 0)),
                "teacher_num_kept_visual_tokens": int(teacher_pruned["metadata"].get("num_kept_visual_tokens", 0)),
            }
    teacher_logits = extract_generated_logits(teacher_outputs.logits, teacher_prompt_len, int(answer_ids.numel())).detach()
    distillation_loss = compute_forward_kl(
        teacher_logits,
        student_logits,
        temperature=float(get_nested(cfg, "epic.temperature", 1.0)),
    )
    alpha = float(get_nested(cfg, "epic.alpha", 0.5))
    loss = alpha * distillation_loss + (1.0 - alpha) * student_loss_sft
    return loss, {
        "loss_type": "epic_tcd",
        "generated_tokens": int(answer_ids.numel()),
        "student_retention_ratio": float(retention_ratio),
        "teacher_retention_ratio": "full" if teacher_ratio == "full" else float(teacher_ratio),
        "distillation_loss": float(distillation_loss.detach().cpu()),
        "student_loss_sft": float(student_loss_sft.detach().cpu()),
        "epic_alpha": alpha,
        "epic_temperature": float(get_nested(cfg, "epic.temperature", 1.0)),
        "epic_reference": "ZichenWen1/EPIC TCD adapted to Qwen2.5-VL/VisionZip",
        **teacher_meta,
        **numeric_metadata(student_pruned["metadata"]),
    }


def opsd_nogt_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
    teacher_model: Any | None = None,
    ema_shadow: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
    gen_ids, gen_text, _ = generate_pruned(
        model,
        processor,
        prompt_inputs,
        retention_ratio,
        max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
        do_sample=True,
        temperature=float(get_nested(cfg, "generation.temperature", 0.7)),
        top_p=float(get_nested(cfg, "generation.top_p", 0.9)),
        top_k=generation_top_k(cfg),
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    if gen_ids.numel() == 0:
        raise RuntimeError("OPSD student generated zero tokens.")
    seq_inputs = sequence_inputs_from_prompt(prompt_inputs, gen_ids)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    raw_teacher_strategy = str(get_nested(cfg, "opsd.teacher_strategy", "") or "").strip()
    explicit_teacher_strategy = (
        teacher_model is not None
        or ema_shadow is not None
        or raw_teacher_strategy
        or bool(get_nested(cfg, "opsd.use_ema_teacher", False))
    )
    teacher_strategy = resolve_opsd_teacher_strategy(cfg, teacher_model) if explicit_teacher_strategy else "fixed_base"
    if teacher_strategy == "external":
        with torch.no_grad():
            teacher_outputs = teacher_model(**model_input_subset(seq_inputs), use_cache=False)
        teacher_source = "external_no_gt_full_token"
    elif teacher_strategy == "ema":
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(**model_input_subset(seq_inputs), use_cache=False)
            teacher_source = "ema_no_gt_full_token"
        elif ema_shadow is not None:
            with torch.no_grad(), swapped_ema_parameters(model, ema_shadow), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(seq_inputs), use_cache=False)
            teacher_source = "ema_lora_shadow_no_gt_full_token"
        else:
            with torch.no_grad(), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(seq_inputs), use_cache=False)
            teacher_source = "ema_uninitialized_current_no_gt_full_token"
    elif teacher_strategy == "dynamic_shared_current":
        with torch.no_grad(), temporary_eval(model):
            teacher_outputs = model(**model_input_subset(seq_inputs), use_cache=False)
        teacher_source = "dynamic_shared_current_no_gt_full_token"
    elif teacher_strategy == "fixed_base":
        with torch.no_grad(), teacher_adapter_disabled(model):
            teacher_outputs = model(**model_input_subset(seq_inputs), use_cache=False)
        teacher_source = "base_full_token"
    else:
        raise AssertionError(teacher_strategy)

    student_outputs, pruned = forward_pruned(
        model,
        seq_inputs,
        retention_ratio,
        prompt_len=prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    teacher_logits = extract_generated_logits(teacher_outputs.logits, prompt_len, int(gen_ids.numel()))
    student_logits = extract_generated_logits(student_outputs.logits, int(pruned["metadata"]["student_prompt_len"]), int(gen_ids.numel()))
    kl = compute_forward_kl(teacher_logits, student_logits, temperature=float(get_nested(cfg, "opsd.temperature", 1.0)))
    return kl, {
        "loss_type": "opsd_nogt_forward_kl",
        "generated_tokens": int(gen_ids.numel()),
        "kl_loss": float(kl.detach().cpu()),
        "parseable": parse_final_answer(gen_text) is not None,
        "student_correct": parse_final_answer(gen_text) == sample.correct_letter,
        "teacher_source": teacher_source,
        "opsd_teacher_strategy": teacher_strategy,
        "teacher_context": "student_prompt_no_ground_truth",
        "opsd_reference": (
            "no_gt_ema_teacher_ablation"
            if teacher_strategy == "ema"
            else "no_gt_dynamic_shared_current_teacher_ablation"
            if teacher_strategy == "dynamic_shared_current"
            else "legacy_no_gt_ablation"
            if teacher_strategy == "fixed_base"
            else "no_gt_external_teacher_ablation"
        ),
        **numeric_metadata(pruned["metadata"]),
    }


def opsd_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
    teacher_model: Any | None = None,
    ema_shadow: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Official-style OPSD adapted to A-OKVQA/Qwen2.5-VL/VisionZip.

    Student rolls out from the normal visual-question prompt.  Teacher scores
    the same generated suffix from a privileged prompt containing the reference
    A-OKVQA solution, matching the official OPSD student/teacher context split.
    """

    device = primary_device(model)
    image_root = get_nested(cfg, "dataset.image_root", "")
    student_prompt_inputs = encode_prompt(processor, sample, image_root=image_root, device=device)
    gen_ids, gen_text, _ = generate_pruned(
        model,
        processor,
        student_prompt_inputs,
        retention_ratio,
        max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
        do_sample=True,
        temperature=float(get_nested(cfg, "generation.temperature", 0.7)),
        top_p=float(get_nested(cfg, "generation.top_p", 0.9)),
        top_k=generation_top_k(cfg),
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    if gen_ids.numel() == 0:
        raise RuntimeError("OPSD student generated zero tokens.")

    student_seq_inputs = sequence_inputs_from_prompt(student_prompt_inputs, gen_ids)
    student_prompt_len = int(student_prompt_inputs["input_ids"].shape[1])

    teacher_prompt = build_opsd_teacher_prompt(sample.question, sample.options, sample.target)
    teacher_prompt_inputs = encode_prompt_text(processor, sample, teacher_prompt, image_root=image_root, device=device)
    teacher_seq_inputs = sequence_inputs_from_prompt(teacher_prompt_inputs, gen_ids)
    teacher_prompt_len = int(teacher_prompt_inputs["input_ids"].shape[1])

    teacher_strategy = resolve_opsd_teacher_strategy(cfg, teacher_model)
    if teacher_strategy == "external":
        with torch.no_grad():
            teacher_outputs = teacher_model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = "external_privileged_full_token"
    elif teacher_strategy == "ema":
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = "ema_privileged_full_token"
        elif ema_shadow is not None:
            with torch.no_grad(), swapped_ema_parameters(model, ema_shadow), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = "ema_lora_shadow_privileged_full_token"
        else:
            with torch.no_grad(), temporary_eval(model):
                teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
            teacher_source = "ema_uninitialized_current_privileged_full_token"
    elif teacher_strategy == "dynamic_shared_current":
        with torch.no_grad():
            teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = "dynamic_shared_current_privileged_full_token"
    elif teacher_strategy == "fixed_base":
        with torch.no_grad(), teacher_adapter_disabled(model):
            teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
        teacher_source = "fixed_base_privileged_full_token"
    else:
        raise AssertionError(teacher_strategy)

    token_count = int(gen_ids.numel())
    teacher_logits = extract_generated_logits(teacher_outputs.logits, teacher_prompt_len, token_count).detach()
    del teacher_outputs

    student_outputs, pruned = forward_pruned(
        model,
        student_seq_inputs,
        retention_ratio,
        prompt_len=student_prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    student_logits = extract_generated_logits(student_outputs.logits, int(pruned["metadata"]["student_prompt_len"]), token_count)

    gt_weight = float(get_nested(cfg, "opsd.ground_truth_ce_weight", 0.0) or 0.0)
    if gt_weight != 0.0:
        raise ValueError(
            "opsd.ground_truth_ce_weight is not part of the official-aligned OPSD path. "
            "Use training.method=opsd_nogt for the legacy no-GT ablation or create a separate explicit CE ablation."
        )
    beta = float(get_nested(cfg, "opsd.beta", 0.0))
    temperature = float(get_nested(cfg, "opsd.temperature", 1.0))
    top_k_raw = int(get_nested(cfg, "opsd.top_k_loss", 0) or 0)
    top_k = top_k_raw if top_k_raw > 0 else None
    token_clip_raw = float(get_nested(cfg, "opsd.jsd_token_clip", 0.05) or 0.0)
    token_clip = token_clip_raw if token_clip_raw > 0.0 else None
    distillation_loss = compute_generalized_jsd(
        teacher_logits,
        student_logits,
        beta=beta,
        temperature=temperature,
        top_k=top_k,
        token_clip=token_clip,
        clip_mode=str(get_nested(cfg, "opsd.jsd_clip_mode", "token")),
    )
    parsed = parse_final_answer(gen_text)
    return distillation_loss, {
        "loss_type": "official_opsd_generalized_jsd",
        "generated_tokens": token_count,
        "distillation_loss": float(distillation_loss.detach().cpu()),
        "kl_loss": float(distillation_loss.detach().cpu()) if beta == 0.0 else None,
        "parseable": parsed is not None,
        "student_correct": parsed == sample.correct_letter,
        "teacher_source": teacher_source,
        "opsd_teacher_strategy": teacher_strategy,
        "teacher_context": "ground_truth_reference_solution",
        "teacher_prompt_tokens": teacher_prompt_len,
        "student_prompt_tokens": student_prompt_len,
        "opsd_beta": beta,
        "opsd_temperature": temperature,
        "opsd_top_k_loss": top_k_raw,
        "opsd_jsd_token_clip": token_clip_raw,
        "opsd_jsd_clip_mode": str(get_nested(cfg, "opsd.jsd_clip_mode", "token")),
        "opsd_reference": (
            "siyan-zhao/OPSD EMA reference teacher adapted to Qwen2.5-VL/VisionZip"
            if teacher_strategy == "ema"
            else
            "siyan-zhao/OPSD latest dynamic shared-current teacher adapted to Qwen2.5-VL/VisionZip"
            if teacher_strategy == "dynamic_shared_current"
            else "legacy_fixed_base_teacher_ablation"
            if teacher_strategy == "fixed_base"
            else "external_teacher_ablation"
        ),
        **numeric_metadata(pruned["metadata"]),
    }


def offpolicy_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
    with teacher_adapter_disabled(model):
        gen_ids, gen_text = generate_full_teacher(model, processor, prompt_inputs, cfg)
    if gen_ids.numel() == 0:
        raise RuntimeError("Off-policy teacher generated zero tokens.")

    seq_inputs = sequence_inputs_from_prompt(prompt_inputs, gen_ids)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    with torch.no_grad(), teacher_adapter_disabled(model):
        teacher_outputs = model(**model_input_subset(seq_inputs), use_cache=False)
    student_outputs, pruned = forward_pruned(
        model,
        seq_inputs,
        retention_ratio,
        prompt_len=prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    token_count = int(gen_ids.numel())
    teacher_logits = extract_generated_logits(teacher_outputs.logits, prompt_len, token_count).detach()
    student_logits = extract_generated_logits(student_outputs.logits, int(pruned["metadata"]["student_prompt_len"]), token_count)
    temperature = float(get_nested(cfg, "offpolicy.temperature", 1.0))
    kl = compute_forward_kl(teacher_logits, student_logits, temperature=temperature)
    parsed = parse_final_answer(gen_text)
    return kl, {
        "loss_type": "offpolicy_kl",
        "generated_tokens": token_count,
        "kl_loss": float(kl.detach().cpu()),
        "teacher_parseable": parsed is not None,
        "teacher_correct": parsed == sample.correct_letter,
        "teacher_retention_ratio": "full",
        "offpolicy_temperature": temperature,
        **numeric_metadata(pruned["metadata"]),
    }


def grpo_step(
    model: Any,
    processor: Any,
    sample: FormattedAOKVQASample,
    cfg: dict[str, Any],
    retention_ratio: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = primary_device(model)
    prompt_inputs = encode_prompt(processor, sample, image_root=get_nested(cfg, "dataset.image_root", ""), device=device)
    group_size = int(get_nested(cfg, "grpo.group_size", 4))
    logprobs = []
    rewards = []
    parseable_count = 0
    generated_lengths = []
    for _ in range(group_size):
        gen_ids, text, _ = generate_pruned(
            model,
            processor,
            prompt_inputs,
            retention_ratio,
            max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
            do_sample=True,
            temperature=float(get_nested(cfg, "generation.temperature", 0.7)),
            top_p=float(get_nested(cfg, "generation.top_p", 0.9)),
            top_k=generation_top_k(cfg),
            allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
        )
        if gen_ids.numel() == 0:
            continue
        parsed = parse_final_answer(text)
        parseable = parsed is not None
        parseable_count += int(parseable)
        reward = (1.0 if parsed == sample.correct_letter else 0.0) + (0.1 if parseable else 0.0)
        seq_inputs = sequence_inputs_from_prompt(prompt_inputs, gen_ids)
        outputs, pruned = forward_pruned(
            model,
            seq_inputs,
            retention_ratio,
            prompt_len=int(prompt_inputs["input_ids"].shape[1]),
            allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
        )
        logits = extract_generated_logits(outputs.logits, int(pruned["metadata"]["student_prompt_len"]), int(gen_ids.numel()))
        logprobs.append(compute_sequence_logprob(logits, gen_ids))
        rewards.append(reward)
        generated_lengths.append(int(gen_ids.numel()))
    if not logprobs:
        raise RuntimeError("GRPO generated no trainable completions.")
    logprobs_t = torch.stack(logprobs)
    rewards_t = torch.tensor(rewards, device=logprobs_t.device, dtype=torch.float32)
    advantages = grpo_group_advantages(rewards_t)
    return grpo_policy_loss(logprobs_t, advantages), {
        "loss_type": "grpo",
        "reward_mean": float(rewards_t.mean().detach().cpu()),
        "reward_std": float(rewards_t.std(unbiased=False).detach().cpu()),
        "parseable_rate": parseable_count / max(1, len(rewards)),
        "generated_tokens": sum(generated_lengths) / max(1, len(generated_lengths)),
    }


def numeric_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def train(cfg: dict[str, Any]) -> Path:
    distributed, rank, local_rank, world_size = setup_distributed()
    method = str(get_nested(cfg, "training.method", "sft")).lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}.")
    if method == "opsd_fixed_teacher":
        set_nested(cfg, "opsd.teacher_strategy", "fixed_base")
        set_nested(cfg, "opsd.fixed_teacher", True)
    elif (
        method == "opsd"
        and not str(get_nested(cfg, "opsd.teacher_strategy", "") or "").strip()
        and not str(get_nested(cfg, "opsd.teacher_adapter_path", "") or "").strip()
    ):
        set_nested(cfg, "opsd.teacher_strategy", "dynamic_shared_current")
        set_nested(cfg, "opsd.fixed_teacher", False)
    output_dir = Path(str(cfg.get("output_dir", OUTPUT_ROOT / "checkpoints" / method)))
    log_path = output_dir / "training_log.jsonl"
    max_steps = int(get_nested(cfg, "training.max_steps", 1000))
    start_step = int(get_nested(cfg, "training.start_step", 0) or 0)
    if start_step < 0 or start_step > max_steps:
        raise ValueError(f"training.start_step must be in [0, max_steps]; got {start_step} with max_steps={max_steps}.")
    if is_main_process(rank) and log_path.exists() and start_step == 0:
        log_path.unlink()
    if is_main_process(rank):
        save_resolved_config(cfg, output_dir)
    rng = random.Random(int(get_nested(cfg, "training.seed", 42)) + rank)
    torch.manual_seed(int(get_nested(cfg, "training.seed", 42)) + rank)
    dataset = load_aokvqa_dataset(
        get_nested(cfg, "dataset.name", "HuggingFaceM4/A-OKVQA"),
        splits=list(get_nested(cfg, "dataset.use_splits", ["train", "validation"])),
        limit=int(get_nested(cfg, "dataset.limit", 0) or 0),
        seed=int(get_nested(cfg, "training.seed", 42)),
    )
    dataset = apply_selected_ids(dataset, get_nested(cfg, "dataset.selected_ids_path", ""))
    if not dataset:
        raise ValueError("Training dataset is empty.")
    if distributed and max_steps % world_size != 0:
        raise ValueError(f"DDP requires max_steps divisible by world_size; got max_steps={max_steps}, world_size={world_size}.")
    if distributed and start_step % world_size != 0:
        raise ValueError(f"DDP requires start_step divisible by world_size; got start_step={start_step}, world_size={world_size}.")
    remaining_steps = max_steps - start_step
    if distributed and remaining_steps % world_size != 0:
        raise ValueError(
            f"DDP requires remaining steps divisible by world_size; got remaining={remaining_steps}, world_size={world_size}."
        )
    device_map = get_nested(cfg, "training.device_map", "auto")
    if distributed:
        device_map = {"": local_rank}
        stagger = float(os.environ.get("OPSD_DDP_STAGGER_LOAD_SECONDS", "0"))
        if stagger > 0:
            time.sleep(float(local_rank) * stagger)
    model, processor = load_qwen_model_and_processor(
        str(get_nested(cfg, "base_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        bf16=bool(get_nested(cfg, "training.bf16", True)),
        attn_implementation=str(get_nested(cfg, "training.attn_implementation", "flash_attention_2")),
        device_map=device_map,
    )
    if bool(get_nested(cfg, "training.use_lora", True)):
        model = apply_lora(
            model,
            r=int(get_nested(cfg, "training.lora_r", 16)),
            alpha=int(get_nested(cfg, "training.lora_alpha", 32)),
            dropout=float(get_nested(cfg, "training.lora_dropout", 0.05)),
            target_modules=list(get_nested(cfg, "training.target_modules", [])) or None,
            adapter_path=str(get_nested(cfg, "training.adapter_path", "")),
        )
    teacher_model = None
    ema_shadow: dict[str, torch.Tensor] | None = None
    ema_parameter_names: list[str] = []
    teacher_adapter_path = str(get_nested(cfg, "opsd.teacher_adapter_path", "") or "").strip()
    ema_teacher_enabled = bool(get_nested(cfg, "opsd.use_ema_teacher", False)) or (
        str(get_nested(cfg, "opsd.teacher_strategy", "") or "").strip().lower() in OPSD_EMA_TEACHER_ALIASES
    )
    ema_settings = resolve_ema_update_settings(cfg) if ema_teacher_enabled else {}
    if teacher_adapter_path:
        if not Path(teacher_adapter_path).exists():
            raise FileNotFoundError(f"OPSD teacher adapter path does not exist: {teacher_adapter_path}")
        teacher_model, _teacher_processor = load_qwen_model_and_processor(
            str(get_nested(cfg, "base_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
            bf16=bool(get_nested(cfg, "training.bf16", True)),
            attn_implementation=str(get_nested(cfg, "training.attn_implementation", "flash_attention_2")),
            device_map=device_map,
        )
        teacher_model = apply_lora(teacher_model, adapter_path=teacher_adapter_path)
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        teacher_model.eval()
    if ema_teacher_enabled:
        ema_parameter_names = trainable_parameter_names(model)
        adapter_path = str(get_nested(cfg, "training.adapter_path", "") or "").strip()
        ema_shadow = load_ema_shadow(adapter_path, model, ema_parameter_names)
        if ema_shadow is None and not bool(ema_settings.get("lazy_init", True)):
            ema_shadow = create_ema_shadow(model, ema_parameter_names)
    if distributed:
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main_process(rank):
        (output_dir / "trainable_params.txt").write_text(
            f"trainable={sum(p.numel() for p in trainable)}\n"
            f"total={sum(p.numel() for p in model.parameters())}\n"
            f"distributed={distributed}\nworld_size={world_size}\n"
            f"opsd_teacher_adapter_path={teacher_adapter_path}\n"
            f"opsd_teacher_strategy={get_nested(cfg, 'opsd.teacher_strategy', '')}\n"
            f"opsd_use_ema_teacher={ema_teacher_enabled}\n"
            f"opsd_ema_mode={ema_settings.get('mode', '')}\n"
            f"opsd_ema_decay={ema_settings.get('decay', '')}\n"
            f"opsd_ema_alpha={ema_settings.get('alpha', '')}\n"
            f"opsd_ema_lazy_init={ema_settings.get('lazy_init', '')}\n"
            f"opsd_ema_parameter_count={len(ema_parameter_names)}\n"
            f"opsd_ema_shadow={ema_shadow is not None}\n",
            encoding="utf-8",
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(get_nested(cfg, "training.learning_rate", 2e-5)),
        weight_decay=float(get_nested(cfg, "training.weight_decay", 0.0)),
    )
    save_every = int(get_nested(cfg, "training.save_every", 500))
    grad_accum = int(get_nested(cfg, "training.gradient_accumulation_steps", 1))
    local_steps = remaining_steps // world_size if distributed else remaining_steps
    step = 0
    accum = 0
    start = time.time()
    try:
        while step < local_steps:
            global_index = start_step + step * world_size + rank if distributed else start_step + step
            sample = dataset[global_index % len(dataset)]
            ratio = sample_retention_ratio(cfg, rng)
            try:
                if method == "sft":
                    loss, metrics = sft_like_step(model, processor, sample, cfg, ratio, sample.target)
                elif method == "epic":
                    loss, metrics = epic_tcd_step(model, processor, sample, cfg, ratio)
                elif method == "grpo":
                    loss, metrics = grpo_step(model, processor, sample, cfg, ratio)
                elif method in {"opsd", "opsd_fixed_teacher"}:
                    loss, metrics = opsd_step(
                        model,
                        processor,
                        sample,
                        cfg,
                        ratio,
                        teacher_model=teacher_model,
                        ema_shadow=ema_shadow,
                    )
                elif method == "opsd_nogt":
                    loss, metrics = opsd_nogt_step(
                        model,
                        processor,
                        sample,
                        cfg,
                        ratio,
                        teacher_model=teacher_model,
                        ema_shadow=ema_shadow,
                    )
                elif method == "offpolicy":
                    loss, metrics = offpolicy_step(model, processor, sample, cfg, ratio)
                else:
                    raise AssertionError(method)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: {loss}")
                sync_context = model.no_sync() if distributed and hasattr(model, "no_sync") and (accum + 1) < grad_accum else nullcontext()
                with sync_context:
                    (loss / grad_accum).backward()
                accum += 1
                ema_update_metrics: dict[str, Any] = {}
                if accum >= grad_accum:
                    optimizer.step()
                    if ema_teacher_enabled and ema_shadow is not None:
                        update_ema_shadow(
                            model,
                            ema_shadow,
                            ema_parameter_names,
                            decay=float(ema_settings["decay"]),
                        )
                        ema_update_metrics = {
                            "opsd_ema_update": "updated",
                            "opsd_ema_mode": ema_settings["mode"],
                            "opsd_ema_decay": float(ema_settings["decay"]),
                        }
                    elif ema_teacher_enabled and ema_shadow is None and teacher_model is None:
                        ema_shadow = create_ema_shadow(model, ema_parameter_names)
                        ema_update_metrics = {
                            "opsd_ema_update": "initialized",
                            "opsd_ema_mode": ema_settings["mode"],
                            "opsd_ema_decay": float(ema_settings["decay"]),
                        }
                    elif ema_teacher_enabled and teacher_model is not None:
                        update_ema_teacher(
                            model,
                            teacher_model,
                            ema_parameter_names,
                            decay=float(ema_settings["decay"]),
                        )
                        ema_update_metrics = {
                            "opsd_ema_update": "updated_external_teacher",
                            "opsd_ema_mode": ema_settings["mode"],
                            "opsd_ema_decay": float(ema_settings["decay"]),
                        }
                        teacher_model.eval()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                step += 1
                global_step = min(max_steps, start_step + step * world_size) if distributed else start_step + step
                row = {
                    "step": global_step,
                    "local_step": step,
                    "rank": rank,
                    "world_size": world_size,
                    "method": method,
                    "sample_id": sample.sample_id,
                    "retention_ratio": ratio,
                    "loss": float(loss.detach().cpu()),
                    "elapsed_seconds": time.time() - start,
                    **metrics,
                    **ema_update_metrics,
                }
                if is_main_process(rank):
                    write_jsonl(log_path, row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    if save_every > 0 and global_step % save_every == 0:
                        save_checkpoint(model, output_dir / f"step_{global_step}", ema_shadow=ema_shadow)
            except Exception as exc:
                row = {
                    "step": start_step + (step + 1) * world_size if distributed else start_step + step + 1,
                    "local_step": step + 1,
                    "rank": rank,
                    "method": method,
                    "sample_id": sample.sample_id,
                    "retention_ratio": ratio,
                    "error": repr(exc),
                }
                if is_main_process(rank):
                    write_jsonl(log_path, row)
                raise
        if is_main_process(rank):
            save_checkpoint(model, output_dir / "final", ema_shadow=ema_shadow)
    finally:
        cleanup_distributed(distributed)
    return output_dir


def save_checkpoint(model: Any, path: Path, ema_shadow: dict[str, torch.Tensor] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model = unwrap_model(model)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), path / "pytorch_model.bin")
    save_ema_shadow(path, ema_shadow)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
