#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DISALLOWED_QWEN25_BOOTSTRAP = Path("/scratch/enmingzz/temp/qwen25_bootstrap")
sys.path = [
    path for path in sys.path if not path or not path.startswith(str(DISALLOWED_QWEN25_BOOTSTRAP))
]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from opsd.pruning_distill.losses import (
    compute_ce_loss,
    compute_kd_loss,
    teacher_confidence_weights,
    teacher_entropy,
)
from opsd.pruning_distill.pruners import build_pruner
from opsd.pruning_distill.qwen25_pruned_forward import (
    build_pruned_inputs_embeds,
    compute_full_position_ids,
    extract_next_token_logits,
    get_qwen25_visual_embeds,
    maybe_disable_adapter,
    validate_single_image_qwen_inputs,
)


ARMEN_TRANSFORMERS_SRC = ROOT / "opsd" / "third_party" / "VLMEvalKit_armen51682" / "transformers" / "src"
HF_HUB034_ROOT = Path(os.environ.get("HF_HUB034_ROOT", "/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2"))
TOKENIZERS_QWEN25_ROOT = Path(
    os.environ.get("TOKENIZERS_QWEN25_ROOT", "/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only")
)


@dataclass
class EncodedSample:
    record: dict[str, Any]
    sample_id: str
    question: str
    answer: str
    prompt_inputs: dict[str, torch.Tensor]
    full_inputs: dict[str, torch.Tensor] | None
    prompt_len: int
    answer_token_ids: torch.Tensor | None


@dataclass
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class JsonlOrWandbLogger:
    def __init__(self, output_dir: Path, args: argparse.Namespace) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "training_log.jsonl"
        self.wandb = None
        try:
            import wandb

            wandb.init(project="opsd-qwen25vl-prune-distill", config=vars(args), dir=str(output_dir))
            self.wandb = wandb
        except Exception as exc:
            self.wandb = None
            print(f"[logger] wandb unavailable ({type(exc).__name__}: {exc}); writing {self.jsonl_path}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        clean = {k: _jsonable(v) for k, v in metrics.items()}
        clean["step"] = int(step)
        if self.wandb is not None:
            self.wandb.log(clean, step=step)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_float_list(value: Any, name: str) -> list[float]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).replace(",", " ").replace("[", " ").replace("]", " ").split()
    try:
        return [float(x) for x in values if str(x).strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must contain floats, got {value!r}.") from exc


def parse_keep_ratios(value: Any) -> list[float]:
    ratios = parse_float_list(value, "--keep_ratios")
    if not ratios:
        raise ValueError("--keep_ratios must contain at least one float.")
    for ratio in ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"keep ratios must be in (0, 1], got {ratio}.")
    return ratios


def parse_ratio_sampling_probs(value: Any, expected_len: int) -> list[float] | None:
    probs = parse_float_list(value, "--ratio_sampling_probs")
    if not probs:
        return None
    if len(probs) != expected_len:
        raise ValueError(
            f"--ratio_sampling_probs length {len(probs)} must match keep_ratios length {expected_len}."
        )
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("--ratio_sampling_probs must have positive sum.")
    return [float(p / total) for p in probs]


def parse_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).replace(",", " ").split() if x.strip()]


def maybe_load_yaml_defaults(argv: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    known, _ = parser.parse_known_args(argv)
    if known.config is None:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("--config requires PyYAML to be installed.") from exc
    with open(known.config, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data)!r}.")
    return data


def build_arg_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", default=None)
    p.add_argument("--image_root", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--keep_ratios", default="0.25")
    p.add_argument("--sample_budget_each_step", type=str_to_bool, default=True)
    p.add_argument("--ratio_sampling_probs", default="")
    p.add_argument(
        "--pruner",
        choices=["random", "grid", "divprune_lite", "vscan_stage1", "existing", "keep_all"],
        default="divprune_lite",
    )
    p.add_argument("--student_input_mode", choices=["drop_tokens", "mask_fill_debug"], default="drop_tokens")
    p.add_argument(
        "--distill_mode",
        choices=["teacher_rollout", "on_policy", "mixed", "gold_prefix_debug"],
        default="teacher_rollout",
    )
    p.add_argument("--on_policy_mix", type=float, default=0.0)
    p.add_argument("--loss", choices=["kl_only", "kd_ce"], default="kl_only")
    p.add_argument("--enable_ce_baseline", type=str_to_bool, default=False)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--kd_alpha", type=float, default=1.0)
    p.add_argument("--ce_alpha", type=float, default=0.0)
    p.add_argument("--kd_topk", type=int, default=0)
    p.add_argument("--teacher_confidence_weighting", type=str_to_bool, default=True)
    p.add_argument("--filter_teacher_wrong", type=str_to_bool, default=False)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--bf16", type=str_to_bool, default=True)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--ddp_backend", default="nccl")
    p.add_argument("--ddp_timeout_minutes", type=int, default=60)
    p.add_argument("--disable_lora", type=str_to_bool, default=False)
    p.add_argument("--adapter_path", default="", help="Optional PEFT LoRA adapter checkpoint for warmup continuation.")
    p.add_argument("--divprune_grid_floor", type=str_to_bool, default=False)
    p.add_argument("--divprune_grid_size", type=int, default=4)
    p.add_argument("--divprune_chunk_size", type=int, default=8192)
    p.add_argument("--vscan_grid_size", type=int, default=4)
    p.add_argument("--vscan_score_mode", choices=["cosine_mean", "norm"], default="cosine_mean")
    p.add_argument("--vscan_global_fraction", type=float, default=0.5)
    p.add_argument("--vscan_merge_dropped", type=str_to_bool, default=False)
    p.set_defaults(**defaults)
    return p


def bootstrap_qwen25() -> None:
    if find_spec("transformers.models.qwen2_5_vl") is not None:
        return
    path_roots = [
        str(ARMEN_TRANSFORMERS_SRC) if ARMEN_TRANSFORMERS_SRC.exists() else "",
        str(HF_HUB034_ROOT) if HF_HUB034_ROOT.exists() else "",
        str(TOKENIZERS_QWEN25_ROOT) if TOKENIZERS_QWEN25_ROOT.exists() else "",
    ]
    path_roots = [path for path in path_roots if path]
    if not path_roots:
        return
    for package_name in ["transformers", "huggingface_hub", "tokenizers"]:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
    local_transformers = str(ROOT / "vlm" / "official_thinking_in_space" / "transformers" / "src")
    disallowed_bootstrap = str(DISALLOWED_QWEN25_BOOTSTRAP)
    sys.path = [
        path
        for path in sys.path
        if path != local_transformers and path not in path_roots and not path.startswith(disallowed_bootstrap)
    ]
    for path in reversed(path_roots):
        sys.path.insert(0, path)


def import_qwen25_modules():
    bootstrap_qwen25()
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except Exception:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    return AutoProcessor, Qwen2_5_VLForConditionalGeneration


def resolve_attn_implementation(requested: str) -> str:
    requested = str(requested).strip()
    if requested == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as exc:
            warnings.warn(f"flash_attention_2 requested but flash_attn is unavailable: {exc}; falling back to sdpa.")
            return "sdpa"
    return requested


def setup_distributed(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return DistributedContext()
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device_map = {"": local_rank}
    elif str(args.device_map).lower() == "auto":
        args.device_map = "none"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=args.ddp_backend,
            timeout=timedelta(minutes=max(1, int(args.ddp_timeout_minutes))),
        )
    return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def distributed_barrier(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def average_trainable_gradients(model: Any, ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(ctx.world_size)


def load_one_model(model_cls: Any, name: str, dtype: torch.dtype, attn_implementation: str, device_map: Any) -> Any:
    kwargs = {"torch_dtype": dtype, "attn_implementation": attn_implementation}
    if device_map and str(device_map).lower() != "none" and torch.cuda.is_available():
        kwargs["device_map"] = device_map
    try:
        return model_cls.from_pretrained(name, **kwargs)
    except Exception:
        if attn_implementation == "flash_attention_2":
            warnings.warn("Loading with flash_attention_2 failed; retrying with sdpa.")
            kwargs["attn_implementation"] = "sdpa"
            return model_cls.from_pretrained(name, **kwargs)
        raise


def load_model_bundle(args: argparse.Namespace):
    _, model_cls = import_qwen25_modules()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    attn_impl = resolve_attn_implementation(args.attn_implementation)
    base_model = load_one_model(model_cls, args.model_name_or_path, dtype, attn_impl, args.device_map)
    teacher_model = None

    if args.disable_lora:
        warnings.warn("--disable_lora=True trains the whole student model; this is intended only for debugging.")
        for p in base_model.parameters():
            p.requires_grad_(True)
        teacher_model = load_one_model(model_cls, args.model_name_or_path, dtype, attn_impl, args.device_map).eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        return base_model, teacher_model

    try:
        from peft import LoraConfig, PeftModel, get_peft_model

        if args.adapter_path:
            model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)
            if not hasattr(model, "disable_adapter"):
                warnings.warn("PEFT model has no disable_adapter(); loading a separate frozen teacher model.")
                teacher_model = load_one_model(model_cls, args.model_name_or_path, dtype, attn_impl, args.device_map).eval()
                for p in teacher_model.parameters():
                    p.requires_grad_(False)
            return model, teacher_model

        targets = parse_string_list(args.target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        if not hasattr(model, "disable_adapter"):
            warnings.warn("PEFT model has no disable_adapter(); loading a separate frozen teacher model.")
            teacher_model = load_one_model(model_cls, args.model_name_or_path, dtype, attn_impl, args.device_map).eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
        return model, teacher_model
    except Exception as exc:
        warnings.warn(f"PEFT LoRA unavailable ({type(exc).__name__}: {exc}); using two full model objects.")
        teacher_model = load_one_model(model_cls, args.model_name_or_path, dtype, attn_impl, args.device_map).eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        return base_model, teacher_model


def primary_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in inputs.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def model_input_subset(inputs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "inputs_embeds",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
        "mm_token_type_ids",
    }
    return {k: v for k, v in inputs.items() if k in allowed}


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("sample_id", str(line_no - 1))
            records.append(record)
    return records


def image_path_for(record: dict[str, Any], image_root: str) -> Path:
    image_value = record.get("image")
    if isinstance(image_value, (list, tuple)):
        raise NotImplementedError("Multiple images per sample are not supported yet.")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError("Each JSONL record must contain an 'image' string.")
    path = Path(image_value)
    if not path.is_absolute():
        path = Path(image_root) / path if image_root else path
    return path


def format_question(record: dict[str, Any]) -> str:
    question = str(record.get("question", "")).strip()
    choices = record.get("choices")
    if choices is not None:
        if not isinstance(choices, list) or not all(isinstance(x, str) for x in choices):
            raise ValueError("'choices' must be a list of strings when provided.")
        question = question + "\n" + "\n".join(choices) + "\nAnswer with the option letter only."
    return question


def messages_for(question: str, answer: str | None = None, add_generation_prompt: bool = True) -> tuple[list[dict[str, Any]], bool]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        add_generation_prompt = False
    return messages, add_generation_prompt


def encode_sample(
    processor: Any,
    record: dict[str, Any],
    image_root: str,
    device: torch.device,
) -> EncodedSample:
    answer = str(record.get("answer", ""))
    question = format_question(record)
    image_path = image_path_for(record, image_root)
    image = Image.open(image_path).convert("RGB")

    prompt_messages, prompt_gen = messages_for(question, None, add_generation_prompt=True)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=prompt_gen)
    prompt_inputs = dict(processor(text=[prompt_text], images=[image], return_tensors="pt"))
    prompt_inputs = move_inputs(prompt_inputs, device)
    validate_single_image_qwen_inputs(prompt_inputs)

    prompt_ids = prompt_inputs["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    full_inputs = None
    answer_token_ids = None
    if answer:
        full_messages, full_gen = messages_for(question, answer, add_generation_prompt=False)
        full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=full_gen)
        full_inputs = dict(processor(text=[full_text], images=[image], return_tensors="pt"))
        full_inputs = move_inputs(full_inputs, device)
        validate_single_image_qwen_inputs(full_inputs)
        full_ids = full_inputs["input_ids"]
        common_len = min(prompt_len, int(full_ids.shape[1]))
        prefix_matches = torch.equal(prompt_ids[0, :common_len], full_ids[0, :common_len]) and common_len == prompt_len
        if not prefix_matches:
            raise ValueError(
                "Prompt-only tokenization is not a prefix of full prompt+answer tokenization; "
                "cannot safely compute answer span."
            )
        answer_token_ids = full_ids[0, prompt_len:].clone()
        if int(answer_token_ids.numel()) == 0:
            raise ValueError("Answer span is empty after comparing prompt-only and full tokenization.")

    return EncodedSample(
        record=record,
        sample_id=str(record.get("sample_id", record.get("id", ""))),
        question=question,
        answer=answer,
        prompt_inputs=prompt_inputs,
        full_inputs=full_inputs,
        prompt_len=prompt_len,
        answer_token_ids=answer_token_ids,
    )


@contextmanager
def teacher_forward_context(model: Any, teacher_model: Any | None):
    if teacher_model is not None:
        was_training = teacher_model.training
        teacher_model.eval()
        with torch.no_grad():
            yield teacher_model
        teacher_model.train(was_training)
    else:
        was_training = model.training
        model.eval()
        with maybe_disable_adapter(model), torch.no_grad():
            yield model
        model.train(was_training)


def choose_keep_ratio(
    args: argparse.Namespace,
    keep_ratios: list[float],
    step_index: int,
    ratio_sampling_probs: list[float] | None = None,
) -> float:
    if args.sample_budget_each_step:
        if ratio_sampling_probs is not None:
            return float(random.choices(keep_ratios, weights=ratio_sampling_probs, k=1)[0])
        return float(random.choice(keep_ratios))
    return float(keep_ratios[step_index % len(keep_ratios)])


def assert_answer_alignment(pruned: dict[str, Any], prompt_len: int, answer_token_ids: torch.Tensor) -> None:
    answer_positions = torch.arange(prompt_len, prompt_len + int(answer_token_ids.numel()), device=answer_token_ids.device)
    mapped = pruned["metadata"]["full_to_student"].to(answer_token_ids.device).index_select(0, answer_positions)
    if bool((mapped < 0).any().item()):
        raise AssertionError("A text answer token was dropped from the student sequence.")
    student_answer_ids = pruned["input_ids"][0].to(answer_token_ids.device).index_select(0, mapped)
    if not torch.equal(student_answer_ids, answer_token_ids):
        raise AssertionError("Student answer token ids differ from teacher answer token ids after pruning.")


def build_student_inputs(
    model: Any,
    encoded_inputs: dict[str, torch.Tensor],
    keep_indices: torch.Tensor,
    mode: str,
    prompt_len: int | None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    image_embeds = get_qwen25_visual_embeds(model, encoded_inputs)
    full_position_ids = compute_full_position_ids(
        model,
        encoded_inputs["input_ids"],
        encoded_inputs.get("image_grid_thw"),
        encoded_inputs.get("video_grid_thw"),
        encoded_inputs.get("attention_mask"),
        encoded_inputs.get("second_per_grid_ts"),
        encoded_inputs.get("mm_token_type_ids"),
    )
    pruned = build_pruned_inputs_embeds(
        model,
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
        full_position_ids,
        image_embeds,
        keep_indices,
        mode=mode,
        prompt_len=prompt_len,
        full_mm_token_type_ids=encoded_inputs.get("mm_token_type_ids"),
    )
    return pruned, image_embeds, full_position_ids


def run_gold_prefix_debug_step(
    model: Any,
    teacher_model: Any | None,
    sample: EncodedSample,
    pruner: Any,
    keep_ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if sample.full_inputs is None or sample.answer_token_ids is None:
        raise ValueError("gold_prefix_debug requires records with non-empty ground-truth 'answer'.")
    full_inputs = sample.full_inputs
    with teacher_forward_context(model, teacher_model) as teacher:
        teacher_outputs = teacher(**model_input_subset(full_inputs), use_cache=False, return_dict=True)
        teacher_logits_answer = extract_next_token_logits(
            teacher_outputs.logits,
            sample.prompt_len,
            int(sample.answer_token_ids.numel()),
        ).detach()

    if args.filter_teacher_wrong:
        pred_ids = teacher_logits_answer.argmax(dim=-1).to(sample.answer_token_ids.device)
        if not torch.equal(pred_ids, sample.answer_token_ids):
            return None

    vision_embeds = get_qwen25_visual_embeds(model, full_inputs)
    keep_indices = pruner.select(
        vision_embeds,
        full_inputs.get("image_grid_thw"),
        keep_ratio,
        question=sample.question,
        metadata={"sample_id": sample.sample_id},
    )
    full_position_ids = compute_full_position_ids(
        model,
        full_inputs["input_ids"],
        full_inputs.get("image_grid_thw"),
        full_inputs.get("video_grid_thw"),
        full_inputs.get("attention_mask"),
        full_inputs.get("second_per_grid_ts"),
        full_inputs.get("mm_token_type_ids"),
    )
    pruned = build_pruned_inputs_embeds(
        model,
        full_inputs["input_ids"],
        full_inputs["attention_mask"],
        full_position_ids,
        vision_embeds,
        keep_indices,
        mode=args.student_input_mode,
        prompt_len=sample.prompt_len,
        full_mm_token_type_ids=full_inputs.get("mm_token_type_ids"),
    )
    assert_answer_alignment(pruned, sample.prompt_len, sample.answer_token_ids)

    student_outputs = model(
        inputs_embeds=pruned["inputs_embeds"],
        attention_mask=pruned["attention_mask"],
        position_ids=pruned["position_ids"],
        use_cache=False,
        return_dict=True,
    )
    student_prompt_len = int(pruned["metadata"]["student_prompt_len"])
    student_logits_answer = extract_next_token_logits(
        student_outputs.logits,
        student_prompt_len,
        int(sample.answer_token_ids.numel()),
    )

    weights = teacher_confidence_weights(teacher_logits_answer) if args.teacher_confidence_weighting else None
    kd_loss = compute_kd_loss(
        teacher_logits_answer,
        student_logits_answer,
        temperature=args.temperature,
        topk=args.kd_topk,
        direction="teacher_forward",
        weights=weights,
    )
    if not args.enable_ce_baseline or args.loss == "kl_only":
        ce_loss = torch.zeros_like(kd_loss)
        total_loss = kd_loss
    else:
        ce_loss = compute_ce_loss(student_logits_answer, sample.answer_token_ids)
        total_loss = args.kd_alpha * kd_loss + args.ce_alpha * ce_loss
    entropy = teacher_entropy(teacher_logits_answer).mean()

    return {
        "loss": total_loss,
        "total_loss": total_loss.detach(),
        "kd_loss": kd_loss.detach(),
        "ce_loss": ce_loss.detach(),
        "teacher_entropy": entropy.detach(),
        "num_full_visual_tokens": pruned["metadata"]["num_full_visual_tokens"],
        "num_kept_visual_tokens": pruned["metadata"]["num_kept_visual_tokens"],
        "keep_ratio": keep_ratio,
        "step_mode": "gold_prefix_debug",
        "generated_tokens": int(sample.answer_token_ids.numel()),
    }


def append_suffix_to_inputs(inputs: dict[str, torch.Tensor], suffix_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    suffix_ids = suffix_ids.to(device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    out = {k: v for k, v in inputs.items()}
    out["input_ids"] = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
    suffix_mask = torch.ones_like(suffix_ids, dtype=inputs["attention_mask"].dtype)
    out["attention_mask"] = torch.cat([inputs["attention_mask"], suffix_mask], dim=1)
    if "mm_token_type_ids" in inputs:
        suffix_types = torch.zeros_like(suffix_ids, dtype=inputs["mm_token_type_ids"].dtype)
        out["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], suffix_types], dim=1)
    return out


def generate_student_tokens(model: Any, pruned_prompt: dict[str, Any], max_new_tokens: int, processor: Any) -> torch.Tensor:
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    was_training = model.training
    model.eval()
    gen_kwargs = {
        "input_ids": pruned_prompt["input_ids"],
        "inputs_embeds": pruned_prompt["inputs_embeds"],
        "attention_mask": pruned_prompt["attention_mask"],
        "position_ids": pruned_prompt["position_ids"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    if "mm_token_type_ids" in pruned_prompt:
        gen_kwargs["mm_token_type_ids"] = pruned_prompt["mm_token_type_ids"]
    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)
    model.train(was_training)
    prompt_len = int(pruned_prompt["input_ids"].shape[1])
    if int(output_ids.shape[1]) > prompt_len:
        return output_ids[:, prompt_len:]
    return output_ids


def generate_teacher_tokens(
    model: Any,
    teacher_model: Any | None,
    prompt_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    processor: Any,
) -> torch.Tensor:
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    with teacher_forward_context(model, teacher_model) as teacher:
        output_ids = teacher.generate(
            **model_input_subset(prompt_inputs),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    if int(output_ids.shape[1]) > prompt_len:
        return output_ids[:, prompt_len:]
    return output_ids


def run_rollout_kd_step(
    model: Any,
    teacher_model: Any | None,
    processor: Any,
    sample: EncodedSample,
    pruner: Any,
    keep_ratio: float,
    args: argparse.Namespace,
    rollout_source: str,
) -> dict[str, Any] | None:
    prompt_inputs = sample.prompt_inputs
    prompt_vision_embeds = get_qwen25_visual_embeds(model, prompt_inputs)
    keep_indices = pruner.select(
        prompt_vision_embeds,
        prompt_inputs.get("image_grid_thw"),
        keep_ratio,
        question=sample.question,
        metadata={"sample_id": sample.sample_id},
    )
    prompt_position_ids = compute_full_position_ids(
        model,
        prompt_inputs["input_ids"],
        prompt_inputs.get("image_grid_thw"),
        prompt_inputs.get("video_grid_thw"),
        prompt_inputs.get("attention_mask"),
        prompt_inputs.get("second_per_grid_ts"),
        prompt_inputs.get("mm_token_type_ids"),
    )
    pruned_prompt = build_pruned_inputs_embeds(
        model,
        prompt_inputs["input_ids"],
        prompt_inputs["attention_mask"],
        prompt_position_ids,
        prompt_vision_embeds,
        keep_indices,
        mode=args.student_input_mode,
        prompt_len=int(prompt_inputs["input_ids"].shape[1]),
        full_mm_token_type_ids=prompt_inputs.get("mm_token_type_ids"),
    )
    if rollout_source == "student":
        generated_ids = generate_student_tokens(model, pruned_prompt, args.max_new_tokens, processor)
    elif rollout_source == "teacher":
        generated_ids = generate_teacher_tokens(model, teacher_model, prompt_inputs, args.max_new_tokens, processor)
    else:
        raise ValueError("rollout_source must be 'student' or 'teacher'.")
    if int(generated_ids.numel()) == 0:
        return None

    teacher_inputs = append_suffix_to_inputs(prompt_inputs, generated_ids)
    student_full_inputs = append_suffix_to_inputs(prompt_inputs, generated_ids)
    with teacher_forward_context(model, teacher_model) as teacher:
        teacher_outputs = teacher(**model_input_subset(teacher_inputs), use_cache=False, return_dict=True)
        teacher_logits_answer = extract_next_token_logits(
            teacher_outputs.logits,
            int(prompt_inputs["input_ids"].shape[1]),
            int(generated_ids.shape[1]),
        ).detach()

    student_position_ids = compute_full_position_ids(
        model,
        student_full_inputs["input_ids"],
        student_full_inputs.get("image_grid_thw"),
        student_full_inputs.get("video_grid_thw"),
        student_full_inputs.get("attention_mask"),
        student_full_inputs.get("second_per_grid_ts"),
        student_full_inputs.get("mm_token_type_ids"),
    )
    pruned_student = build_pruned_inputs_embeds(
        model,
        student_full_inputs["input_ids"],
        student_full_inputs["attention_mask"],
        student_position_ids,
        prompt_vision_embeds,
        keep_indices,
        mode=args.student_input_mode,
        prompt_len=int(prompt_inputs["input_ids"].shape[1]),
        full_mm_token_type_ids=student_full_inputs.get("mm_token_type_ids"),
    )
    student_outputs = model(
        inputs_embeds=pruned_student["inputs_embeds"],
        attention_mask=pruned_student["attention_mask"],
        position_ids=pruned_student["position_ids"],
        use_cache=False,
        return_dict=True,
    )
    student_logits_answer = extract_next_token_logits(
        student_outputs.logits,
        int(pruned_student["metadata"]["student_prompt_len"]),
        int(generated_ids.shape[1]),
    )

    weights = teacher_confidence_weights(teacher_logits_answer) if args.teacher_confidence_weighting else None
    kd_loss = compute_kd_loss(
        teacher_logits_answer,
        student_logits_answer,
        temperature=args.temperature,
        topk=args.kd_topk,
        direction="teacher_forward",
        weights=weights,
    )
    ce_loss = torch.zeros_like(kd_loss)
    entropy = teacher_entropy(teacher_logits_answer).mean()
    step_mode = "on_policy" if rollout_source == "student" else "teacher_rollout"
    return {
        "loss": kd_loss,
        "total_loss": kd_loss.detach(),
        "kd_loss": kd_loss.detach(),
        "ce_loss": ce_loss.detach(),
        "teacher_entropy": entropy.detach(),
        "num_full_visual_tokens": pruned_student["metadata"]["num_full_visual_tokens"],
        "num_kept_visual_tokens": pruned_student["metadata"]["num_kept_visual_tokens"],
        "keep_ratio": keep_ratio,
        "step_mode": step_mode,
        "generated_tokens": int(generated_ids.shape[1]),
    }


def run_on_policy_step(
    model: Any,
    teacher_model: Any | None,
    processor: Any,
    sample: EncodedSample,
    pruner: Any,
    keep_ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    return run_rollout_kd_step(
        model,
        teacher_model,
        processor,
        sample,
        pruner,
        keep_ratio,
        args,
        rollout_source="student",
    )


def run_teacher_rollout_step(
    model: Any,
    teacher_model: Any | None,
    processor: Any,
    sample: EncodedSample,
    pruner: Any,
    keep_ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    return run_rollout_kd_step(
        model,
        teacher_model,
        processor,
        sample,
        pruner,
        keep_ratio,
        args,
        rollout_source="teacher",
    )


def choose_step_mode(args: argparse.Namespace) -> str:
    if args.distill_mode == "gold_prefix_debug":
        return "gold_prefix_debug"
    if args.distill_mode == "on_policy":
        return "on_policy"
    if args.distill_mode == "teacher_rollout":
        return "teacher_rollout"
    return "on_policy" if random.random() < float(args.on_policy_mix) else "teacher_rollout"


def trainable_parameter_count(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def assert_no_frozen_gradients(model: Any, teacher_model: Any | None) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad and param.grad is not None:
            raise AssertionError(f"Frozen student/base parameter received a gradient: {name}")
    if teacher_model is not None:
        for name, param in teacher_model.named_parameters():
            if param.grad is not None:
                raise AssertionError(f"Frozen teacher parameter received a gradient: {name}")


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def save_checkpoint(model: Any, processor: Any, output_dir: Path, step: int) -> None:
    ckpt = output_dir / f"step_{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ckpt)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(ckpt)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    defaults = maybe_load_yaml_defaults(argv)
    args = build_arg_parser(defaults).parse_args(argv)
    dist_ctx = setup_distributed(args)
    if args.loss != "kl_only" and not args.enable_ce_baseline:
        warnings.warn("--loss is not kl_only but --enable_ce_baseline is false; forcing pure KL.")
        args.loss = "kl_only"
    if args.enable_ce_baseline and args.ce_alpha <= 0.0:
        warnings.warn("--enable_ce_baseline is true but --ce_alpha <= 0; CE will still have zero weight.")
    random.seed(args.seed + dist_ctx.rank)
    torch.manual_seed(args.seed + dist_ctx.rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist_ctx.is_main:
        with (output_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        try:
            import yaml

            with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(vars(args), f, sort_keys=True)
        except Exception:
            with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, sort_keys=True)
    distributed_barrier(dist_ctx)

    AutoProcessor, _ = import_qwen25_modules()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    model, teacher_model = load_model_bundle(args)
    model.train()
    trainable, total = trainable_parameter_count(model)
    if dist_ctx.is_main:
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.4f}%)")
        if dist_ctx.enabled:
            print(
                "Distributed training: manual gradient all-reduce, "
                f"world_size={dist_ctx.world_size}, local_rank={dist_ctx.local_rank}"
            )
        with (output_dir / "trainable_params.txt").open("w", encoding="utf-8") as f:
            f.write(f"trainable_parameters: {trainable}\n")
            f.write(f"total_parameters: {total}\n")
            f.write(f"trainable_percent: {100.0 * trainable / max(total, 1):.6f}\n")
            f.write(f"distributed_world_size: {dist_ctx.world_size}\n")
        if teacher_model is None:
            print("Teacher path: same model with LoRA adapter disabled.")
        else:
            print("Teacher path: separate frozen model object.")
    distributed_barrier(dist_ctx)

    device = primary_device(model)
    records = read_jsonl(args.train_jsonl)
    if not records:
        raise ValueError("--train_jsonl contains no records.")
    keep_ratios = parse_keep_ratios(args.keep_ratios)
    ratio_sampling_probs = parse_ratio_sampling_probs(args.ratio_sampling_probs, len(keep_ratios))
    pruner = build_pruner(
        args.pruner,
        seed=args.seed,
        divprune_grid_floor=args.divprune_grid_floor,
        divprune_grid_size=args.divprune_grid_size,
        divprune_chunk_size=args.divprune_chunk_size,
        vscan_grid_size=args.vscan_grid_size,
        vscan_score_mode=args.vscan_score_mode,
        vscan_global_fraction=args.vscan_global_fraction,
        vscan_merge_dropped=args.vscan_merge_dropped,
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    logger = JsonlOrWandbLogger(output_dir, args) if dist_ctx.is_main else None

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    started = time.perf_counter()
    try:
        for epoch in range(args.num_train_epochs):
            order = list(range(len(records)))
            order_rng = random.Random(args.seed + epoch)
            order_rng.shuffle(order)
            for order_pos, idx in enumerate(order):
                if dist_ctx.enabled and order_pos % dist_ctx.world_size != dist_ctx.rank:
                    continue
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
                sample_t0 = time.perf_counter()
                sample = encode_sample(processor, records[idx], args.image_root, device)
                keep_ratio = choose_keep_ratio(args, keep_ratios, micro_step, ratio_sampling_probs)
                step_mode = choose_step_mode(args)
                if step_mode == "on_policy":
                    result = run_on_policy_step(model, teacher_model, processor, sample, pruner, keep_ratio, args)
                elif step_mode == "teacher_rollout":
                    result = run_teacher_rollout_step(model, teacher_model, processor, sample, pruner, keep_ratio, args)
                else:
                    result = run_gold_prefix_debug_step(model, teacher_model, sample, pruner, keep_ratio, args)
                if result is None:
                    continue

                loss = result["loss"] / int(args.gradient_accumulation_steps)
                loss.backward()
                micro_step += 1
                assert_no_frozen_gradients(model, teacher_model)

                if micro_step % int(args.gradient_accumulation_steps) == 0:
                    average_trainable_gradients(model, dist_ctx)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    metrics = {
                        **{k: v for k, v in result.items() if k != "loss"},
                        "epoch": epoch,
                        "sample_id": sample.sample_id,
                        "gpu_memory_gb": gpu_memory_gb(),
                        "samples_per_sec": float((micro_step * dist_ctx.world_size) / elapsed),
                        "sample_latency_sec": time.perf_counter() - sample_t0,
                        "student_input_mode": args.student_input_mode,
                        "distill_mode": args.distill_mode,
                        "loss_type": "kl_only" if not args.enable_ce_baseline else args.loss,
                        "pruner": args.pruner,
                        "distributed_world_size": dist_ctx.world_size,
                    }
                    if dist_ctx.is_main and (global_step % args.log_every == 0 or global_step == 1):
                        assert logger is not None
                        logger.log(metrics, global_step)
                        print(json.dumps({k: _jsonable(v) for k, v in metrics.items()}, ensure_ascii=False))
                    if args.save_every > 0 and global_step % args.save_every == 0:
                        if dist_ctx.is_main:
                            save_checkpoint(model, processor, output_dir, global_step)
                        distributed_barrier(dist_ctx)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if dist_ctx.is_main:
            save_checkpoint(model, processor, output_dir, global_step)
        distributed_barrier(dist_ctx)
    finally:
        if logger is not None:
            logger.close()
        cleanup_distributed(dist_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
