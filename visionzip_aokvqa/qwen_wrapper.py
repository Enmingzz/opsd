from __future__ import annotations

import math
import os
import sys
import warnings
from contextlib import contextmanager, nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from opsd.pruning_distill.qwen25_pruned_forward import (
    _get_visual_module,
    _unwrap_qwen_model,
    build_pruned_inputs_embeds,
    compute_full_position_ids,
    extract_next_token_logits,
    maybe_disable_adapter,
    validate_single_image_qwen_inputs,
)

from .aokvqa import FormattedAOKVQASample, resolve_image
from .prompting import format_chat_messages, format_chat_with_assistant
from .visionzip import VisionZipOutput, VisionZipPruner


ROOT = Path(__file__).resolve().parents[2]
QWEN25_BOOTSTRAP = Path("/scratch/enmingzz/temp/qwen25_bootstrap")


def bootstrap_qwen25() -> None:
    if find_spec("transformers.models.qwen2_5_vl") is None and QWEN25_BOOTSTRAP.exists():
        for package_name in ["transformers", "huggingface_hub", "tokenizers", "safetensors", "qwen_vl_utils"]:
            for name in list(sys.modules):
                if name == package_name or name.startswith(f"{package_name}."):
                    sys.modules.pop(name, None)
        local_transformers = str(ROOT / "vlm" / "official_thinking_in_space" / "transformers" / "src")
        sys.path = [path for path in sys.path if path != local_transformers]
        sys.path.insert(0, str(QWEN25_BOOTSTRAP))


def import_qwen25_modules():
    bootstrap_qwen25()
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError("Qwen2.5-VL support is unavailable in transformers.") from exc
    return Qwen2_5_VLForConditionalGeneration, AutoProcessor


def flash_attention_available() -> bool:
    return find_spec("flash_attn") is not None


def resolve_attn_implementation(requested: str) -> str:
    requested = str(requested or "sdpa")
    if requested == "flash_attention_2" and not flash_attention_available():
        warnings.warn("flash_attention_2 requested but flash_attn is unavailable; falling back to sdpa.", stacklevel=2)
        return "sdpa"
    return requested


def str_to_torch_dtype(name: str, bf16: bool = True) -> torch.dtype:
    if bf16:
        return torch.bfloat16
    lowered = str(name).lower()
    if lowered in {"float16", "fp16"}:
        return torch.float16
    return torch.float32


def load_qwen_model_and_processor(
    model_name_or_path: str,
    bf16: bool = True,
    attn_implementation: str = "flash_attention_2",
    device_map: str | None = "auto",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
):
    model_cls, processor_cls = import_qwen25_modules()
    attn_impl = resolve_attn_implementation(attn_implementation)
    dtype = torch.bfloat16 if bf16 else torch.float16
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": attn_impl,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        kwargs["device_map"] = device_map
    model = model_cls.from_pretrained(model_name_or_path, **kwargs)
    processor_kwargs: dict[str, Any] = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = int(max_pixels)
    processor = processor_cls.from_pretrained(model_name_or_path, **processor_kwargs)
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def apply_lora(
    model: Any,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
    adapter_path: str = "",
) -> Any:
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except Exception as exc:
        raise RuntimeError("PEFT is required for this experiment's trainable student adapters.") from exc

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        return model
    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def primary_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def model_input_subset(inputs: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
        "mm_token_type_ids",
    }
    return {key: value for key, value in inputs.items() if key in keep and value is not None}


def encode_prompt(
    processor: Any,
    sample: FormattedAOKVQASample,
    image_root: str | Path = "",
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    return encode_prompt_text(processor, sample, sample.prompt, image_root=image_root, device=device)


def encode_prompt_text(
    processor: Any,
    sample: FormattedAOKVQASample,
    prompt: str,
    image_root: str | Path = "",
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    image = resolve_image(sample.image, image_root=image_root)
    text = processor.apply_chat_template(format_chat_messages(prompt), tokenize=False, add_generation_prompt=True)
    inputs = dict(processor(text=[text], images=[image], return_tensors="pt"))
    if device is not None:
        inputs = move_inputs(inputs, device)
    validate_single_image_qwen_inputs(inputs)
    return inputs


def encode_prompt_and_response(
    processor: Any,
    sample: FormattedAOKVQASample,
    response: str,
    image_root: str | Path = "",
    device: torch.device | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    image = resolve_image(sample.image, image_root=image_root)
    prompt_text = processor.apply_chat_template(format_chat_messages(sample.prompt), tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(
        format_chat_with_assistant(sample.prompt, response),
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_inputs = dict(processor(text=[prompt_text], images=[image], return_tensors="pt"))
    full_inputs = dict(processor(text=[full_text], images=[image], return_tensors="pt"))
    if device is not None:
        prompt_inputs = move_inputs(prompt_inputs, device)
        full_inputs = move_inputs(full_inputs, device)
    validate_single_image_qwen_inputs(prompt_inputs)
    validate_single_image_qwen_inputs(full_inputs)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    answer_ids = full_inputs["input_ids"][0, prompt_len:].detach().clone()
    if answer_ids.numel() == 0:
        raise ValueError("Encoded response produced zero answer tokens.")
    return prompt_inputs, full_inputs, answer_ids


def decode_new_tokens(processor: Any, output_ids: torch.Tensor, prompt_len: int) -> str:
    new_ids = output_ids[:, int(prompt_len) :]
    return processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def decode_token_ids(processor: Any, token_ids: torch.Tensor) -> str:
    return processor.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    temperature = max(float(temperature), 1e-6)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), int(probs.shape[-1]))
        values, indices = torch.topk(probs, k=k, dim=-1)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=indices, src=values)
        probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    top_p = float(top_p)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sampled)
    return torch.multinomial(probs, num_samples=1)


def manual_generate_pruned(
    model: Any,
    processor: Any,
    pruned: dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int | None,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, str]:
    """Autoregress from pruned embeddings without HF generation cache.

    Qwen2.5-VL's cached generation path assumes dense cache positions. VisionZip
    drop-token pruning preserves sparse MRoPE position ids from the full visual
    sequence, which can make SDPA's generated causal mask disagree with the KV
    length. Recomputing the short pruned sequence each step avoids that mismatch.
    """

    generate_model = getattr(model, "module", model)
    embed_tokens = generate_model.get_input_embeddings()
    input_ids = pruned["input_ids"]
    inputs_embeds = pruned["inputs_embeds"]
    attention_mask = pruned["attention_mask"]
    position_ids = pruned["position_ids"]
    mm_token_type_ids = pruned.get("mm_token_type_ids")
    generated: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            kwargs = {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": False,
            }
            if mm_token_type_ids is not None:
                kwargs["mm_token_type_ids"] = mm_token_type_ids
            outputs = generate_model(**kwargs)
            next_token = _sample_next_token(outputs.logits[:, -1, :], do_sample, temperature, top_p, top_k)
            generated.append(next_token)
            input_ids = torch.cat([input_ids, next_token.to(device=input_ids.device, dtype=input_ids.dtype)], dim=1)
            next_embed = embed_tokens(next_token.to(device=inputs_embeds.device)).to(dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            one = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, one], dim=1)
            position_ids = torch.cat([position_ids, position_ids[:, :, -1:] + 1], dim=2)
            if mm_token_type_ids is not None:
                mm_zero = torch.zeros((mm_token_type_ids.shape[0], 1), device=mm_token_type_ids.device, dtype=mm_token_type_ids.dtype)
                mm_token_type_ids = torch.cat([mm_token_type_ids, mm_zero], dim=1)
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                break

    if not generated:
        empty = pruned["input_ids"].new_empty((1, 0))
        return empty, ""
    gen_ids = torch.cat(generated, dim=1)
    return gen_ids, decode_token_ids(processor, gen_ids)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _last_attention_metric(attn: Any, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]):
    seq_length = int(hidden_states.shape[0])
    q, k, _ = attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = _apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    qh = q.permute(1, 0, 2).float()
    kh = k.permute(1, 0, 2).float()
    scaling = float(getattr(attn, "scaling", 1.0 / math.sqrt(max(1, qh.shape[-1]))))
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scaling
    attn_probs = F.softmax(logits, dim=-1)
    return attn_probs, kh


def _standard_visual_with_metrics(visual: Any, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
    """Run recent transformers Qwen visual code while extracting VisionZip metrics."""

    hidden_states = visual.patch_embed(pixel_values)
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, device=hidden_states.device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    spatial_merge_unit = int(visual.spatial_merge_unit)
    hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    attn_probs = None
    attn_key = None
    blocks = list(visual.blocks)
    for layer_num, blk in enumerate(blocks):
        if layer_num in set(getattr(visual, "fullatt_block_indexes", [])):
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if layer_num == len(blocks) - 1:
            attn_probs, attn_key = _last_attention_metric(blk.attn, blk.norm1(hidden_states), position_embeddings)
        hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    merged_hidden_states = visual.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    merged_hidden_states = merged_hidden_states[reverse_indices, :]
    if attn_probs is None or attn_key is None:
        raise RuntimeError("Could not extract VisionZip last-layer visual attention metrics.")

    with torch.no_grad():
        attn_mean = attn_probs.mean(dim=0).sum(dim=0)
        if attn_mean.numel() % spatial_merge_unit != 0:
            raise RuntimeError("VisionZip attention metric is incompatible with Qwen spatial merge unit.")
        attn_mean = attn_mean.view(attn_mean.shape[0] // spatial_merge_unit, spatial_merge_unit).mean(dim=-1)
        attn_mean = attn_mean[reverse_indices]
        attn_key = attn_key.view(attn_key.shape[0], attn_key.shape[1] // spatial_merge_unit, spatial_merge_unit, attn_key.shape[-1])
        attn_key = attn_key.mean(dim=2)
        attn_key = attn_key[:, reverse_indices, :].mean(dim=0).unsqueeze(0)
    return merged_hidden_states, attn_mean, attn_key


def get_qwen25_visionzip_visual_outputs(
    model: Any,
    inputs: dict[str, Any],
    allow_embedding_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    validate_single_image_qwen_inputs(inputs, model=model)
    qwen = _unwrap_qwen_model(model)
    visual = _get_visual_module(qwen)
    try:
        visual_device = next(visual.parameters()).device
    except StopIteration:
        visual_device = primary_device(qwen)
    pixel_values = inputs["pixel_values"].to(device=visual_device)
    image_grid_thw = inputs["image_grid_thw"].to(device=visual_device)
    visual_dtype = getattr(visual, "dtype", None)
    if visual_dtype is not None:
        pixel_values = pixel_values.to(dtype=visual_dtype)

    with torch.no_grad():
        try:
            out = visual(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            out = visual(pixel_values, image_grid_thw=image_grid_thw)
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            image_embeds, attn_scores, attn_key = out[0], out[1], out[2]
            source = "visionzip_qwen_visual"
        else:
            try:
                image_embeds, attn_scores, attn_key = _standard_visual_with_metrics(visual, pixel_values, image_grid_thw)
                source = "standard_qwen_manual_metric"
            except Exception:
                if not allow_embedding_fallback:
                    raise
                from opsd.pruning_distill.qwen25_pruned_forward import get_qwen25_visual_embeds

                image_embeds = get_qwen25_visual_embeds(model, inputs)
                attn_scores = None
                attn_key = None
                source = "embedding_fallback"

    if image_embeds.ndim == 3 and int(image_embeds.shape[0]) == 1:
        image_embeds = image_embeds[0]
    image_token_id = int(qwen.config.image_token_id)
    n_image_tokens = int((inputs["input_ids"] == image_token_id).sum().item())
    if int(image_embeds.shape[0]) != n_image_tokens:
        raise ValueError(f"Image features and placeholders do not match: {image_embeds.shape[0]} vs {n_image_tokens}.")
    return image_embeds, attn_scores, attn_key, {"visionzip_metric_source": source}


def build_visionzip_pruned_inputs(
    model: Any,
    inputs: dict[str, torch.Tensor],
    retention_ratio: float,
    prompt_len: int | None = None,
    mode: str = "drop_tokens",
    allow_embedding_fallback: bool = False,
) -> dict[str, Any]:
    image_embeds, attn_scores, attn_key, metric_meta = get_qwen25_visionzip_visual_outputs(
        model,
        inputs,
        allow_embedding_fallback=allow_embedding_fallback,
    )
    pruner = VisionZipPruner(allow_embedding_fallback=allow_embedding_fallback)
    vz = pruner.select_and_merge(
        image_embeds,
        inputs.get("image_grid_thw"),
        retention_ratio,
        attn_scores=attn_scores,
        attn_key=attn_key,
        metadata=metric_meta,
    )
    position_ids = compute_full_position_ids(
        model,
        inputs["input_ids"],
        inputs.get("image_grid_thw"),
        inputs.get("video_grid_thw"),
        inputs.get("attention_mask"),
        inputs.get("second_per_grid_ts"),
        inputs.get("mm_token_type_ids"),
    )
    pruned = build_pruned_inputs_embeds(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        position_ids,
        vz.image_embeds,
        vz.keep_indices,
        mode=mode,
        prompt_len=prompt_len,
        full_mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    pruned["metadata"].update(vz.metadata)
    return pruned


def forward_pruned(
    model: Any,
    inputs: dict[str, torch.Tensor],
    retention_ratio: float,
    prompt_len: int | None = None,
    mode: str = "drop_tokens",
    allow_embedding_fallback: bool = False,
    **forward_kwargs: Any,
):
    pruned = build_visionzip_pruned_inputs(
        model,
        inputs,
        retention_ratio,
        prompt_len=prompt_len,
        mode=mode,
        allow_embedding_fallback=allow_embedding_fallback,
    )
    kwargs = {
        "input_ids": pruned["input_ids"],
        "inputs_embeds": pruned["inputs_embeds"],
        "attention_mask": pruned["attention_mask"],
        "position_ids": pruned["position_ids"],
        "use_cache": False,
    }
    if "mm_token_type_ids" in pruned:
        kwargs["mm_token_type_ids"] = pruned["mm_token_type_ids"]
    kwargs.update(forward_kwargs)
    outputs = model(**kwargs)
    return outputs, pruned


def generate_pruned(
    model: Any,
    processor: Any,
    prompt_inputs: dict[str, torch.Tensor],
    retention_ratio: float,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int | None = None,
    allow_embedding_fallback: bool = False,
    manual_decode: bool = False,
) -> tuple[torch.Tensor, str, dict[str, Any]]:
    pruned = build_visionzip_pruned_inputs(
        model,
        prompt_inputs,
        retention_ratio,
        prompt_len=int(prompt_inputs["input_ids"].shape[1]),
        mode="drop_tokens",
        allow_embedding_fallback=allow_embedding_fallback,
    )
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or eos_token_id
    if manual_decode:
        gen_ids, text = manual_generate_pruned(
            model,
            processor,
            pruned,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
        return gen_ids, text, pruned["metadata"]
    kwargs = {
        "input_ids": pruned["input_ids"],
        "inputs_embeds": pruned["inputs_embeds"],
        "attention_mask": pruned["attention_mask"],
        "position_ids": pruned["position_ids"],
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "use_cache": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
        if top_k is not None and int(top_k) > 0:
            kwargs["top_k"] = int(top_k)
    if "mm_token_type_ids" in pruned:
        kwargs["mm_token_type_ids"] = pruned["mm_token_type_ids"]
    generate_model = getattr(model, "module", model)
    output_ids = generate_model.generate(**kwargs)
    prompt_len = int(pruned["input_ids"].shape[1])
    text = decode_new_tokens(processor, output_ids, prompt_len)
    return output_ids[:, prompt_len:], text, pruned["metadata"]


@contextmanager
def teacher_adapter_disabled(model: Any):
    with maybe_disable_adapter(model):
        yield model


def extract_generated_logits(
    logits: torch.Tensor,
    prompt_len: int,
    generated_count: int,
) -> torch.Tensor:
    return extract_next_token_logits(logits, prompt_len, generated_count)
