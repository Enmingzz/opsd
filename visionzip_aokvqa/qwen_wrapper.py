from __future__ import annotations

import math
import os
import sys
import types
import warnings
from contextlib import contextmanager, nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import Any


DISALLOWED_QWEN25_BOOTSTRAP = Path("/scratch/enmingzz/temp/qwen25_bootstrap")


def _drop_disallowed_qwen25_bootstrap_path() -> None:
    disallowed_bootstrap = str(DISALLOWED_QWEN25_BOOTSTRAP)
    sys.path = [path for path in sys.path if not path or not path.startswith(disallowed_bootstrap)]


_drop_disallowed_qwen25_bootstrap_path()

import torch
import torch.nn.functional as F

from opsd.pruning_distill.qwen25_pruned_forward import (
    _unwrap_qwen_model,
    extract_next_token_logits,
    maybe_disable_adapter,
    validate_single_image_qwen_inputs,
)

from .aokvqa import FormattedAOKVQASample, resolve_image
from .prompting import format_chat_messages, format_chat_with_assistant, parse_final_answer


ROOT = Path(__file__).resolve().parents[2]
HF_HUB034_ROOT = Path(os.environ.get("HF_HUB034_ROOT", "/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2"))
TOKENIZERS_QWEN25_ROOT = Path(
    os.environ.get("TOKENIZERS_QWEN25_ROOT", "/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only")
)
ARMEN_TRANSFORMERS_SRC = ROOT / "opsd" / "third_party" / "VLMEvalKit_armen51682" / "transformers" / "src"
OFFICIAL_VISIONZIP_QWEN25 = Path(
    os.environ.get(
        "VISIONZIP_QWEN25VL_ROOT",
        os.environ.get(
            "OFFICIAL_VISIONZIP_QWEN25VL_ROOT",
            str(ROOT / "opsd" / "third_party" / "VisionZip" / "Qwen2_5_VL"),
        ),
    )
)
VISIONZIP_NO_PRUNE_DOMINANT = 0.999999
VISIONZIP_NO_PRUNE_CONTEXTUAL = 0.000001


def bootstrap_qwen25() -> None:
    if not ARMEN_TRANSFORMERS_SRC.exists() and not TOKENIZERS_QWEN25_ROOT.exists() and not HF_HUB034_ROOT.exists():
        return
    _drop_disallowed_qwen25_bootstrap_path()
    hf_hub034 = str(HF_HUB034_ROOT) if HF_HUB034_ROOT.exists() else ""
    tokenizers_root = str(TOKENIZERS_QWEN25_ROOT) if TOKENIZERS_QWEN25_ROOT.exists() else ""
    armen_transformers = str(ARMEN_TRANSFORMERS_SRC) if ARMEN_TRANSFORMERS_SRC.exists() else ""
    disallowed_bootstrap = str(DISALLOWED_QWEN25_BOOTSTRAP)
    shadow_transformers = {
        str(ROOT / "vlm" / "official_thinking_in_space" / "transformers" / "src"),
    }
    sys.path = [
        path
        for path in sys.path
        if path
        and path not in shadow_transformers
        and not path.startswith(disallowed_bootstrap)
        and path != hf_hub034
        and path != tokenizers_root
        and path != armen_transformers
    ]
    if tokenizers_root:
        sys.path.insert(0, tokenizers_root)
    if hf_hub034:
        sys.path.insert(0, hf_hub034)
    if armen_transformers:
        sys.path.insert(0, armen_transformers)
    allowed_roots = tuple(path for path in (armen_transformers, hf_hub034, tokenizers_root) if path)
    for package_name in ["transformers", "huggingface_hub", "tokenizers"]:
        module = sys.modules.get(package_name)
        module_file = str(getattr(module, "__file__", "")) if module is not None else ""
        if module is None or (allowed_roots and module_file.startswith(allowed_roots)):
            continue
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)


def import_qwen25_modules():
    bootstrap_qwen25()
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not ARMEN_TRANSFORMERS_SRC.exists():
        raise RuntimeError(
            "A patched Qwen2.5-VL VisionZip source is required for training. "
            f"Missing Armen transformers directory: {ARMEN_TRANSFORMERS_SRC}"
        )
    return Qwen2_5_VLForConditionalGeneration, AutoConfig, AutoProcessor


def flash_attention_available() -> bool:
    return find_spec("flash_attn") is not None


def compute_default_rope_parameters_for_visionzip(
    config: Any,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    del seq_len, layer_type
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    base = float(rope_scaling.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
    partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", None) or int(config.hidden_size) // int(config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def resolve_attn_implementation(requested: str, require_visionzip_flash: bool = False) -> str:
    requested = str(requested or "sdpa")
    if require_visionzip_flash and requested != "flash_attention_2":
        raise RuntimeError(
            "Official VisionZip Qwen2.5-VL training requires training.attn_implementation=flash_attention_2; "
            "the VisionZip visual attention path needs flash-attn logits for token scoring."
        )
    if requested == "flash_attention_2" and not flash_attention_available():
        if require_visionzip_flash:
            raise RuntimeError(
                "training.attn_implementation=flash_attention_2 was requested, but flash_attn is unavailable. "
                "Activate an environment with flash-attn before running official VisionZip training."
            )
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


def normalize_qwen25vl_config_for_official_visionzip(config: Any, pad_token_id: int) -> Any:
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_act",
            "rms_norm_eps",
            "rope_scaling",
            "attention_dropout",
            "max_position_embeddings",
            "initializer_range",
            "use_cache",
            "use_sliding_window",
            "sliding_window",
            "max_window_layers",
            "bos_token_id",
            "eos_token_id",
        ):
            if getattr(config, name, None) is None and hasattr(text_config, name):
                setattr(config, name, getattr(text_config, name))
    if getattr(config, "pad_token_id", None) is None:
        setattr(config, "pad_token_id", int(pad_token_id))
    return config


def ensure_peft_tensor_parallel_compat() -> None:
    """Provide no-op tensor-parallel hooks for PEFT with older Transformers."""
    module_name = "transformers.integrations.tensor_parallel"
    if module_name in sys.modules or find_spec(module_name) is not None:
        return

    module = types.ModuleType(module_name)

    class ColwiseParallel:
        pass

    class RowwiseParallel:
        pass

    class EmbeddingParallel:
        pass

    module.ALL_PARALLEL_STYLES = {}
    module.ColwiseParallel = ColwiseParallel
    module.RowwiseParallel = RowwiseParallel
    module.EmbeddingParallel = EmbeddingParallel
    sys.modules[module_name] = module


def load_qwen_model_and_processor(
    model_name_or_path: str,
    bf16: bool = True,
    attn_implementation: str = "flash_attention_2",
    device_map: str | None = "auto",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    visionzip_official: bool = True,
):
    model_cls, config_cls, processor_cls = import_qwen25_modules()
    attn_impl = resolve_attn_implementation(attn_implementation, require_visionzip_flash=bool(visionzip_official))
    dtype = torch.bfloat16 if bf16 else torch.float16
    processor_kwargs: dict[str, Any] = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = int(max_pixels)
    processor = processor_cls.from_pretrained(model_name_or_path, **processor_kwargs)
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    config = config_cls.from_pretrained(model_name_or_path)
    pad_token_id = getattr(config, "pad_token_id", None) or getattr(processor.tokenizer, "pad_token_id", None) or getattr(
        processor.tokenizer, "eos_token_id", None
    )
    if pad_token_id is None:
        raise ValueError("Could not infer pad_token_id from the Qwen2.5-VL processor tokenizer.")
    normalize_qwen25vl_config_for_official_visionzip(config, int(pad_token_id))
    kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype,
        "attn_implementation": attn_impl,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        kwargs["device_map"] = device_map
    model = model_cls.from_pretrained(model_name_or_path, **kwargs)
    setattr(model, "visionzip_disable", not bool(visionzip_official))
    set_visionzip_ratios(
        model,
        dominant_ratio=VISIONZIP_NO_PRUNE_DOMINANT,
        contextual_ratio=VISIONZIP_NO_PRUNE_CONTEXTUAL,
    )
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
        ensure_peft_tensor_parallel_compat()
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
    stop_strings: tuple[str, ...] = (),
    max_unparseable_tokens: int | None = None,
    stop_on_parse: bool = True,
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
            if (stop_strings or max_unparseable_tokens is not None) and len(generated) % 4 == 0:
                partial = decode_token_ids(processor, torch.cat(generated, dim=1))
                parsed = parse_final_answer(partial) if stop_on_parse else None
                if any(stop in partial for stop in stop_strings) or (stop_on_parse and parsed is not None):
                    break
                if max_unparseable_tokens is not None and len(generated) >= int(max_unparseable_tokens):
                    break

    if not generated:
        empty = pruned["input_ids"].new_empty((1, 0))
        return empty, ""
    gen_ids = torch.cat(generated, dim=1)
    return gen_ids, decode_token_ids(processor, gen_ids)


def _visionzip_model_targets(model: Any) -> list[Any]:
    targets: list[Any] = []
    queue = [model]
    seen: set[int] = set()
    while queue:
        item = queue.pop(0)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, "visionzip_dominant_ratio") or hasattr(item, "visual"):
            targets.append(item)
        if hasattr(item, "get_base_model"):
            try:
                queue.append(item.get_base_model())
            except TypeError:
                pass
        for attr in ("module", "base_model", "model"):
            queue.append(getattr(item, attr, None))
    qwen = _unwrap_qwen_model(model)
    if id(qwen) not in {id(x) for x in targets}:
        targets.append(qwen)
    return targets


def set_visionzip_ratios(model: Any, dominant_ratio: float, contextual_ratio: float) -> None:
    for target in _visionzip_model_targets(model):
        setattr(target, "visionzip_dominant_ratio", float(dominant_ratio))
        setattr(target, "visionzip_contextual_ratio", float(contextual_ratio))


@contextmanager
def temporary_visionzip_ratios(model: Any, retention_ratio: float):
    retention = min(max(float(retention_ratio), 0.0), 1.0)
    contextual = min(0.05, retention)
    dominant = max(0.0, retention - contextual)
    targets = _visionzip_model_targets(model)
    previous = [
        (
            target,
            getattr(target, "visionzip_dominant_ratio", None),
            getattr(target, "visionzip_contextual_ratio", None),
        )
        for target in targets
    ]
    set_visionzip_ratios(model, dominant, contextual)
    try:
        yield dominant, contextual
    finally:
        for target, dominant_prev, contextual_prev in previous:
            if dominant_prev is None:
                try:
                    delattr(target, "visionzip_dominant_ratio")
                except AttributeError:
                    pass
            else:
                setattr(target, "visionzip_dominant_ratio", dominant_prev)
            if contextual_prev is None:
                try:
                    delattr(target, "visionzip_contextual_ratio")
                except AttributeError:
                    pass
            else:
                setattr(target, "visionzip_contextual_ratio", contextual_prev)


def visionzip_ratio_from_split(dominant_ratio: float, contextual_ratio: float) -> float:
    del contextual_ratio
    return max(0.0, min(1.0, 1.0 - float(dominant_ratio)))


def last_visionzip_pruned_inputs(model: Any) -> dict[str, Any] | None:
    qwen = _unwrap_qwen_model(model)
    for candidate in (qwen, getattr(qwen, "model", None), getattr(getattr(qwen, "model", None), "model", None)):
        last = getattr(candidate, "_last_visionzip_pruned_inputs", None)
        if isinstance(last, dict):
            return last
    return None


def official_visionzip_metadata(
    model: Any,
    inputs: dict[str, torch.Tensor],
    prompt_len: int | None,
    dominant_ratio: float,
    contextual_ratio: float,
) -> dict[str, Any]:
    qwen = _unwrap_qwen_model(model)
    last = last_visionzip_pruned_inputs(model)
    if not isinstance(last, dict) or "input_ids" not in last:
        raise RuntimeError("Official VisionZip forward did not expose _last_visionzip_pruned_inputs.")
    image_token_id = int(qwen.config.image_token_id)
    full_input_ids = inputs["input_ids"]
    pruned_input_ids = last["input_ids"].to(device=full_input_ids.device)
    num_full = int((full_input_ids == image_token_id).sum().item())
    num_kept = int((pruned_input_ids == image_token_id).sum().item())
    dominant_num = min(num_full, max(0, int(float(dominant_ratio) * num_full)))
    contextual_num = max(0, int(float(contextual_ratio) * num_full))
    if contextual_ratio > 0 and dominant_num < num_full:
        contextual_num = max(contextual_num, 1)
    contextual_num = min(contextual_num, max(0, num_full - dominant_num))
    if num_kept != dominant_num + contextual_num:
        raise RuntimeError(
            "Official VisionZip token count mismatch: "
            f"kept={num_kept}, dominant+contextual={dominant_num + contextual_num}."
        )
    student_prompt_len = None
    if prompt_len is not None:
        full_prompt_image_tokens = int((full_input_ids[:, :prompt_len] == image_token_id).sum().item())
        student_prompt_len = int(prompt_len) - full_prompt_image_tokens + num_kept
    return {
        "student_prompt_len": student_prompt_len,
        "num_full_visual_tokens": num_full,
        "num_kept_visual_tokens": num_kept,
        "visionzip_exact_metrics": True,
        "visionzip_metric_source": "armen_qwen25vl_visionzip",
        "visionzip_target_tokens": num_kept,
        "visionzip_dominant_tokens": dominant_num,
        "visionzip_contextual_tokens": contextual_num,
        "visionzip_merged_tokens": max(0, num_full - dominant_num - contextual_num),
        "visionzip_contextual_fraction": float(contextual_ratio),
        "visionzip_dominant_ratio": float(dominant_ratio),
        "visionzip_contextual_ratio": float(contextual_ratio),
    }


def official_model_kwargs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model_input_subset(inputs)


def forward_pruned(
    model: Any,
    inputs: dict[str, torch.Tensor],
    retention_ratio: float,
    prompt_len: int | None = None,
    mode: str = "drop_tokens",
    allow_embedding_fallback: bool = False,
    **forward_kwargs: Any,
):
    del mode
    if allow_embedding_fallback:
        raise ValueError("Embedding fallback is disabled: training must use official VisionZip metrics.")
    validate_single_image_qwen_inputs(inputs, model=model)
    kwargs = official_model_kwargs(inputs)
    kwargs["use_cache"] = False
    kwargs.update(forward_kwargs)
    with temporary_visionzip_ratios(model, retention_ratio) as (dominant_ratio, contextual_ratio):
        kwargs["enable_visionzip"] = True
        kwargs["visionzip_ratio"] = visionzip_ratio_from_split(dominant_ratio, contextual_ratio)
        outputs = model(**kwargs)
        metadata = official_visionzip_metadata(model, inputs, prompt_len, dominant_ratio, contextual_ratio)
    return outputs, {"metadata": metadata}


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
    max_unparseable_tokens: int | None = None,
    stop_on_parse: bool = True,
) -> tuple[torch.Tensor, str, dict[str, Any]]:
    if allow_embedding_fallback:
        raise ValueError("Embedding fallback is disabled: generation must use official VisionZip metrics.")
    validate_single_image_qwen_inputs(prompt_inputs, model=model)
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or eos_token_id
    kwargs = {
        **official_model_kwargs(prompt_inputs),
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
    generate_model = getattr(model, "module", model)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    with temporary_visionzip_ratios(model, retention_ratio) as (dominant_ratio, contextual_ratio):
        kwargs["enable_visionzip"] = True
        kwargs["visionzip_ratio"] = visionzip_ratio_from_split(dominant_ratio, contextual_ratio)
        if manual_decode:
            with torch.no_grad():
                prefill_outputs = generate_model(
                    **official_model_kwargs(prompt_inputs),
                    use_cache=False,
                    enable_visionzip=True,
                    visionzip_ratio=visionzip_ratio_from_split(dominant_ratio, contextual_ratio),
                )
            del prefill_outputs
            metadata = official_visionzip_metadata(model, prompt_inputs, prompt_len, dominant_ratio, contextual_ratio)
            pruned = last_visionzip_pruned_inputs(model)
            if not isinstance(pruned, dict) or "inputs_embeds" not in pruned:
                raise RuntimeError("Official VisionZip forward did not expose pruned inputs for manual generation.")
            gen_ids, text = manual_generate_pruned(
                model,
                processor,
                pruned,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=top_k,
                eos_token_id=eos_token_id,
                stop_strings=("</answer>",),
                max_unparseable_tokens=max_unparseable_tokens,
                stop_on_parse=bool(stop_on_parse),
            )
            return gen_ids, text, metadata
        output_ids = generate_model.generate(**kwargs)
        metadata = official_visionzip_metadata(model, prompt_inputs, prompt_len, dominant_ratio, contextual_ratio)
    text = decode_new_tokens(processor, output_ids, prompt_len)
    return output_ids[:, prompt_len:], text, metadata


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
