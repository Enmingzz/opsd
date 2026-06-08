#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw

from opsd.pruning_distill.losses import compute_kd_loss
from opsd.pruning_distill.pruners import BaseVisualTokenPruner, build_pruner
from opsd.pruning_distill.qwen25_pruned_forward import (
    build_pruned_inputs_embeds,
    compute_full_position_ids,
    extract_next_token_logits,
    get_qwen25_visual_embeds,
    maybe_disable_adapter,
    validate_single_image_qwen_inputs,
)
from opsd.scripts.train_qwen25vl_prune_distill import (
    append_suffix_to_inputs,
    generate_teacher_tokens,
    import_qwen25_modules,
    load_model_bundle,
    messages_for,
    model_input_subset,
    move_inputs,
    primary_device,
    str_to_bool,
    teacher_forward_context,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--bf16", type=str_to_bool, default=True)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--max_new_tokens", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p


def make_image(width: int, height: int, index: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (245, 245, 240))
    draw = ImageDraw.Draw(image)
    step = max(32, min(width, height) // 8)
    colors = [(180, 50, 50), (50, 120, 200), (60, 160, 90), (210, 170, 40)]
    for y in range(0, height, step):
        for x in range(0, width, step):
            color = colors[((x // step) + (y // step) + index) % len(colors)]
            draw.rectangle([x, y, min(width, x + step) - 1, min(height, y + step) - 1], fill=color)
    draw.ellipse([width // 4, height // 4, width // 2, height // 2], fill=(30, 30, 30))
    draw.rectangle([width // 2, height // 2, width - 20, height - 20], outline=(255, 255, 255), width=6)
    return image


def encode_prompt(processor: Any, image: Image.Image, device: torch.device) -> dict[str, torch.Tensor]:
    messages, add_generation_prompt = messages_for("Describe the image in detail.", None, add_generation_prompt=True)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    inputs = dict(processor(text=[prompt_text], images=[image], return_tensors="pt"))
    inputs = move_inputs(inputs, device)
    validate_single_image_qwen_inputs(inputs)
    return inputs


def pruned_forward(
    model: Any,
    inputs: dict[str, torch.Tensor],
    image_embeds: torch.Tensor,
    keep_indices: torch.Tensor,
    mode: str = "drop_tokens",
    prompt_len: int | None = None,
) -> tuple[Any, dict[str, Any], torch.Tensor]:
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
        image_embeds,
        keep_indices,
        mode=mode,
        prompt_len=int(inputs["input_ids"].shape[1]) if prompt_len is None else int(prompt_len),
        full_mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    outputs = model(
        inputs_embeds=pruned["inputs_embeds"],
        attention_mask=pruned["attention_mask"],
        position_ids=pruned["position_ids"],
        use_cache=False,
        return_dict=True,
    )
    return outputs, pruned, position_ids


def make_train_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        model_name_or_path=args.model_name_or_path,
        bf16=args.bf16,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        disable_lora=False,
        adapter_path="",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    AutoProcessor, _ = import_qwen25_modules()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model, teacher_model = load_model_bundle(make_train_args(args))
    model.eval()
    device = primary_device(model)

    sizes = [(448, 448), (672, 672), (896, 672)]
    pruner_names = ["keep_all", "random", "grid", "divprune_lite", "vscan_stage1"]
    keep_ratios = [1.0, 0.5, 0.25, 0.125]
    token_counts: dict[str, int] = {}
    max_keep_all_diff = 0.0

    for image_index, size in enumerate(sizes):
        image = make_image(size[0], size[1], image_index)
        prompt_inputs = encode_prompt(processor, image, device)
        image_embeds = get_qwen25_visual_embeds(model, prompt_inputs)
        num_placeholders = int((prompt_inputs["input_ids"] == model.config.image_token_id).sum().item())
        token_counts[f"{size[0]}x{size[1]}"] = int(image_embeds.shape[0])
        full_position_ids = compute_full_position_ids(
            model,
            prompt_inputs["input_ids"],
            prompt_inputs.get("image_grid_thw"),
            prompt_inputs.get("video_grid_thw"),
            prompt_inputs.get("attention_mask"),
            prompt_inputs.get("second_per_grid_ts"),
            prompt_inputs.get("mm_token_type_ids"),
        )

        with teacher_forward_context(model, teacher_model) as teacher:
            full_outputs = teacher(**model_input_subset(prompt_inputs), use_cache=False, return_dict=True)
            keep_all = build_pruner("keep_all")
            keep_all_indices = keep_all.select(image_embeds, prompt_inputs.get("image_grid_thw"), 1.0)
            pruned_outputs, pruned_keep_all, _ = pruned_forward(teacher, prompt_inputs, image_embeds, keep_all_indices)
            diff = float((full_outputs.logits - pruned_outputs.logits).abs().max().item())
            max_keep_all_diff = max(max_keep_all_diff, diff)
            print(json.dumps({"image_size": size, "keep_all_max_abs_logits_diff": diff}))
            if diff >= 1e-4:
                raise AssertionError(f"keep_all drop_tokens diff too large for {size}: {diff}")

        generated_ids = generate_teacher_tokens(model, teacher_model, prompt_inputs, args.max_new_tokens, processor)
        sequence_inputs = append_suffix_to_inputs(prompt_inputs, generated_ids)
        with teacher_forward_context(model, teacher_model) as teacher:
            teacher_outputs = teacher(**model_input_subset(sequence_inputs), use_cache=False, return_dict=True)
            teacher_logits_answer = extract_next_token_logits(
                teacher_outputs.logits,
                int(prompt_inputs["input_ids"].shape[1]),
                int(generated_ids.shape[1]),
            ).detach()
        sequence_embeds = get_qwen25_visual_embeds(model, sequence_inputs)

        for pruner_name in pruner_names:
            for keep_ratio in keep_ratios:
                pruner = build_pruner(
                    pruner_name,
                    seed=args.seed,
                    divprune_grid_floor=True,
                    divprune_grid_size=4,
                    vscan_grid_size=4,
                    vscan_merge_dropped=False,
                )
                keep_indices = pruner.select(
                    sequence_embeds,
                    sequence_inputs.get("image_grid_thw"),
                    keep_ratio,
                    question="Describe the image in detail.",
                    metadata={"sample_id": f"{size[0]}x{size[1]}"},
                )
                sorted_indices = bool(torch.equal(keep_indices, keep_indices.sort().values))
                expected = int(sequence_embeds.shape[0]) if pruner_name == "keep_all" else BaseVisualTokenPruner._target_count(int(sequence_embeds.shape[0]), keep_ratio)
                if int(keep_indices.numel()) != expected:
                    raise AssertionError(f"{pruner_name} ratio={keep_ratio} kept {int(keep_indices.numel())}, expected {expected}")
                if not sorted_indices:
                    raise AssertionError(f"{pruner_name} returned unsorted keep indices.")

                outputs, pruned, before_position_ids = pruned_forward(
                    model,
                    sequence_inputs,
                    sequence_embeds,
                    keep_indices,
                    prompt_len=int(prompt_inputs["input_ids"].shape[1]),
                )
                if torch.isnan(outputs.logits).any():
                    raise AssertionError(f"NaN logits for {pruner_name} ratio={keep_ratio} size={size}.")
                student_logits_answer = extract_next_token_logits(
                    outputs.logits,
                    int(pruned["metadata"]["student_prompt_len"]),
                    int(generated_ids.shape[1]),
                )
                kd = compute_kd_loss(teacher_logits_answer, student_logits_answer, temperature=2.0)
                if torch.isnan(kd):
                    raise AssertionError(f"NaN KL loss for {pruner_name} ratio={keep_ratio} size={size}.")

                info = {
                    "image_size": f"{size[0]}x{size[1]}",
                    "pruner": pruner_name,
                    "keep_ratio": keep_ratio,
                    "input_ids_shape": list(sequence_inputs["input_ids"].shape),
                    "attention_mask_shape": list(sequence_inputs["attention_mask"].shape),
                    "image_grid_thw": sequence_inputs["image_grid_thw"].detach().cpu().tolist(),
                    "num_image_token_id_placeholders": num_placeholders,
                    "num_visual_embeds": int(sequence_embeds.shape[0]),
                    "num_full_visual_tokens": int(pruned["metadata"]["num_full_visual_tokens"]),
                    "num_kept_visual_tokens": int(pruned["metadata"]["num_kept_visual_tokens"]),
                    "keep_indices_count": int(keep_indices.numel()),
                    "keep_indices_min": int(keep_indices.min().item()),
                    "keep_indices_max": int(keep_indices.max().item()),
                    "keep_indices_sorted": sorted_indices,
                    "position_ids_shape_before_pruning": list(before_position_ids.shape),
                    "position_ids_shape_after_pruning": list(pruned["position_ids"].shape),
                    "attention_mask_shape_after_pruning": list(pruned["attention_mask"].shape),
                    "kd_loss": float(kd.detach().cpu().item()),
                }
                print(json.dumps(info))

    # Gradient sanity check: teacher/base frozen params should not receive gradients.
    model.train()
    model.zero_grad(set_to_none=True)
    image = make_image(448, 448, 99)
    prompt_inputs = encode_prompt(processor, image, device)
    generated_ids = generate_teacher_tokens(model, teacher_model, prompt_inputs, args.max_new_tokens, processor)
    sequence_inputs = append_suffix_to_inputs(prompt_inputs, generated_ids)
    with teacher_forward_context(model, teacher_model) as teacher:
        teacher_outputs = teacher(**model_input_subset(sequence_inputs), use_cache=False, return_dict=True)
        teacher_logits_answer = extract_next_token_logits(
            teacher_outputs.logits,
            int(prompt_inputs["input_ids"].shape[1]),
            int(generated_ids.shape[1]),
        ).detach()
    image_embeds = get_qwen25_visual_embeds(model, sequence_inputs)
    keep_indices = build_pruner("divprune_lite").select(image_embeds, sequence_inputs.get("image_grid_thw"), 0.25)
    outputs, pruned, _ = pruned_forward(
        model,
        sequence_inputs,
        image_embeds,
        keep_indices,
        prompt_len=int(prompt_inputs["input_ids"].shape[1]),
    )
    student_logits_answer = extract_next_token_logits(outputs.logits, int(pruned["metadata"]["student_prompt_len"]), int(generated_ids.shape[1]))
    loss = compute_kd_loss(teacher_logits_answer, student_logits_answer, temperature=2.0)
    loss.backward()
    trainable_with_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    frozen_with_grad = [name for name, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    if not trainable_with_grad:
        raise AssertionError("No trainable LoRA parameter received gradients.")
    if frozen_with_grad:
        raise AssertionError(f"Frozen/base parameters received gradients: {frozen_with_grad[:5]}")

    print(
        json.dumps(
            {
                "normal_resolution_stress_passed": True,
                "full_visual_token_counts": token_counts,
                "max_keep_all_abs_logits_diff": max_keep_all_diff,
                "trainable_params_with_grad": len(trainable_with_grad),
                "frozen_params_with_grad": len(frozen_with_grad),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
