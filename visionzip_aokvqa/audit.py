#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opsd.visionzip_aokvqa.aokvqa import FormattedAOKVQASample, load_aokvqa_dataset
from opsd.visionzip_aokvqa.losses import compute_generalized_jsd
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
from opsd.visionzip_aokvqa.train import (
    build_epic_target,
    generate_full_teacher,
    get_nested,
    grpo_group_advantages,
    load_yaml,
    prompt_mode_from_config,
    sequence_inputs_from_prompt,
)


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/aokvqa_visionzip_opsd.yaml")
    p.add_argument("--reports_dir", default=str(OUTPUT_ROOT / "reports"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sections", default="all")
    p.add_argument("--allow_embedding_fallback", action="store_true")
    p.add_argument("--grpo_group_size", type=int, default=4)
    p.add_argument("--grpo_samples", type=int, default=16)
    return p


def sample_records(cfg: dict[str, Any], count: int, seed: int) -> list[FormattedAOKVQASample]:
    data = load_aokvqa_dataset(
        get_nested(cfg, "dataset.name", "HuggingFaceM4/A-OKVQA"),
        splits=list(get_nested(cfg, "dataset.use_splits", ["train", "validation"])),
        limit=0,
        seed=seed,
        prompt_mode=prompt_mode_from_config(cfg),
    )
    rng = random.Random(seed)
    return rng.sample(data, min(count, len(data)))


def load_model(cfg: dict[str, Any], allow_embedding_fallback: bool):
    if allow_embedding_fallback:
        cfg.setdefault("pruning", {})["allow_embedding_fallback"] = True
    model, processor = load_qwen_model_and_processor(
        str(get_nested(cfg, "base_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        bf16=bool(get_nested(cfg, "training.bf16", True)),
        attn_implementation=str(get_nested(cfg, "training.attn_implementation", "flash_attention_2")),
        device_map=get_nested(cfg, "training.device_map", "auto"),
    )
    model = apply_lora(
        model,
        r=int(get_nested(cfg, "training.lora_r", 16)),
        alpha=int(get_nested(cfg, "training.lora_alpha", 32)),
        dropout=float(get_nested(cfg, "training.lora_dropout", 0.05)),
        target_modules=list(get_nested(cfg, "training.target_modules", [])) or None,
    )
    return model, processor


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path}")


def prompt_leakage_audit(cfg: dict[str, Any], reports_dir: Path, seed: int) -> bool:
    samples = sample_records(cfg, 10, seed)
    lines = [
        "# Prompt Leakage Audit",
        "",
        "Status: PASS unless explicitly marked otherwise.",
        "",
        "The correct option text appears in the options by design. This audit checks that the prompt does not include the target response, exact rationale text, or `Final answer: <letter>`.",
        "",
    ]
    ok = True
    for idx, sample in enumerate(samples, start=1):
        final_answer_leak = f"Final answer: {sample.correct_letter}" in sample.prompt
        target_leak = sample.target.strip() in sample.prompt
        rationale_leak = bool(sample.reasoning and sample.reasoning.strip() in sample.prompt)
        sample_ok = not final_answer_leak and not target_leak and not rationale_leak
        ok = ok and sample_ok
        lines.extend(
            [
                f"## Sample {idx}: {sample.sample_id}",
                "",
                f"- question: {sample.question}",
                f"- A: {sample.options[0]}",
                f"- B: {sample.options[1]}",
                f"- C: {sample.options[2]}",
                f"- D: {sample.options[3]}",
                f"- correct_option_letter: {sample.correct_letter}",
                f"- leakage_check: {'PASS' if sample_ok else 'FAIL'}",
                f"- final_answer_letter_leak: {final_answer_leak}",
                f"- target_response_leak: {target_leak}",
                f"- exact_rationale_leak: {rationale_leak}",
                "",
                "### Input Prompt",
                "```text",
                sample.prompt,
                "```",
                "",
                "### Target Response",
                "```text",
                sample.target,
                "```",
                "",
            ]
        )
    lines[2] = f"Status: {'PASS' if ok else 'FAIL'}"
    write_report(reports_dir / "prompt_audit.md", lines)
    return ok


def visionzip_pruning_audit(model: Any, processor: Any, cfg: dict[str, Any], reports_dir: Path, seed: int) -> bool:
    samples = sample_records(cfg, 4, seed + 1)
    ratios = [0.1, 0.2, 0.3, 0.4]
    lines = ["# VisionZip Pruning Audit", "", "Status: PASS unless explicitly marked otherwise.", ""]
    ok = True
    device = primary_device(model)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            prompt_inputs = encode_prompt(processor, sample, get_nested(cfg, "dataset.image_root", ""), device)
            lines.extend([f"## Sample {sample.sample_id}", ""])
            for ratio in ratios:
                _outputs, pruned = forward_pruned(
                    model,
                    prompt_inputs,
                    ratio,
                    prompt_len=int(prompt_inputs["input_ids"].shape[1]),
                    allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
                )
                before = int(pruned["metadata"]["num_full_visual_tokens"])
                after = int(pruned["metadata"]["num_kept_visual_tokens"])
                expected = int(pruned["metadata"]["visionzip_dominant_tokens"]) + int(
                    pruned["metadata"]["visionzip_contextual_tokens"]
                )
                actual_ratio = after / before if before else 0.0
                count_ok = after == expected
                ok = ok and count_ok
                lines.append(
                    f"- ratio {ratio}: visual_tokens_before={before}, visual_tokens_after={after}, "
                    f"expected_tokens={expected}, expected_retention_ratio={ratio}, "
                    f"actual_retention_ratio={actual_ratio:.4f}, status={'PASS' if count_ok else 'FAIL'}"
                )
            lines.append("")
    lines[2] = f"Status: {'PASS' if ok else 'FAIL'}"
    write_report(reports_dir / "visionzip_audit.md", lines)
    return ok


def grpo_reward_audit(model: Any, processor: Any, cfg: dict[str, Any], reports_dir: Path, seed: int, group_size: int, sample_count: int) -> bool:
    samples = sample_records(cfg, sample_count, seed + 2)
    rng = random.Random(seed + 3)
    lines = [
        "# GRPO Reward Audit",
        "",
        "Status: PASS if at least one sampled group has non-identical rewards.",
        "",
    ]
    variable_groups = 0
    device = primary_device(model)
    model.eval()
    for idx, sample in enumerate(samples, start=1):
        ratio = rng.choice([0.1, 0.2, 0.3, 0.4])
        prompt_inputs = encode_prompt(processor, sample, get_nested(cfg, "dataset.image_root", ""), device)
        rewards: list[float] = []
        parsed_answers: list[str | None] = []
        responses: list[str] = []
        for _ in range(group_size):
            with torch.no_grad():
                _, text, _ = generate_pruned(
                    model,
                    processor,
                    prompt_inputs,
                    ratio,
                    max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
                    do_sample=True,
                    temperature=float(get_nested(cfg, "generation.temperature", 0.7)),
                    top_p=float(get_nested(cfg, "generation.top_p", 0.9)),
                    allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
                )
            parsed = parse_final_answer(text)
            reward = (1.0 if parsed == sample.correct_letter else 0.0) + (0.1 if parsed is not None else 0.0)
            responses.append(text)
            parsed_answers.append(parsed)
            rewards.append(reward)
        mean = statistics.mean(rewards)
        std = statistics.pstdev(rewards)
        if std > 1e-8:
            variable_groups += 1
        advantages = grpo_group_advantages(torch.tensor(rewards)).detach().cpu().tolist()
        lines.extend(
            [
                f"## Sample {idx}: {sample.sample_id}",
                "",
                f"- retention_ratio: {ratio}",
                f"- correct_option_letter: {sample.correct_letter}",
                f"- reward_mean: {mean:.4f}",
                f"- reward_std: {std:.4f}",
                f"- advantages: {advantages}",
                "",
            ]
        )
        for j, response in enumerate(responses, start=1):
            lines.extend(
                [
                    f"### Response {j}",
                    f"- parsed_final_answer: {parsed_answers[j - 1]}",
                    f"- reward: {rewards[j - 1]}",
                    "```text",
                    response,
                    "```",
                    "",
                ]
            )
    ok = variable_groups > 0
    lines[2] = f"Status: {'PASS' if ok else 'FAIL'}"
    lines.extend(
        [
            "## Summary",
            "",
            f"- groups_with_non_identical_rewards: {variable_groups}/{len(samples)}",
        ]
    )
    if not ok:
        lines.extend(
            [
                "",
                "All sampled GRPO groups had identical rewards. Suggested fixes before full GRPO training:",
                "- increase sampling temperature",
                "- increase group size",
                "- sample harder examples",
                "- add reward shaping beyond exact final-option correctness and parseability",
            ]
        )
    write_report(reports_dir / "grpo_reward_audit.md", lines)
    return ok


def epic_target_audit(model: Any, processor: Any, cfg: dict[str, Any], reports_dir: Path, seed: int) -> bool:
    samples = sample_records(cfg, 10, seed + 4)
    lines = [
        "# EPIC Target Audit",
        "",
        "Status: PASS if teacher-incorrect samples are filtered unless explicit correction is configured.",
        "",
        f"- filter_teacher_correct: {get_nested(cfg, 'epic.filter_teacher_correct', True)}",
        f"- correct_final_answer: {get_nested(cfg, 'epic.correct_final_answer', False)}",
        "",
    ]
    ok = True
    device = primary_device(model)
    model.eval()
    for idx, sample in enumerate(samples, start=1):
        prompt_inputs = encode_prompt(processor, sample, get_nested(cfg, "dataset.image_root", ""), device)
        with torch.no_grad(), teacher_adapter_disabled(model):
            _, teacher_text = generate_full_teacher(model, processor, prompt_inputs, cfg)
        parsed = parse_final_answer(teacher_text)
        target, meta = build_epic_target(teacher_text, sample.correct_letter, cfg)
        illegal_correction = (parsed != sample.correct_letter) and target is not None and not bool(
            get_nested(cfg, "epic.correct_final_answer", False)
        )
        ok = ok and not illegal_correction
        lines.extend(
            [
                f"## Sample {idx}: {sample.sample_id}",
                "",
                f"- ground_truth_final_option: {sample.correct_letter}",
                f"- parsed_teacher_final_answer: {parsed}",
                f"- kept: {target is not None}",
                f"- policy: {meta.get('epic_target_policy')}",
                f"- illegal_silent_correction: {illegal_correction}",
                "",
                "### Teacher Generated Response",
                "```text",
                teacher_text,
                "```",
                "",
            ]
        )
    lines[2] = f"Status: {'PASS' if ok else 'FAIL'}"
    write_report(reports_dir / "epic_target_audit.md", lines)
    return ok


def opsd_audit(model: Any, processor: Any, cfg: dict[str, Any], reports_dir: Path, seed: int) -> bool:
    sample = sample_records(cfg, 1, seed + 5)[0]
    ratio = 0.2
    device = primary_device(model)
    model.train()
    model.zero_grad(set_to_none=True)
    prompt_inputs = encode_prompt(processor, sample, get_nested(cfg, "dataset.image_root", ""), device)
    gen_ids, gen_text, gen_meta = generate_pruned(
        model,
        processor,
        prompt_inputs,
        ratio,
        max_new_tokens=int(get_nested(cfg, "generation.max_new_tokens", 128)),
        do_sample=True,
        temperature=float(get_nested(cfg, "generation.temperature", 0.7)),
        top_p=float(get_nested(cfg, "generation.top_p", 0.9)),
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    student_seq_inputs = sequence_inputs_from_prompt(prompt_inputs, gen_ids)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    teacher_prompt = build_opsd_teacher_prompt(
        sample.question,
        sample.options,
        sample.target,
        prompt_mode=prompt_mode_from_config(cfg),
    )
    teacher_prompt_inputs = encode_prompt_text(
        processor,
        sample,
        teacher_prompt,
        get_nested(cfg, "dataset.image_root", ""),
        device,
    )
    teacher_seq_inputs = sequence_inputs_from_prompt(teacher_prompt_inputs, gen_ids)
    teacher_prompt_len = int(teacher_prompt_inputs["input_ids"].shape[1])
    with torch.no_grad(), teacher_adapter_disabled(model):
        teacher_outputs = model(**model_input_subset(teacher_seq_inputs), use_cache=False)
    student_outputs, pruned = forward_pruned(
        model,
        student_seq_inputs,
        ratio,
        prompt_len=prompt_len,
        allow_embedding_fallback=bool(get_nested(cfg, "pruning.allow_embedding_fallback", False)),
    )
    teacher_logits = extract_generated_logits(teacher_outputs.logits, teacher_prompt_len, int(gen_ids.numel()))
    student_logits = extract_generated_logits(student_outputs.logits, int(pruned["metadata"]["student_prompt_len"]), int(gen_ids.numel()))
    teacher_detached = not teacher_logits.requires_grad
    loss = compute_generalized_jsd(
        teacher_logits,
        student_logits,
        beta=float(get_nested(cfg, "opsd.beta", 0.0)),
        temperature=float(get_nested(cfg, "opsd.temperature", 1.0)),
        top_k=int(get_nested(cfg, "opsd.top_k_loss", 0) or 0) or None,
        token_clip=float(get_nested(cfg, "opsd.jsd_token_clip", 0.05) or 0.0) or None,
    )
    loss.backward()
    trainable_with_grad = []
    frozen_with_grad = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.requires_grad:
                trainable_with_grad.append(name)
            else:
                frozen_with_grad.append(name)
    _, _, target_ids = encode_prompt_and_response(
        processor,
        sample,
        sample.target,
        image_root=get_nested(cfg, "dataset.image_root", ""),
        device=device,
    )
    generated_differs_from_gold = not torch.equal(gen_ids.flatten().detach().cpu(), target_ids[: gen_ids.numel()].detach().cpu())
    full_tokens = int(pruned["metadata"]["num_full_visual_tokens"])
    kept_tokens = int(pruned["metadata"]["num_kept_visual_tokens"])
    ok = (
        full_tokens > kept_tokens
        and teacher_detached
        and bool(trainable_with_grad)
        and not frozen_with_grad
        and generated_differs_from_gold
    )
    lines = [
        "# OPSD Audit",
        "",
        f"Status: {'PASS' if ok else 'FAIL'}",
        "",
        f"- sample_id: {sample.sample_id}",
        f"- retention_ratio: {ratio}",
        f"- teacher_uses_full_visual_tokens: true",
        f"- teacher_uses_ground_truth_reference_solution: true",
        f"- student_uses_visionzip_pruned_tokens: {full_tokens > kept_tokens}",
        f"- full_visual_tokens: {full_tokens}",
        f"- kept_visual_tokens: {kept_tokens}",
        f"- student_prompt_tokens: {prompt_len}",
        f"- teacher_prompt_tokens: {teacher_prompt_len}",
        f"- teacher_adapter_disabled_context_used: true",
        f"- teacher_forward_no_grad_used: true",
        f"- teacher_logits_detached: {teacher_detached}",
        f"- generalized_jsd_computed_on_student_generated_token_count: {int(gen_ids.numel())}",
        f"- generated_tokens_are_not_gold_target_prefix: {generated_differs_from_gold}",
        f"- frozen_parameters_with_grad_count: {len(frozen_with_grad)}",
        f"- trainable_parameters_with_grad_count: {len(trainable_with_grad)}",
        f"- loss: {float(loss.detach().cpu())}",
        "",
        "## Student On-Policy Generation",
        "```text",
        gen_text,
        "```",
        "",
        "## Gradient Check",
        "",
        "First trainable parameters with gradients:",
        "```text",
        "\n".join(trainable_with_grad[:20]),
        "```",
        "",
    ]
    if frozen_with_grad:
        lines.extend(["Frozen parameters unexpectedly receiving gradients:", "```text", "\n".join(frozen_with_grad[:20]), "```"])
    write_report(reports_dir / "opsd_audit.md", lines)
    return ok


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config)
    if args.allow_embedding_fallback:
        cfg.setdefault("pruning", {})["allow_embedding_fallback"] = True
    reports_dir = Path(args.reports_dir)
    sections = {"prompt", "visionzip", "grpo", "epic", "opsd"} if args.sections == "all" else set(args.sections.split(","))

    results: dict[str, bool] = {}
    if "prompt" in sections:
        results["prompt"] = prompt_leakage_audit(cfg, reports_dir, args.seed)

    model = processor = None
    if sections & {"visionzip", "grpo", "epic", "opsd"}:
        model, processor = load_model(cfg, args.allow_embedding_fallback)

    if "visionzip" in sections:
        results["visionzip"] = visionzip_pruning_audit(model, processor, cfg, reports_dir, args.seed)
    if "grpo" in sections:
        results["grpo"] = grpo_reward_audit(model, processor, cfg, reports_dir, args.seed, args.grpo_group_size, args.grpo_samples)
    if "epic" in sections:
        results["epic"] = epic_target_audit(model, processor, cfg, reports_dir, args.seed)
    if "opsd" in sections:
        results["opsd"] = opsd_audit(model, processor, cfg, reports_dir, args.seed)

    failed = [name for name, ok in results.items() if not ok]
    summary = ["# Audit Summary", "", *[f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in results.items()]]
    write_report(reports_dir / "audit_summary.md", summary)
    if failed:
        print(f"Audit failed: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
