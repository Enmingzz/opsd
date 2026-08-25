#!/usr/bin/env python3
"""Inspect full/r010/r011 next-token distributions at selected negative-F tokens."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
OPSD_ROOT = HERE.parents[1]
PROJECT_ROOT = OPSD_ROOT.parent
DEFAULT_SAMPLES = (
    OPSD_ROOT
    / "data/openmmreasoner_llava_cot_holdout1k_decontam_v1_seed42"
    / "holdout1k_metric_samples.jsonl"
)
DEFAULT_STEP_ROOT = HERE / "outputs/step_000000"
DEFAULT_OUTPUT = HERE / "outputs/analysis/negative_f_topk_step0_d01.json"
TARGETS = {
    "openmm_llava_cot_000260892": 13,
    "openmm_llava_cot_000064316": 18,
    "openmm_llava_cot_000274912": 33,
    "openmm_llava_cot_000403373": 27,
    "openmm_llava_cot_000486550": 13,
}

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from opsd.analysis.r010_only_lcot1k_fixed_teacher_deltas_20260823.metrics import (  # noqa: E402
    pairwise_divergence,
)
from opsd.experiments.llm_only.teacher_gap_persistence_opsd_pilot_20260801 import (  # noqa: E402
    fixed_prefix_probe as probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--step-root", type=Path, default=DEFAULT_STEP_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-id", choices=sorted(TARGETS))
    parser.add_argument(
        "--pruning-method",
        choices=("visionzip", "divprune", "random", "fastv"),
        default="visionzip",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_payload(root: Path, kind: str, sample_id: str) -> dict[str, Any]:
    matches = []
    for path in root.glob(f"**/{kind}/samples/*.json"):
        payload = json.loads(path.read_text())
        if payload.get("sample_id") == sample_id:
            matches.append((path, payload))
    if not matches:
        raise RuntimeError(f"No {kind} payload for {sample_id}")
    # Prefer the canonical non-sharded artifact if duplicates exist.
    matches.sort(key=lambda item: ("/shards/" in str(item[0]), len(str(item[0]))))
    return matches[0][1]


def main() -> None:
    args = parse_args()
    os.environ["OPSD_PRUNING_METHOD"] = args.pruning_method
    targets = (
        {args.sample_id: TARGETS[args.sample_id]} if args.sample_id else TARGETS
    )
    sources = {str(row["sample_id"]): row for row in read_jsonl(args.samples.resolve())}
    step_root = args.step_root.resolve()
    scores = {sid: find_payload(step_root, "scores", sid) for sid in targets}
    rollouts = {sid: find_payload(step_root, "rollouts", sid) for sid in targets}
    adapter_paths = {str(row["adapter_path"]) for row in scores.values()}
    if len(adapter_paths) != 1:
        raise RuntimeError(f"Expected one adapter, found {sorted(adapter_paths)}")
    adapter_path = adapter_paths.pop()

    (
        _,
        _,
        sample_cls,
        _,
        decode_token_ids,
        encode_prompt,
        extract_generated_logits,
        forward_pruned,
        _,
        _,
        model_input_subset,
        primary_device,
        teacher_adapter_disabled,
        _,
        sequence_inputs_from_prompt,
        _,
        temporary_eval,
    ) = probe.load_stack()
    model, processor = probe.load_model(
        adapter_path,
        merge_adapter=False,
        device_map_mode="single_gpu",
        min_pixels=1280 * 28 * 28,
        max_pixels=4096 * 28 * 28,
    )
    model.requires_grad_(False)
    device = primary_device(model)

    def decode(token_id: int) -> str:
        return decode_token_ids(
            processor, torch.tensor([token_id], dtype=torch.long, device=device)
        )

    results = []
    with torch.inference_mode(), temporary_eval(model):
        for sample_id, position in targets.items():
            source = sources[sample_id]
            score = scores[sample_id]
            rollout = rollouts[sample_id]
            token_ids = [int(value) for value in rollout["generated_token_ids"]]
            if token_ids != [int(value) for value in score["generated_token_ids"]]:
                raise RuntimeError(f"Rollout/score token mismatch for {sample_id}")
            sample = probe.make_sample(source, sample_cls)
            prompt_inputs = encode_prompt(processor, sample, image_root="", device=device)
            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            generated_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            sequence_inputs = sequence_inputs_from_prompt(prompt_inputs, generated_ids)
            count = len(token_ids)

            def score_pruned(ratio: float) -> tuple[torch.Tensor, dict[str, Any]]:
                output, metadata = forward_pruned(
                    model,
                    sequence_inputs,
                    ratio,
                    prompt_len=prompt_len,
                    allow_embedding_fallback=False,
                    sample_id=sample_id,
                    question=sample.question,
                )
                native = dict(metadata["metadata"])
                logits = extract_generated_logits(
                    output.logits, int(native["student_prompt_len"]), count
                ).detach()
                del output
                return logits, native

            logits_b, metadata_b = score_pruned(0.10)
            logits_plus, metadata_plus = score_pruned(0.11)
            teacher_context = teacher_adapter_disabled(model)
            with teacher_context:
                output = model(**model_input_subset(sequence_inputs), use_cache=False)
            logits_teacher = extract_generated_logits(output.logits, prompt_len, count).detach()
            del output

            token_slice = slice(position, position + 1)
            pair_a = pairwise_divergence(
                logits_teacher[token_slice], logits_b[token_slice]
            )
            pair_b = pairwise_divergence(logits_b[token_slice], logits_plus[token_slice])
            pair_c = pairwise_divergence(
                logits_teacher[token_slice], logits_plus[token_slice]
            )
            a = float(pair_a.jsd.item())
            b = float(pair_b.jsd.item())
            c = float(pair_c.jsd.item())
            p = 0.5 * (a + b - c)

            observed_id = token_ids[position]

            def distribution(label: str, logits: torch.Tensor) -> dict[str, Any]:
                probabilities = torch.softmax(logits[position].float(), dim=-1)
                values, indices = probabilities.topk(args.top_k)
                return {
                    "label": label,
                    "top_k": [
                        {
                            "rank": rank,
                            "token_id": int(token_id),
                            "token_text": decode(int(token_id)),
                            "probability": float(probability),
                        }
                        for rank, (probability, token_id) in enumerate(
                            zip(values.cpu().tolist(), indices.cpu().tolist()), start=1
                        )
                    ],
                    "observed_token_probability": float(probabilities[observed_id]),
                    "observed_token_rank": int(
                        (probabilities > probabilities[observed_id]).sum().item()
                    )
                    + 1,
                }

            stored = score["metrics"]["d01"]
            recomputed = {
                "teacher_to_r010_forward_kl": float(pair_a.forward_kl.item()),
                "A_jsd": a,
                "B_jsd": b,
                "C_jsd": c,
                "P": p,
                "F": p / max(a, 1e-12),
            }
            expected = {
                "teacher_to_r010_forward_kl": float(stored["A"]["forward_kl"][position]),
                "A_jsd": float(stored["A"]["jsd"][position]),
                "B_jsd": float(stored["B"]["jsd"][position]),
                "C_jsd": float(stored["C"]["jsd"][position]),
            }
            results.append(
                {
                    "sample_id": sample_id,
                    "question": source["question"],
                    "ground_truth": source.get("answer"),
                    "token_index": position,
                    "observed_token_id": observed_id,
                    "observed_token_text": decode(observed_id),
                    "prefix_before_token": decode_token_ids(
                        processor,
                        torch.tensor([token_ids[:position]], dtype=torch.long, device=device),
                    ),
                    "recomputed": recomputed,
                    "max_abs_metric_reproduction_error": (
                        max(abs(recomputed[key] - value) for key, value in expected.items())
                        if args.pruning_method == "visionzip"
                        else None
                    ),
                    "metric_reproduction_reference": (
                        "stored VisionZip d01 metrics"
                        if args.pruning_method == "visionzip"
                        else "not applicable: stored metrics use VisionZip"
                    ),
                    "teacher_full": distribution("teacher_full", logits_teacher),
                    "student_r010": distribution("student_r010", logits_b),
                    "student_r011": distribution("student_r011", logits_plus),
                    "visual_tokens": {
                        "full": int(metadata_b["num_full_visual_tokens"]),
                        "r010": int(metadata_b["num_kept_visual_tokens"]),
                        "r011": int(metadata_plus["num_kept_visual_tokens"]),
                    },
                }
            )
            del logits_b, logits_plus, logits_teacher, prompt_inputs, sequence_inputs
            torch.cuda.empty_cache()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "adapter_path": adapter_path,
                "pruning_method": args.pruning_method,
                "teacher": "fixed step-0 base with full visual tokens",
                "student_prefix": "saved step-0 r010 greedy rollout",
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
