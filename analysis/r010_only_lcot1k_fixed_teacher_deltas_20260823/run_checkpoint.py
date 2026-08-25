#!/usr/bin/env python3
"""Generate r010 rollouts and score three native VisionZip interventions."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
OPSD_ROOT = HERE.parents[1]
PROJECT_ROOT = OPSD_ROOT.parent
DEFAULT_SAMPLES = (
    OPSD_ROOT
    / "data/openmmreasoner_llava_cot_holdout1k_decontam_v1_seed42/holdout1k_metric_samples.jsonl"
)
DEFAULT_CONFIG = (
    OPSD_ROOT
    / "experiments/llm_only/opsd_random_r010_only_dropout0_20260815/configs/train_10240.yaml"
)
LOW_RATIO = 0.10
DELTAS = (0.01, 0.02, 0.05)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from opsd.analysis.r010_only_lcot1k_fixed_teacher_deltas_20260823.metrics import (  # noqa: E402
    PairwiseDivergence,
    pairwise_divergence,
)
from opsd.experiments.llm_only.teacher_gap_persistence_opsd_pilot_20260801 import (  # noqa: E402
    fixed_prefix_probe as probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=4096 * 28 * 28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher-mode",
        choices=("fixed_base", "current_checkpoint"),
        default="fixed_base",
        help="Use either the adapter-disabled base teacher or the active checkpoint at full tokens.",
    )
    parser.add_argument(
        "--pruning-method",
        choices=("visionzip", "divprune", "fastv", "random"),
        default="visionzip",
        help="Native visual-token pruner used by the student branches.",
    )
    parser.add_argument(
        "--random-pruner-seed",
        type=int,
        default=42,
        help="Base seed for deterministic per-sample RandomPruner rankings.",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=list(DELTAS),
        help="Visual-budget interventions to score; defaults to 0.01 0.02 0.05.",
    )
    parser.add_argument(
        "--rollout-source-dir",
        type=Path,
        help="Optional directory containing hashed rollout sample JSON files to replay.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def json_default(value: Any) -> Any:
    """Serialize model metadata without changing its numerical contents."""

    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=json_default,
                )
                + "\n"
            )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_file(root: Path, sample_id: str) -> Path:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return root / f"{digest}.json"


def metric_payload(value: PairwiseDivergence) -> dict[str, list[float]]:
    return {
        "jsd": value.jsd.cpu().tolist(),
        "forward_kl": value.forward_kl.cpu().tolist(),
        "reverse_kl": value.reverse_kl.cpu().tolist(),
    }


def delta_tag(delta: float) -> str:
    return f"d{int(round(100 * delta)):02d}"


def consolidate(
    root: Path,
    expected_samples: list[dict[str, Any]],
    step: int,
    deltas: tuple[float, ...],
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rollout_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for source in expected_samples:
        sample_id = str(source["sample_id"])
        rollout_path = sample_file(root / "rollouts/samples", sample_id)
        score_path = sample_file(root / "scores/samples", sample_id)
        if rollout_path.is_file():
            rollout_rows.append(json.loads(rollout_path.read_text(encoding="utf-8")))
        if score_path.is_file():
            score_rows.append(json.loads(score_path.read_text(encoding="utf-8")))
    rollout_rows.sort(key=lambda row: int(row["sample_index"]))
    score_rows.sort(key=lambda row: int(row["sample_index"]))
    atomic_jsonl(root / "rollouts.jsonl", rollout_rows)
    atomic_jsonl(root / "scores.jsonl", score_rows)

    parquet_path = root / "per_token_metrics.parquet"
    temporary = parquet_path.with_name(f".{parquet_path.name}.tmp.{os.getpid()}")
    writer: pq.ParquetWriter | None = None
    token_count = 0
    metric_columns = [
        f"{delta_tag(delta)}_{pair}_{metric}"
        for delta in deltas
        for pair in ("A", "B", "C")
        for metric in ("jsd", "forward_kl", "reverse_kl")
    ]
    try:
        for score in score_rows:
            count = int(score["generated_tokens"])
            columns: dict[str, Any] = {
                "checkpoint_step": [int(step)] * count,
                "sample_index": [int(score["sample_index"])] * count,
                "sample_id": [str(score["sample_id"])] * count,
                "token_index": list(range(count)),
                "token_id": [int(value) for value in score["generated_token_ids"]],
                "token_text": [str(value) for value in score["generated_token_text"]],
                "is_eos": [bool(value) for value in score["is_eos"]],
                "student_ratio": [LOW_RATIO] * count,
            }
            for delta in deltas:
                tag = delta_tag(delta)
                block = score["metrics"][tag]
                for pair in ("A", "B", "C"):
                    for metric in ("jsd", "forward_kl", "reverse_kl"):
                        values = block[pair][metric]
                        if len(values) != count:
                            raise ValueError(
                                f"Metric length mismatch {score['sample_id']} {tag}/{pair}/{metric}: "
                                f"{len(values)} != {count}"
                            )
                        columns[f"{tag}_{pair}_{metric}"] = values
            table = pa.Table.from_pydict(columns)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=["sample_id", "token_text"],
                )
            writer.write_table(table)
            token_count += count
    finally:
        if writer is not None:
            writer.close()
    if writer is not None:
        temporary.replace(parquet_path)
    elif temporary.exists():
        temporary.unlink()
    return {
        "completed_rollouts": len(rollout_rows),
        "completed_scores": len(score_rows),
        "token_rows": token_count,
        "metric_columns": metric_columns,
        "per_token_metric_count": len(metric_columns),
    }


def main() -> None:
    args = parse_args()
    if args.step < 0 or args.step % 1024 != 0 or args.step > 10240:
        raise ValueError("step must be one of 0,1024,...,10240")
    if args.limit <= 0 or args.offset < 0:
        raise ValueError("limit must be positive and offset nonnegative")
    deltas = tuple(float(value) for value in args.deltas)
    if not deltas or len(set(deltas)) != len(deltas):
        raise ValueError("deltas must be nonempty and unique")
    if any(delta <= 0.0 or LOW_RATIO + delta > 1.0 for delta in deltas):
        raise ValueError(f"Invalid deltas for student ratio {LOW_RATIO}: {deltas}")
    adapter_path = args.adapter_path.expanduser().resolve()
    adapter_file = adapter_path / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    samples_path = args.samples.expanduser().resolve()
    all_samples = read_jsonl(samples_path)
    rows = all_samples[args.offset : args.offset + args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"Requested {args.limit} samples at offset {args.offset}, found {len(rows)}")
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rollout_source_dir = (
        args.rollout_source_dir.expanduser().resolve() if args.rollout_source_dir else None
    )
    if rollout_source_dir is not None and not rollout_source_dir.is_dir():
        raise FileNotFoundError(rollout_source_dir)
    if args.overwrite:
        for directory in (root / "rollouts/samples", root / "scores/samples"):
            if directory.exists():
                for path in directory.glob("*.json"):
                    path.unlink()

    os.environ["OPSD_PRUNING_METHOD"] = str(args.pruning_method)
    os.environ["OPSD_RANDOM_PRUNER_SEED"] = str(args.random_pruner_seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    (
        _,
        _,
        sample_cls,
        _,
        decode_token_ids,
        encode_prompt,
        extract_generated_logits,
        forward_pruned,
        generate_pruned,
        _,
        model_input_subset,
        primary_device,
        teacher_adapter_disabled,
        _,
        sequence_inputs_from_prompt,
        temporary_cached_rollout,
        temporary_eval,
    ) = probe.load_stack()
    model, processor = probe.load_model(
        str(adapter_path),
        merge_adapter=False,
        device_map_mode="single_gpu",
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
    )
    if not hasattr(model, "peft_config"):
        raise RuntimeError("Fixed-base teacher requires an unmerged PEFT student")
    device = primary_device(model)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    completed_scores = 0
    for local_index, row in enumerate(rows):
        sample_index = int(args.offset + local_index)
        sample_id = str(row["sample_id"])
        rollout_path = sample_file(root / "rollouts/samples", sample_id)
        score_path = sample_file(root / "scores/samples", sample_id)
        if score_path.is_file() and not args.overwrite:
            completed_scores += 1
            continue
        sample = probe.make_sample(row, sample_cls)
        prompt_inputs = encode_prompt(processor, sample, image_root="", device=device)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        source_rollout_path = (
            sample_file(rollout_source_dir, sample_id) if rollout_source_dir is not None else None
        )
        reusable_rollout_path = (
            rollout_path
            if rollout_path.is_file()
            else source_rollout_path
            if source_rollout_path is not None and source_rollout_path.is_file()
            else None
        )
        if reusable_rollout_path is not None and not args.overwrite:
            rollout = json.loads(reusable_rollout_path.read_text(encoding="utf-8"))
            if str(rollout["sample_id"]) != sample_id:
                raise RuntimeError(f"Rollout sample mismatch: {reusable_rollout_path}")
            if int(rollout["checkpoint_step"]) != int(args.step):
                raise RuntimeError(
                    f"Rollout checkpoint mismatch for {sample_id}: "
                    f"{rollout['checkpoint_step']} != {args.step}"
                )
            if reusable_rollout_path != rollout_path:
                atomic_json(rollout_path, rollout)
            generated_ids = torch.tensor(
                [rollout["generated_token_ids"]], dtype=torch.long, device=device
            )
        else:
            with torch.inference_mode(), temporary_eval(model), temporary_cached_rollout(model):
                generated_ids, generated_text, generation_metadata = generate_pruned(
                    model,
                    processor,
                    prompt_inputs,
                    LOW_RATIO,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    allow_embedding_fallback=False,
                    manual_decode=False,
                    max_unparseable_tokens=None,
                    stop_on_parse=False,
                    sample_id=sample.sample_id,
                    question=sample.question,
                )
            if generated_ids.numel() == 0:
                raise RuntimeError(f"Zero-token rollout: {sample_id}")
            rollout = {
                "schema_version": "r010_lcot1k_rollout_v1",
                "checkpoint_step": int(args.step),
                "sample_index": sample_index,
                "sample_id": sample_id,
                "question": sample.question,
                "ground_truth": sample.target,
                "image_path": sample.image,
                "prompt": sample.prompt,
                "retention_ratio": LOW_RATIO,
                "generated_token_ids": [
                    int(value) for value in generated_ids.reshape(-1).cpu().tolist()
                ],
                "generated_text": generated_text,
                "generated_tokens": int(generated_ids.numel()),
                "generation_metadata": generation_metadata,
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "use_cache": True,
                    "max_new_tokens": int(args.max_new_tokens),
                },
            }
            atomic_json(rollout_path, rollout)

        count = int(generated_ids.numel())
        sequence_inputs = sequence_inputs_from_prompt(prompt_inputs, generated_ids)

        def score_pruned(ratio: float) -> tuple[torch.Tensor, dict[str, Any]]:
            outputs, metadata = forward_pruned(
                model,
                sequence_inputs,
                ratio,
                prompt_len=prompt_len,
                allow_embedding_fallback=False,
                sample_id=sample.sample_id,
                question=sample.question,
            )
            native = dict(metadata["metadata"])
            random_keep_indices = metadata.get("random_keep_indices")
            if random_keep_indices is not None:
                native["random_keep_indices"] = [
                    int(value) for value in random_keep_indices.reshape(-1).tolist()
                ]
            logits = extract_generated_logits(
                outputs.logits, int(native["student_prompt_len"]), count
            ).detach().clone()
            del outputs, metadata
            if logits.shape[0] != count:
                raise RuntimeError(f"Prefix length mismatch for {sample_id} at ratio {ratio}")
            return logits, native

        with torch.inference_mode(), temporary_eval(model):
            logits_b, metadata_b = score_pruned(LOW_RATIO)
            teacher_context = (
                teacher_adapter_disabled(model)
                if args.teacher_mode == "fixed_base"
                else contextlib.nullcontext()
            )
            with teacher_context:
                teacher_outputs = model(**model_input_subset(sequence_inputs), use_cache=False)
            logits_teacher = extract_generated_logits(
                teacher_outputs.logits, prompt_len, count
            ).detach().clone()
            del teacher_outputs
            if logits_teacher.shape != logits_b.shape:
                raise RuntimeError(
                    f"Teacher/student logits do not align for {sample_id}: "
                    f"{tuple(logits_teacher.shape)} vs {tuple(logits_b.shape)}"
                )
            pair_a = pairwise_divergence(logits_teacher, logits_b)
            metrics: dict[str, Any] = {}
            intervention_metadata: dict[str, Any] = {}
            for delta in deltas:
                tag = delta_tag(delta)
                ratio_plus = LOW_RATIO + delta
                logits_plus, metadata_plus = score_pruned(ratio_plus)
                if args.pruning_method == "random":
                    retained = set(metadata_b.get("random_keep_indices", []))
                    retained_plus = set(metadata_plus.get("random_keep_indices", []))
                    if not retained or not retained.issubset(retained_plus):
                        raise RuntimeError(
                            f"RandomPruner masks are not nested for {sample_id}: "
                            f"r010={len(retained)} r{ratio_plus:.2f}={len(retained_plus)}"
                        )
                pair_b = pairwise_divergence(logits_b, logits_plus)
                pair_c = pairwise_divergence(logits_teacher, logits_plus)
                metrics[tag] = {
                    "delta": float(delta),
                    "student_ratio": LOW_RATIO,
                    "student_plus_ratio": float(ratio_plus),
                    "A": metric_payload(pair_a),
                    "B": metric_payload(pair_b),
                    "C": metric_payload(pair_c),
                }
                intervention_metadata[tag] = {
                    "student_plus_ratio": float(ratio_plus),
                    "native_student_plus_visual_tokens": int(
                        metadata_plus["num_kept_visual_tokens"]
                    ),
                    "student_plus_pruning_metadata": metadata_plus,
                }
                del logits_plus, pair_b, pair_c

        token_ids = [int(value) for value in generated_ids.reshape(-1).cpu().tolist()]
        token_text = [
            decode_token_ids(processor, torch.tensor([token_id], device=device))
            for token_id in token_ids
        ]
        tokenizer = getattr(processor, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        score = {
            "schema_version": "r010_lcot1k_fixed_teacher_token_divergence_v1",
            "checkpoint_step": int(args.step),
            "sample_index": sample_index,
            "sample_id": sample_id,
            "adapter_path": str(adapter_path),
            "teacher_source": (
                "fixed_step0_base_adapter_disabled_full_tokens"
                if args.teacher_mode == "fixed_base"
                else "active_current_checkpoint_full_tokens"
            ),
            "student_ratio": LOW_RATIO,
            "generated_tokens": count,
            "generated_token_ids": token_ids,
            "generated_token_text": token_text,
            "is_eos": [token_id == eos_token_id for token_id in token_ids],
            "native_full_visual_tokens": int(metadata_b["num_full_visual_tokens"]),
            "native_student_visual_tokens": int(metadata_b["num_kept_visual_tokens"]),
            "student_pruning_metadata": metadata_b,
            "interventions": intervention_metadata,
            "metric_definitions": {
                "A": ["teacher_full", "student_r010"],
                "B": ["student_r010", "student_r010_plus_delta"],
                "C": ["teacher_full", "student_r010_plus_delta"],
                "forward_kl": "KL(first_distribution || second_distribution)",
                "reverse_kl": "KL(second_distribution || first_distribution)",
                "jsd": "0.5*KL(first||mixture)+0.5*KL(second||mixture)",
                "precision": "full-vocabulary FP32",
            },
            "same_text_prefix_all_branches": True,
            "metrics": metrics,
        }
        atomic_json(score_path, score)
        completed_scores += 1
        if completed_scores % 10 == 0 or completed_scores == len(rows):
            print(
                f"step={args.step} completed={completed_scores}/{len(rows)} "
                f"sample={sample_id} tokens={count}",
                flush=True,
            )
        del (
            generated_ids,
            prompt_inputs,
            sequence_inputs,
            logits_b,
            logits_teacher,
            pair_a,
        )

    consolidated = consolidate(root, rows, int(args.step), deltas)
    complete = consolidated["completed_scores"] == len(rows)
    manifest = {
        "schema_version": "r010_lcot1k_fixed_teacher_delta_sweep_manifest_v1",
        "status": "complete" if complete else "partial",
        "checkpoint_step": int(args.step),
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_file),
        "adapter_merged": False,
        "student_semantics": "active PEFT LoRA in eval mode",
        "teacher_source": (
            "Qwen/Qwen2.5-VL-7B-Instruct base weights with adapter disabled"
            if args.teacher_mode == "fixed_base"
            else "active checkpoint LoRA with full visual tokens"
        ),
        "teacher_mode": args.teacher_mode,
        "pruning_method": str(args.pruning_method),
        "random_pruner_seed": (
            int(args.random_pruner_seed) if args.pruning_method == "random" else None
        ),
        "teacher_fixed_step": 0 if args.teacher_mode == "fixed_base" else None,
        "teacher_uses_full_visual_tokens": True,
        "sample_source": str(samples_path),
        "sample_source_sha256": sha256_file(samples_path),
        "sample_count_expected": len(rows),
        "offset": int(args.offset),
        "student_ratio": LOW_RATIO,
        "interventions": list(deltas),
        "student_plus_ratios": [LOW_RATIO + value for value in deltas],
        "prefix_protocol": "checkpoint-specific r010 greedy rollout; identical token IDs replayed for all branches",
        "prompt_mode": "training_matched_reasoning_tags",
        "generation": {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "use_cache": True,
        },
        "scoring": {
            "use_cache": False,
            "exact_full_vocabulary": True,
            "probability_dtype": "float32",
            "per_intervention_scalar_count": 9,
            "per_token_scalar_count": 27,
        },
        "image_processing": {
            "min_pixels": int(args.min_pixels),
            "max_pixels": int(args.max_pixels),
        },
        "seed": int(args.seed),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OPSD_ROOT, text=True
        ).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        **consolidated,
    }
    atomic_json(root / "manifest.json", manifest)
    if not complete:
        raise RuntimeError(
            f"Incomplete checkpoint {args.step}: {consolidated['completed_scores']}/{len(rows)}"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
