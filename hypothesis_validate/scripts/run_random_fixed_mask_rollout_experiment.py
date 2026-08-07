#!/usr/bin/env python
"""Generate raw VLM rollouts with one deterministic random mask per sample.

The mask RNG is independent of generation RNG. A sample-specific random mask
is constructed once, persisted, and reused for greedy generation and every
sampled rollout. This script intentionally performs no answer extraction,
correctness scoring, judge call, or benchmark post-processing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_CONFIG = ROOT / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw pass@N rollouts with a fixed random visual-token mask per sample."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="Shared mask directory for greedy and sampled runs (default: OUTPUT_DIR/../fixed_masks).",
    )
    parser.add_argument(
        "--benchmark",
        choices=("MMStar", "MMStar_OpenEnded", "MathVista_MINI", "OCRBench"),
        required=True,
    )
    parser.add_argument("--decode-mode", choices=("greedy", "sample"), required=True)
    parser.add_argument("--retention-ratio", type=float, required=True)
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--base-model")
    parser.add_argument("--seed", type=int, default=42, help="Generation RNG base seed.")
    parser.add_argument("--mask-seed", type=int, default=42, help="Random-mask RNG base seed.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_sample_mask_seed(base_seed: int, sample_id: str) -> int:
    """Match the repository RandomPruner sample-ID seed derivation."""

    digest = hashlib.sha1(str(sample_id).encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**31 - 1)


def mask_file_name(sample_rank: int, sample_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._") or "sample"
    return f"sample_{int(sample_rank):03d}_{safe_id[:80]}.json"


def build_mask_document(
    *,
    source: dict[str, Any],
    mask_seed_base: int,
    sample_mask_seed: int,
    retention_ratio: float,
    num_full_visual_tokens: int,
    kept_indices: list[int],
    prefix_hash: str,
    image_grid_thw: list[list[int]],
) -> dict[str, Any]:
    body = {
        "schema_version": "random_fixed_visual_mask_v1",
        "pruner": "random",
        "selection_policy": "sample_specific_seed_then_sorted_random_subset",
        "sample_id": str(source["sample_id"]),
        "sample_rank": int(source["sample_rank"]),
        "source_row_sha256": canonical_hash(source),
        "image_sha256": str(source["image_sha256"]),
        "mask_seed_base": int(mask_seed_base),
        "sample_mask_seed": int(sample_mask_seed),
        "target_retention_ratio": float(retention_ratio),
        "num_full_visual_tokens": int(num_full_visual_tokens),
        "num_kept_visual_tokens": len(kept_indices),
        "realized_retention_ratio": len(kept_indices) / max(1, int(num_full_visual_tokens)),
        "kept_visual_indices": [int(value) for value in kept_indices],
        "kept_visual_indices_sha256": canonical_hash([int(value) for value in kept_indices]),
        "prefix_hash": str(prefix_hash),
        "image_grid_thw": image_grid_thw,
    }
    return {**body, "fixed_mask_hash": canonical_hash(body)}


def write_or_verify_mask(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise RuntimeError(f"Persisted fixed mask does not match recomputed mask: {path}")
        return
    atomic_json(path, document)


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status_sha256": hashlib.sha256((status or "").encode("utf-8")).hexdigest(),
    }


def compare_cross_mode_masks(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counterpart_name = "sample64" if output_dir.name == "greedy" else "greedy"
    counterpart_path = output_dir.parent / counterpart_name / "raw_outputs.jsonl"
    counterpart_rows = read_jsonl(counterpart_path)
    if not counterpart_rows:
        return {
            "counterpart_available": False,
            "counterpart_path": str(counterpart_path),
            "matched_samples": 0,
            "mismatched_samples": [],
        }
    current = {str(row["sample_id"]): str(row["fixed_mask_hash"]) for row in rows}
    counterpart = {str(row["sample_id"]): str(row["fixed_mask_hash"]) for row in counterpart_rows}
    common = sorted(set(current) & set(counterpart))
    mismatched = [sample_id for sample_id in common if current[sample_id] != counterpart[sample_id]]
    return {
        "counterpart_available": True,
        "counterpart_path": str(counterpart_path),
        "matched_samples": len(common) - len(mismatched),
        "mismatched_samples": mismatched,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.retention_ratio <= 1.0:
        raise ValueError("Random pruning requires retention_ratio in (0, 1].")
    if args.decode_mode == "sample" and args.num_rollouts <= 0:
        raise ValueError("Sample mode requires --num-rollouts > 0.")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")

    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    samples_path = args.samples.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mask_dir = (args.mask_dir or (output_dir.parent / "fixed_masks")).expanduser().resolve()
    output_path = output_dir / "raw_outputs.jsonl"
    error_path = output_dir / "raw_errors.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    validation_path = output_dir / "raw_validation.json"
    if args.overwrite:
        for path in (output_path, error_path, manifest_path, validation_path):
            path.unlink(missing_ok=True)

    samples = read_jsonl(samples_path)[args.start_index :]
    if args.limit > 0:
        samples = samples[: args.limit]
    if len(samples) != args.limit:
        raise RuntimeError(f"Expected exactly {args.limit} samples, found {len(samples)} in {samples_path}.")
    if len({str(row["sample_id"]) for row in samples}) != len(samples):
        raise RuntimeError("Sample IDs are not unique.")
    if any(str(row.get("benchmark")) != args.benchmark for row in samples):
        found = sorted({str(row.get("benchmark")) for row in samples})
        raise RuntimeError(f"Benchmark mismatch: requested {args.benchmark}, found {found}.")
    for row in samples:
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not str(row.get("prompt", "")).strip():
            raise RuntimeError(f"Sample {row['sample_id']} has no prompt.")

    clean_eval_root = Path(config["clean_eval_root"]).expanduser().resolve()
    clean_transformers_src = Path(config["clean_eval_transformers_src"]).expanduser().resolve()
    os.environ["VLM_ROOT"] = str(clean_eval_root)
    os.environ["ARMEN_TRANSFORMERS_SRC"] = str(clean_transformers_src)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(clean_transformers_src))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import transformers
    from opsd.pruning_distill.pruners import RandomPruner
    from opsd.pruning_distill.qwen25_pruned_forward import (
        build_pruned_inputs_embeds,
        compute_full_position_ids,
        get_qwen25_visual_embeds,
    )
    from opsd.visionzip_aokvqa.qwen_wrapper import import_qwen25_modules, primary_device
    from run_mathvista_pass_at_k import generation_seed, pruned_prefix_hash, sample_next_token, set_seed
    from run_position_divergence import encode_clean_armen_prompt

    def prepare_fixed_prefix(
        model: Any,
        prompt_inputs: dict[str, Any],
        source: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        vision_embeds = get_qwen25_visual_embeds(model, prompt_inputs)
        sample_mask_seed = derive_sample_mask_seed(args.mask_seed, str(source["sample_id"]))
        # The repository RandomPruner performs the actual random selection. We
        # pass an already sample-specific seed so selection is independent of
        # generation RNG and identical across all rollout batches.
        pruner = RandomPruner(seed=sample_mask_seed)
        keep_indices = pruner.select(
            vision_embeds,
            prompt_inputs.get("image_grid_thw"),
            args.retention_ratio,
            question=str(source["question"]),
            metadata=None,
        )
        full_position_ids = compute_full_position_ids(
            model,
            prompt_inputs["input_ids"],
            prompt_inputs.get("image_grid_thw"),
            prompt_inputs.get("video_grid_thw"),
            prompt_inputs.get("attention_mask"),
            prompt_inputs.get("second_per_grid_ts"),
            prompt_inputs.get("mm_token_type_ids"),
        )
        pruned = build_pruned_inputs_embeds(
            model,
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
            full_position_ids,
            vision_embeds,
            keep_indices,
            mode="drop_tokens",
            prompt_len=prompt_len,
            full_mm_token_type_ids=prompt_inputs.get("mm_token_type_ids"),
        )
        prefix_hash = pruned_prefix_hash(pruned)
        kept = [int(value) for value in keep_indices.detach().cpu().tolist()]
        grid = prompt_inputs.get("image_grid_thw")
        grid_list = [] if grid is None else [[int(value) for value in row] for row in grid.detach().cpu().tolist()]
        document = build_mask_document(
            source=source,
            mask_seed_base=args.mask_seed,
            sample_mask_seed=sample_mask_seed,
            retention_ratio=args.retention_ratio,
            num_full_visual_tokens=int(vision_embeds.shape[0]),
            kept_indices=kept,
            prefix_hash=prefix_hash,
            image_grid_thw=grid_list,
        )
        mask_path = mask_dir / mask_file_name(int(source["sample_rank"]), str(source["sample_id"]))
        write_or_verify_mask(mask_path, document)
        del vision_embeds, full_position_ids, keep_indices
        return pruned, document, mask_path

    def generate_fixed_prefix_batch(
        model: Any,
        pruned: dict[str, Any],
        *,
        batch_size: int,
        do_sample: bool,
    ) -> tuple[list[list[int]], float]:
        if batch_size <= 0:
            return [], 0.0
        started = time.time()
        with torch.inference_mode():
            prefill = model(
                inputs_embeds=pruned["inputs_embeds"],
                attention_mask=pruned["attention_mask"],
                position_ids=pruned["position_ids"],
                use_cache=True,
                return_dict=True,
            )
        cache = prefill.past_key_values
        if cache is None or not hasattr(cache, "batch_repeat_interleave"):
            raise RuntimeError(f"Expected an expandable Transformers cache, got {type(cache).__name__}.")
        prefix_len = int(pruned["attention_mask"].shape[1])
        if int(cache.get_seq_length()) != prefix_len:
            raise RuntimeError(f"Prefix/cache length mismatch: {prefix_len} != {cache.get_seq_length()}.")

        next_logits = prefill.logits[:, -1, :].expand(batch_size, -1)
        cache.batch_repeat_interleave(batch_size)
        attention_mask = pruned["attention_mask"].repeat(batch_size, 1)
        next_position = pruned["position_ids"][:, :, -1:].repeat(1, batch_size, 1) + 1
        generated = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=next_logits.device)

        for position in range(int(args.max_new_tokens)):
            next_token = sample_next_token(
                next_logits,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else 0.0,
                top_p=args.top_p if do_sample else 0.001,
                top_k=args.top_k if do_sample else 1,
            )
            if bool(finished.any()):
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, int(pad_token_id)),
                    next_token,
                )
            token_values = next_token[:, 0].detach().cpu().tolist()
            for row_index, token_id in enumerate(token_values):
                if not bool(finished[row_index]):
                    generated[row_index].append(int(token_id))
            newly_finished = torch.tensor(
                [int(token_id) in eos_ids for token_id in token_values],
                dtype=torch.bool,
                device=finished.device,
            )
            finished |= newly_finished
            if bool(finished.all()) or position + 1 >= int(args.max_new_tokens):
                break

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=1,
            )
            cache_position = torch.tensor([cache.get_seq_length()], dtype=torch.long, device=next_token.device)
            with torch.inference_mode():
                decode = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=next_position,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                )
            cache = decode.past_key_values
            next_logits = decode.logits[:, -1, :]
            next_position = next_position + 1
            del decode

        del prefill, cache, next_logits
        return generated, time.time() - started

    base_model = str(args.base_model or config["base_model"])
    min_pixels = int(args.min_pixels or config["min_pixels"])
    max_pixels = int(args.max_pixels or config["max_pixels"])
    model_cls, _, processor_cls = import_qwen25_modules()
    processor = processor_cls.from_pretrained(base_model)
    model = model_cls.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    if model.__class__.__name__ != "Qwen2_5_VLForConditionalGeneration":
        raise RuntimeError(f"Unexpected model class: {model.__class__.__name__}")
    device = primary_device(model)
    eos_config = model.generation_config.eos_token_id
    eos_ids = {
        int(value)
        for value in (eos_config if isinstance(eos_config, list) else [eos_config])
        if value is not None
    }
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        if not eos_ids:
            raise RuntimeError("Neither pad_token_id nor eos_token_id is available.")
        pad_token_id = min(eos_ids)

    expected_per_sample = 1 if args.decode_mode == "greedy" else int(args.num_rollouts)
    expected_records = len(samples) * expected_per_sample
    completed = {
        (str(row["sample_id"]), int(row["rollout_index"]))
        for row in read_jsonl(output_path)
    }
    manifest = {
        "schema_version": "random_fixed_mask_raw_rollout_experiment_v1",
        "raw_only": True,
        "post_processing_applied": False,
        "answer_parser_applied": False,
        "judge_applied": False,
        "benchmark": args.benchmark,
        "samples_path": str(samples_path),
        "samples_sha256": sha256_file(samples_path),
        "sample_count": len(samples),
        "sample_ids_sha256": canonical_hash([str(row["sample_id"]) for row in samples]),
        "base_model": base_model,
        "model_class": model.__class__.__name__,
        "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": "flash_attention_2",
        "transformers_version": transformers.__version__,
        "pruning": "random",
        "pruning_implementation": "opsd.pruning_distill.pruners.RandomPruner",
        "student_input_mode": "drop_tokens",
        "fixed_mask_scope": "one_mask_per_sample_shared_by_greedy_and_all_rollouts",
        "mask_seed_base": args.mask_seed,
        "mask_seed_derivation": "base_seed_plus_sha1_sample_id_prefix_mod_2^31_minus_1",
        "mask_dir": str(mask_dir),
        "retention_ratio": args.retention_ratio,
        "decode_mode": args.decode_mode,
        "rollouts_per_sample": expected_per_sample,
        "generation": {
            "temperature": 0.0 if args.decode_mode == "greedy" else args.temperature,
            "top_p": 0.001 if args.decode_mode == "greedy" else args.top_p,
            "top_k": 1 if args.decode_mode == "greedy" else args.top_k,
            "do_sample": args.decode_mode == "sample",
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "rollout_batch_size": 1 if args.decode_mode == "greedy" else args.rollout_batch_size,
        },
        "generation_seed": args.seed,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "clean_eval_root": str(clean_eval_root),
        "clean_eval_transformers_src": str(clean_transformers_src),
        "git": git_metadata(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu": torch.cuda.get_device_name(device),
        "output": str(output_path),
        "created_at_unix": time.time(),
    }
    atomic_json(manifest_path, manifest)

    started = time.time()
    failures = 0
    ordinal = len(completed)

    def write_record(
        *,
        source: dict[str, Any],
        token_ids: list[int],
        rollout_index: int,
        actual_seed: int,
        batch_seed: int,
        sequence_in_batch: int,
        batch_size: int,
        elapsed_seconds: float,
        batch_elapsed_seconds: float,
        mask_document: dict[str, Any],
        mask_path: Path,
    ) -> None:
        nonlocal ordinal
        raw_text = processor.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        terminated_by_eos = bool(token_ids and token_ids[-1] in eos_ids)
        source_metadata = {
            key: value
            for key, value in source.items()
            if key not in {"image_path", "image_sha256", "question", "prompt", "answer", "options"}
        }
        record = {
            "schema_version": "raw_rollout_v1",
            "benchmark": args.benchmark,
            "sample_id": str(source["sample_id"]),
            "sample_rank": int(source["sample_rank"]),
            "image_path": str(source["image_path"]),
            "image_sha256": str(source["image_sha256"]),
            "question": str(source["question"]),
            "prompt": str(source["prompt"]),
            "reference_answer": source.get("answer"),
            "answer_options": source.get("options"),
            "source_metadata": source_metadata,
            "source_row_sha256": canonical_hash(source),
            "model": base_model,
            "pruning": "random",
            "pruning_implementation": "opsd.pruning_distill.pruners.RandomPruner",
            "student_input_mode": "drop_tokens",
            "target_retention_ratio": args.retention_ratio,
            "num_full_visual_tokens": int(mask_document["num_full_visual_tokens"]),
            "num_kept_visual_tokens": int(mask_document["num_kept_visual_tokens"]),
            "realized_retention_ratio": float(mask_document["realized_retention_ratio"]),
            "mask_seed_base": int(mask_document["mask_seed_base"]),
            "sample_mask_seed": int(mask_document["sample_mask_seed"]),
            "fixed_mask_hash": str(mask_document["fixed_mask_hash"]),
            "kept_visual_indices_sha256": str(mask_document["kept_visual_indices_sha256"]),
            "fixed_mask_path": str(mask_path),
            "fixed_mask_reused_for_sample_rollouts": True,
            "prefix_hash": str(mask_document["prefix_hash"]),
            "decode_mode": args.decode_mode,
            "rollout_index": rollout_index,
            "generation_seed": actual_seed,
            "batch_seed": batch_seed,
            "sequence_in_batch": sequence_in_batch,
            "rollout_batch_size": batch_size,
            "generation_engine": "shared_fixed_random_prefix_kv_cache",
            "temperature": 0.0 if args.decode_mode == "greedy" else args.temperature,
            "top_p": 0.001 if args.decode_mode == "greedy" else args.top_p,
            "top_k": 1 if args.decode_mode == "greedy" else args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "generated_token_ids": token_ids,
            "raw_generated_text": raw_text,
            "generated_token_count": len(token_ids),
            "terminated_by_eos": terminated_by_eos,
            "hit_max_new_tokens": bool(len(token_ids) >= args.max_new_tokens and not terminated_by_eos),
            "elapsed_seconds": elapsed_seconds,
            "batch_elapsed_seconds": batch_elapsed_seconds,
        }
        append_jsonl(output_path, record)
        completed.add((str(source["sample_id"]), rollout_index))
        ordinal += 1
        print(
            f"[{ordinal}/{expected_records}] id={source['sample_id']} mode={args.decode_mode} "
            f"rollout={rollout_index} tokens={len(token_ids)} batch={batch_size} "
            f"mask={mask_document['fixed_mask_hash'][:12]}",
            flush=True,
        )

    for source in samples:
        sample_id = str(source["sample_id"])
        prompt_inputs = None
        pruned = None
        sample_started = time.time()
        try:
            expected_indices = [-1] if args.decode_mode == "greedy" else list(range(args.num_rollouts))
            if all((sample_id, index) in completed for index in expected_indices):
                continue
            prompt_inputs = encode_clean_armen_prompt(
                processor,
                source,
                device=device,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pruned, mask_document, mask_path = prepare_fixed_prefix(model, prompt_inputs, source)
            if args.decode_mode == "greedy":
                rollout_index = -1
                if (sample_id, rollout_index) not in completed:
                    actual_seed = generation_seed(args.seed, int(source["sample_rank"]), rollout_index)
                    set_seed(actual_seed)
                    generated_rows, elapsed = generate_fixed_prefix_batch(
                        model,
                        pruned,
                        batch_size=1,
                        do_sample=False,
                    )
                    write_record(
                        source=source,
                        token_ids=generated_rows[0],
                        rollout_index=rollout_index,
                        actual_seed=actual_seed,
                        batch_seed=actual_seed,
                        sequence_in_batch=0,
                        batch_size=1,
                        elapsed_seconds=elapsed,
                        batch_elapsed_seconds=elapsed,
                        mask_document=mask_document,
                        mask_path=mask_path,
                    )
            else:
                batch_size = min(args.rollout_batch_size, args.num_rollouts)
                for batch_start in range(0, args.num_rollouts, batch_size):
                    rollout_indices = list(range(batch_start, min(batch_start + batch_size, args.num_rollouts)))
                    missing = [index for index in rollout_indices if (sample_id, index) not in completed]
                    if not missing:
                        continue
                    batch_seed = generation_seed(args.seed, int(source["sample_rank"]), batch_start)
                    set_seed(batch_seed)
                    generated_rows, batch_elapsed = generate_fixed_prefix_batch(
                        model,
                        pruned,
                        batch_size=len(rollout_indices),
                        do_sample=True,
                    )
                    effective_elapsed = batch_elapsed / max(1, len(rollout_indices))
                    for sequence_in_batch, (rollout_index, token_ids) in enumerate(
                        zip(rollout_indices, generated_rows, strict=True)
                    ):
                        if rollout_index not in missing:
                            continue
                        write_record(
                            source=source,
                            token_ids=token_ids,
                            rollout_index=rollout_index,
                            actual_seed=batch_seed,
                            batch_seed=batch_seed,
                            sequence_in_batch=sequence_in_batch,
                            batch_size=len(rollout_indices),
                            elapsed_seconds=effective_elapsed,
                            batch_elapsed_seconds=batch_elapsed,
                            mask_document=mask_document,
                            mask_path=mask_path,
                        )
            print(f"sample_complete id={sample_id} elapsed={time.time() - sample_started:.1f}s", flush=True)
        except Exception as exc:
            failures += 1
            append_jsonl(
                error_path,
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"ERROR id={sample_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise
        finally:
            if pruned is not None:
                del pruned
            if prompt_inputs is not None:
                del prompt_inputs
            gc.collect()
            torch.cuda.empty_cache()

    rows = read_jsonl(output_path)
    observed_keys = [(str(row["sample_id"]), int(row["rollout_index"])) for row in rows]
    expected_keys = {
        (str(source["sample_id"]), index)
        for source in samples
        for index in ([-1] if args.decode_mode == "greedy" else range(args.num_rollouts))
    }
    observed_set = set(observed_keys)
    forbidden_fields = {
        "parsed_answer",
        "correct",
        "qwen_correct",
        "qwen_extracted_answer",
        "judge_output",
        "post_processed_output",
    }
    forbidden_present = sorted({key for row in rows for key in forbidden_fields if key in row})
    per_sample_hashes: dict[str, set[str]] = {}
    per_sample_prefixes: dict[str, set[str]] = {}
    per_sample_seeds: dict[str, set[int]] = {}
    mask_document_errors: list[str] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        per_sample_hashes.setdefault(sample_id, set()).add(str(row["fixed_mask_hash"]))
        per_sample_prefixes.setdefault(sample_id, set()).add(str(row["prefix_hash"]))
        per_sample_seeds.setdefault(sample_id, set()).add(int(row["sample_mask_seed"]))
        path = Path(str(row["fixed_mask_path"]))
        if not path.is_file():
            mask_document_errors.append(f"{sample_id}:missing_mask_file")
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if str(document.get("fixed_mask_hash")) != str(row["fixed_mask_hash"]):
            mask_document_errors.append(f"{sample_id}:mask_hash_mismatch")
        if str(document.get("prefix_hash")) != str(row["prefix_hash"]):
            mask_document_errors.append(f"{sample_id}:prefix_hash_mismatch")
        if int(document.get("sample_mask_seed", -1)) != int(row["sample_mask_seed"]):
            mask_document_errors.append(f"{sample_id}:mask_seed_mismatch")

    hash_invariant = all(len(values) == 1 for values in per_sample_hashes.values())
    prefix_invariant = all(len(values) == 1 for values in per_sample_prefixes.values())
    seed_invariant = all(len(values) == 1 for values in per_sample_seeds.values())
    unique_sample_seeds = {
        next(iter(values)) for values in per_sample_seeds.values() if len(values) == 1
    }
    target_count_matches = all(
        int(row["num_kept_visual_tokens"])
        == min(
            int(row["num_full_visual_tokens"]),
            max(1, int(round(int(row["num_full_visual_tokens"]) * args.retention_ratio))),
        )
        for row in rows
    )
    cross_mode = compare_cross_mode_masks(output_dir, rows)
    cross_mode_ok = not cross_mode["counterpart_available"] or not cross_mode["mismatched_samples"]
    validation_passed = (
        not failures
        and len(rows) == expected_records
        and observed_set == expected_keys
        and len(observed_keys) == len(observed_set)
        and not forbidden_present
        and len(per_sample_hashes) == len(samples)
        and hash_invariant
        and prefix_invariant
        and seed_invariant
        and len(unique_sample_seeds) == len(samples)
        and target_count_matches
        and not mask_document_errors
        and cross_mode_ok
    )
    realized = [float(row["realized_retention_ratio"]) for row in rows]
    validation = {
        "status": "passed" if validation_passed else "failed",
        "raw_only": True,
        "post_processing_applied": False,
        "benchmark": args.benchmark,
        "decode_mode": args.decode_mode,
        "pruning": "random",
        "fixed_mask_scope": "one_mask_per_sample_shared_by_greedy_and_all_rollouts",
        "target_retention_ratio": args.retention_ratio,
        "sample_count": len(samples),
        "expected_records": expected_records,
        "observed_records": len(rows),
        "unique_records": len(observed_set),
        "missing_records": len(expected_keys - observed_set),
        "unexpected_records": len(observed_set - expected_keys),
        "duplicate_records": len(observed_keys) - len(observed_set),
        "failures": failures,
        "forbidden_postprocess_fields": forbidden_present,
        "raw_text_field": "raw_generated_text",
        "raw_token_field": "generated_token_ids",
        "fixed_mask_invariant_all_samples": hash_invariant,
        "fixed_prefix_invariant_all_samples": prefix_invariant,
        "sample_mask_seed_invariant_all_samples": seed_invariant,
        "unique_sample_mask_seed_count": len(unique_sample_seeds),
        "target_token_count_matches_all_records": target_count_matches,
        "mask_document_errors": sorted(set(mask_document_errors)),
        "cross_mode_mask_check": cross_mode,
        "realized_retention_ratio": {
            "mean": sum(realized) / len(realized) if realized else None,
            "min": min(realized) if realized else None,
            "max": max(realized) if realized else None,
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(validation_path, validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
