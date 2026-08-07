#!/usr/bin/env python
"""Generate one raw-only greedy or sample-N VLM rollout experiment.

This script intentionally performs no answer extraction, correctness scoring,
judge call, or benchmark post-processing. Exact generated token IDs and a
tokenizer decode that preserves special tokens are retained for later audits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
    parser = argparse.ArgumentParser(description="Generate a raw-only pass@N rollout experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        choices=("MMStar", "MMStar_OpenEnded", "MathVista_MINI"),
        required=True,
    )
    parser.add_argument("--decode-mode", choices=("greedy", "sample"), required=True)
    parser.add_argument("--pruning", choices=("visionzip", "none"), required=True)
    parser.add_argument("--retention-ratio", type=float, required=True)
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--base-model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based input-row offset applied before --limit (default: 0).",
    )
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
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def main() -> int:
    args = parse_args()
    if args.decode_mode == "sample" and args.num_rollouts <= 0:
        raise ValueError("Sample mode requires --num-rollouts > 0.")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive.")
    if args.pruning == "visionzip" and not 0.0 < args.retention_ratio < 1.0:
        raise ValueError("VisionZip requires retention_ratio in (0, 1).")
    if args.pruning == "none" and args.retention_ratio != 1.0:
        raise ValueError("No-pruning requires retention_ratio=1.0.")

    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    samples_path = args.samples.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / "raw_outputs.jsonl"
    error_path = output_dir / "raw_errors.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    validation_path = output_dir / "raw_validation.json"
    if args.overwrite:
        for path in (output_path, error_path, manifest_path, validation_path):
            path.unlink(missing_ok=True)

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
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
    os.environ["OPSD_PRUNING_METHOD"] = "visionzip"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(clean_transformers_src))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import transformers
    from opsd.visionzip_aokvqa.qwen_wrapper import (
        import_qwen25_modules,
        last_visionzip_pruned_inputs,
        official_visionzip_metadata,
        primary_device,
    )
    from run_mathvista_pass_at_k import (
        generate_shared_prefix_batch,
        generation_seed,
        measured_visionzip_metadata,
        pruned_prefix_hash,
        set_seed,
    )
    from run_position_divergence import clean_armen_pruning_kwargs, encode_clean_armen_prompt

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
        "schema_version": "raw_rollout_experiment_v1",
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
        "pruning": args.pruning,
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
        "seed": args.seed,
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

    pruning_kwargs = (
        clean_armen_pruning_kwargs(args.retention_ratio) if args.pruning == "visionzip" else {}
    )
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
        engine: str,
        prefix_hash: str | None,
        elapsed_seconds: float,
        batch_elapsed_seconds: float,
        prompt_inputs: dict[str, Any],
        prompt_len: int,
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
        record: dict[str, Any] = {
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
            "pruning": args.pruning,
            "target_retention_ratio": args.retention_ratio,
            "decode_mode": args.decode_mode,
            "rollout_index": rollout_index,
            "generation_seed": actual_seed,
            "batch_seed": batch_seed,
            "sequence_in_batch": sequence_in_batch,
            "rollout_batch_size": batch_size,
            "generation_engine": engine,
            "prefix_hash": prefix_hash,
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
        if args.pruning == "visionzip":
            contextual = min(0.05, args.retention_ratio)
            dominant = max(0.0, args.retention_ratio - contextual)
            metadata = measured_visionzip_metadata(
                model,
                prompt_inputs,
                prompt_len,
                dominant,
                contextual,
                official_visionzip_metadata,
                last_visionzip_pruned_inputs,
            )
            record.update(
                num_full_visual_tokens=int(metadata["num_full_visual_tokens"]),
                num_kept_visual_tokens=int(metadata["num_kept_visual_tokens"]),
                realized_retention_ratio=float(
                    metadata["num_kept_visual_tokens"] / max(1, metadata["num_full_visual_tokens"])
                ),
                visionzip_dominant_ratio=dominant,
                visionzip_contextual_ratio=contextual,
                visionzip_target_count_match=bool(metadata["visionzip_target_count_match"]),
                visionzip_target_count_delta=int(metadata.get("visionzip_target_count_delta", 0)),
            )
        append_jsonl(output_path, record)
        completed.add((str(source["sample_id"]), rollout_index))
        ordinal += 1
        print(
            f"[{ordinal}/{expected_records}] id={source['sample_id']} mode={args.decode_mode} "
            f"rollout={rollout_index} tokens={len(token_ids)} batch={batch_size} ",
            flush=True,
        )

    for source in samples:
        sample_id = str(source["sample_id"])
        prompt_inputs = None
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
            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            if args.decode_mode == "greedy":
                rollout_index = -1
                if (sample_id, rollout_index) not in completed:
                    actual_seed = generation_seed(args.seed, int(source["sample_rank"]), rollout_index)
                    set_seed(actual_seed)
                    generation_kwargs = {
                        **prompt_inputs,
                        **pruning_kwargs,
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "top_p": 0.001,
                        "top_k": 1,
                        "temperature": 0.0,
                        "repetition_penalty": 1.0,
                        "use_cache": True,
                        "num_return_sequences": 1,
                    }
                    rollout_started = time.time()
                    with torch.inference_mode():
                        generated = model.generate(**generation_kwargs)
                    elapsed = time.time() - rollout_started
                    token_ids = [int(value) for value in generated[0, prompt_len:].detach().cpu().tolist()]
                    pruned = last_visionzip_pruned_inputs(model) if args.pruning == "visionzip" else None
                    write_record(
                        source=source,
                        token_ids=token_ids,
                        rollout_index=rollout_index,
                        actual_seed=actual_seed,
                        batch_seed=actual_seed,
                        sequence_in_batch=0,
                        batch_size=1,
                        engine="clean_armen_hf_generate",
                        prefix_hash=pruned_prefix_hash(pruned) if pruned is not None else None,
                        elapsed_seconds=elapsed,
                        batch_elapsed_seconds=elapsed,
                        prompt_inputs=prompt_inputs,
                        prompt_len=prompt_len,
                    )
                    del generated
            else:
                batch_size = min(args.rollout_batch_size, args.num_rollouts)
                for batch_start in range(0, args.num_rollouts, batch_size):
                    rollout_indices = list(range(batch_start, min(batch_start + batch_size, args.num_rollouts)))
                    missing = [index for index in rollout_indices if (sample_id, index) not in completed]
                    if not missing:
                        continue
                    batch_seed = generation_seed(args.seed, int(source["sample_rank"]), batch_start)
                    set_seed(batch_seed)
                    generated_rows, batch_elapsed, prefix_hash = generate_shared_prefix_batch(
                        model,
                        prompt_inputs,
                        pruning_kwargs,
                        batch_size=len(rollout_indices),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        eos_ids=eos_ids,
                        pad_token_id=int(pad_token_id),
                        last_pruned_inputs_fn=(
                            last_visionzip_pruned_inputs if args.pruning == "visionzip" else None
                        ),
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
                            # One seeded RNG stream is consumed jointly by the
                            # batch; sequence_in_batch identifies the branch.
                            actual_seed=batch_seed,
                            batch_seed=batch_seed,
                            sequence_in_batch=sequence_in_batch,
                            batch_size=len(rollout_indices),
                            engine=(
                                "shared_pruned_prefix_kv_cache"
                                if args.pruning == "visionzip"
                                else "shared_full_prefix_kv_cache"
                            ),
                            prefix_hash=prefix_hash,
                            elapsed_seconds=effective_elapsed,
                            batch_elapsed_seconds=batch_elapsed,
                            prompt_inputs=prompt_inputs,
                            prompt_len=prompt_len,
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
    validation = {
        "status": "passed"
        if not failures
        and len(rows) == expected_records
        and observed_set == expected_keys
        and len(observed_keys) == len(observed_set)
        and not forbidden_present
        else "failed",
        "raw_only": True,
        "post_processing_applied": False,
        "benchmark": args.benchmark,
        "decode_mode": args.decode_mode,
        "pruning": args.pruning,
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
        "elapsed_seconds": time.time() - started,
    }
    if args.pruning == "visionzip" and rows:
        realized = [float(row["realized_retention_ratio"]) for row in rows]
        validation["realized_retention_ratio"] = {
            "mean": sum(realized) / len(realized),
            "min": min(realized),
            "max": max(realized),
        }
    atomic_json(validation_path, validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
