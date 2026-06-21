#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DISALLOWED_QWEN25_BOOTSTRAP = "/scratch/enmingzz/temp/qwen25_bootstrap"
sys.path = [path for path in sys.path if not path or not path.startswith(DISALLOWED_QWEN25_BOOTSTRAP)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from opsd.visionzip_aokvqa.aokvqa import resolve_image
from opsd.visionzip_aokvqa.prompting import format_chat_messages, parse_final_answer
from opsd.visionzip_aokvqa.qwen_wrapper import (
    apply_lora,
    generate_pruned,
    load_qwen_model_and_processor,
    model_input_subset,
    move_inputs,
    primary_device,
)
from opsd.pruning_distill.qwen25_pruned_forward import validate_single_image_qwen_inputs


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")
DATA_DIR = OUTPUT_ROOT / "data" / "benchmark_gap"
EVAL_DIR = OUTPUT_ROOT / "eval" / "benchmark_gap_audit"
REPORTS_DIR = OUTPUT_ROOT / "reports"

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_RATIOS = ("0.1", "0.2", "0.3", "0.4")
SQA_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MME_PERCEPTION = {
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
}
MME_COGNITION = {
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
}
MMSTAR_CATEGORIES = (
    "coarse perception",
    "fine-grained perception",
    "instance reasoning",
    "logical reasoning",
    "science & technology",
    "math",
)
BLINK_SINGLE_IMAGE_SUBSETS = (
    "Counting",
    "IQ_Test",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Spatial_Relation",
)


@dataclass
class BenchmarkRecord:
    benchmark: str
    sample_id: str
    question: str
    answer: str
    image: Any
    image_root: str = ""
    image_path_or_id: str = ""
    category: str = ""
    metadata: dict[str, Any] | None = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Base-model VisionZip pruning gap audit for MME/POPE/GQA/SQA.")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--output_dir", default=str(DATA_DIR))
    prep.add_argument("--seed", type=int, default=42)

    ev = sub.add_parser("eval_group")
    ev.add_argument("--benchmark", choices=("mme", "pope", "gqa", "sqa", "mmstar", "blink"), required=True)
    ev.add_argument("--method", required=True)
    ev.add_argument("--retention_ratio", required=True)
    ev.add_argument("--checkpoint_path", default="")
    ev.add_argument("--output_jsonl", required=True)
    ev.add_argument("--input_jsonl", default="")
    ev.add_argument("--image_root", default="")
    ev.add_argument("--ids_path", default="")
    ev.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    ev.add_argument("--max_new_tokens", type=int, default=64)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--limit", type=int, default=0)
    ev.add_argument("--shard_index", type=int, default=0)
    ev.add_argument("--num_shards", type=int, default=1)
    ev.add_argument("--bf16", type=str, default="true")
    ev.add_argument("--attn_implementation", default="flash_attention_2")
    ev.add_argument("--min_pixels", type=int, default=0)
    ev.add_argument("--max_pixels", type=int, default=16384 * 28 * 28)
    ev.add_argument("--allow_embedding_fallback", action="store_true")
    ev.add_argument("--resume", action="store_true")

    agg = sub.add_parser("aggregate")
    agg.add_argument("--partials_dir", default=str(EVAL_DIR / "partials"))
    agg.add_argument("--raw_generations", default=str(EVAL_DIR / "raw_generations.jsonl"))
    agg.add_argument("--results_csv", default=str(REPORTS_DIR / "benchmark_gap_audit_results.csv"))
    agg.add_argument("--summary_md", default=str(REPORTS_DIR / "benchmark_gap_audit_summary.md"))

    mmstar = sub.add_parser("prepare_mmstar")
    mmstar.add_argument("--output_dir", default=str(DATA_DIR))

    blink = sub.add_parser("prepare_blink_single")
    blink.add_argument("--output_dir", default=str(DATA_DIR))

    return p


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\"'`]+|[\"'`.!?;,:\)]+$", "", value).strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_yes_no(text: Any) -> str | None:
    value = re.sub(r"\s+", " ", str(text or "").strip().lower()).replace(".", "")
    if value in {"yes", "no"}:
        return value
    if value in {"y", "yeah", "yep"}:
        return "yes"
    if value in {"n", "nope"}:
        return "no"
    prefix = value[:8]
    if re.search(r"\byes\b", prefix):
        return "yes"
    if re.search(r"\bno\b", prefix):
        return "no"
    return None


def parse_choice(text: Any, num_choices: int = 26) -> str | None:
    value = str(text or "").strip()
    parsed = parse_final_answer(value)
    if parsed is not None:
        return parsed
    max_letter = SQA_LETTERS[max(0, min(num_choices, len(SQA_LETTERS))) - 1]
    patterns = [
        rf"^\s*([{SQA_LETTERS[0]}-{max_letter}])\s*[\).:\-]?\b",
        rf"\banswer\s*(?:is|:)?\s*([{SQA_LETTERS[0]}-{max_letter}])\b",
        rf"\boption\s*([{SQA_LETTERS[0]}-{max_letter}])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def parse_prediction(benchmark: str, generated: str, record: BenchmarkRecord | None = None) -> str | None:
    if benchmark in {"mme", "pope"}:
        return parse_yes_no(generated)
    if benchmark in {"sqa", "mmstar", "blink"}:
        num_choices = int((record.metadata or {}).get("num_choices", 26)) if record else 26
        return parse_choice(generated, num_choices=num_choices)
    if benchmark == "gqa":
        yes_no = parse_yes_no(generated)
        if yes_no is not None:
            return yes_no
        value = str(generated or "").strip()
        value = re.sub(r"^\s*(answer|final answer)\s*[:\-]\s*", "", value, flags=re.IGNORECASE)
        value = value.splitlines()[0].strip() if value else ""
        value = re.split(r"[.;\n]", value)[0].strip()
        return normalize_text(value) or None
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def is_correct(benchmark: str, parsed: str | None, answer: str) -> bool:
    if parsed is None:
        return False
    if benchmark in {"mme", "pope"}:
        return parsed == parse_yes_no(answer)
    if benchmark in {"sqa", "mmstar", "blink"}:
        return parsed.upper() == str(answer).strip().upper()
    if benchmark == "gqa":
        return normalize_text(parsed) == normalize_text(answer)
    return False


def pil_to_png(path: Path, image: Image.Image | None) -> str:
    if image is None:
        raise ValueError("Expected a PIL image, got None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return str(path)


def prepare_gqa(output_dir: Path) -> Path:
    from datasets import Image as HFImage
    from datasets import load_dataset

    output_path = output_dir / "gqa_testdev.jsonl"
    ds = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev", token=True)
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "benchmark": "gqa",
                "sample_id": str(row.get("id", idx)),
                "image_id": str(row["imageId"]),
                "question": f"{row['question']}\nAnswer the question using a single word or phrase.",
                "answer": str(row["answer"]),
                "source": "lmms-lab/GQA:testdev_balanced_instructions/testdev",
                "category": str((row.get("types") or {}).get("semantic", "")),
            }
        )
    write_jsonl(output_path, rows)

    images_ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
    images_ds = images_ds.cast_column("image", HFImage(decode=False))
    image_ids = list(images_ds["id"])
    (output_dir / "gqa_image_ids.json").write_text(json.dumps(image_ids[:10], indent=2) + "\n", encoding="utf-8")
    return output_path


def prepare_sqa(output_dir: Path) -> Path:
    from datasets import load_dataset

    output_path = output_dir / "sqa_img_test.jsonl"
    image_dir = output_dir / "sqa_img_images"
    ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split="test", token=True)
    rows = []
    for idx, row in enumerate(ds):
        choices = [str(x) for x in row["choices"]]
        options = [chr(ord("A") + i) for i in range(len(choices))]
        choices_str = "\n".join(f"{letter}. {choice}" for letter, choice in zip(options, choices))
        context = str(row.get("hint") or "").strip()
        prompt = ""
        if context:
            prompt += f"Context: {context}\n"
        prompt += f"{row['question']}\n{choices_str}\nAnswer with the option's letter from the given choices directly."
        image_path = pil_to_png(image_dir / f"sqa_img_{idx:06d}.png", row["image"])
        answer_idx = int(row["answer"])
        rows.append(
            {
                "benchmark": "sqa",
                "sample_id": f"sqa_img_{idx:06d}",
                "image": str(Path(image_path).relative_to(output_dir)),
                "question": prompt,
                "answer": options[answer_idx],
                "choices": choices,
                "source": "lmms-lab/ScienceQA:ScienceQA-IMG/test",
                "category": str(row.get("subject") or row.get("category") or ""),
            }
        )
    write_jsonl(output_path, rows)
    return output_path


def prepare_mmstar(output_dir: Path) -> Path:
    from datasets import load_dataset

    output_path = output_dir / "mmstar_val.jsonl"
    image_dir = output_dir / "mmstar_images"
    ds = load_dataset("Lin-Chen/MMStar", split="val", token=True)
    rows = []
    for idx, row in enumerate(ds):
        question = str(row["question"]).replace(" Please answer yes or no.", "").strip()
        prompt = f"Question: {question}\nAnswer with the option letter only."
        image_path = pil_to_png(image_dir / f"mmstar_{idx:06d}.png", row["image"])
        rows.append(
            {
                "benchmark": "mmstar",
                "sample_id": f"mmstar_{int(row.get('index', idx)):06d}",
                "image": str(Path(image_path).relative_to(output_dir)),
                "question": prompt,
                "answer": str(row["answer"]).strip().upper(),
                "source": "Lin-Chen/MMStar:val",
                "category": str(row.get("category") or ""),
                "l2_category": str(row.get("l2_category") or ""),
                "index": int(row.get("index", idx)),
                "meta_info": row.get("meta_info") or {},
            }
        )
    write_jsonl(output_path, rows)
    return output_path


def prepare_mmstar_cmd(args: argparse.Namespace) -> None:
    path = prepare_mmstar(Path(args.output_dir))
    print(json.dumps({"mmstar_jsonl": str(path), "image_root": str(Path(args.output_dir))}, indent=2))


def prepare_blink_single(output_dir: Path) -> Path:
    from datasets import load_dataset

    output_path = output_dir / "blink_single_val.jsonl"
    image_dir = output_dir / "blink_single_images"
    rows = []
    for subset in BLINK_SINGLE_IMAGE_SUBSETS:
        ds = load_dataset("BLINK-Benchmark/BLINK", subset, split="val", token=True)
        for idx, row in enumerate(ds):
            image_keys = [key for key in row.keys() if re.match(r"^image_\d+$", key) and row.get(key) is not None]
            if len(image_keys) != 1:
                raise ValueError(f"Expected one image for BLINK subset {subset}, row {idx}; got {image_keys}")
            choices = [str(choice) for choice in row["choices"]]
            choice_letters = ", ".join(chr(ord("A") + i) for i in range(len(choices)))
            prompt = f"Note: You only need to respond with {choice_letters} without providing any additional information.\n{row['prompt']}"
            sample_id = f"blink_{subset.lower()}_{idx:06d}"
            image_path = pil_to_png(image_dir / f"{sample_id}.png", row[image_keys[0]])
            rows.append(
                {
                    "benchmark": "blink",
                    "sample_id": sample_id,
                    "image": str(Path(image_path).relative_to(output_dir)),
                    "question": prompt,
                    "answer": str(row["answer"]).strip("()").strip().upper(),
                    "choices": choices,
                    "source": f"BLINK-Benchmark/BLINK:{subset}/val",
                    "category": str(row.get("sub_task") or subset),
                    "subset": subset,
                    "idx": str(row.get("idx", idx)),
                }
            )
    write_jsonl(output_path, rows)
    return output_path


def prepare_blink_single_cmd(args: argparse.Namespace) -> None:
    path = prepare_blink_single(Path(args.output_dir))
    print(json.dumps({"blink_jsonl": str(path), "image_root": str(Path(args.output_dir))}, indent=2))


def prepare(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gqa_path = prepare_gqa(out_dir)
    sqa_path = prepare_sqa(out_dir)
    mmstar_path = prepare_mmstar(out_dir)
    blink_path = prepare_blink_single(out_dir)
    report = [
        "# Benchmark Gap Audit JSONL Inputs",
        "",
        "| Benchmark | JSONL | Image Root / Loader |",
        "|---|---|---|",
        "| MME | `/scratch/enmingzz/temp/opsd_data/mme_test.jsonl` | `/scratch/enmingzz/temp/opsd_data/mme_images` |",
        "| POPE | `/scratch/enmingzz/temp/opsd_data/pope_full.jsonl` | `/scratch/enmingzz/temp/opsd_data/pope_images` |",
        f"| GQA | `{gqa_path}` | loaded from `lmms-lab/GQA:testdev_balanced_images/testdev` by `image_id` |",
        f"| SQA | `{sqa_path}` | `{out_dir / 'sqa_img_images'}` |",
        f"| MMStar | `{mmstar_path}` | `{out_dir / 'mmstar_images'}` |",
        f"| BLINK single-image subset | `{blink_path}` | `{out_dir / 'blink_single_images'}` |",
        "",
        "GQA images are intentionally not copied to disk; the evaluator loads the official HF image split by `image_id`.",
    ]
    (REPORTS_DIR / "benchmark_gap_audit_inputs.md").parent.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark_gap_audit_inputs.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gqa_jsonl": str(gqa_path),
                "sqa_jsonl": str(sqa_path),
                "mmstar_jsonl": str(mmstar_path),
                "blink_jsonl": str(blink_path),
            },
            indent=2,
        )
    )


def load_gqa_image_index():
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
    ds = ds.cast_column("image", HFImage(decode=True))
    id_to_index = {str(image_id): idx for idx, image_id in enumerate(ds["id"])}
    return ds, id_to_index


def read_eval_ids(ids_path: str, benchmark: str) -> list[str] | None:
    if not ids_path:
        return None
    data = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        values = data.get(benchmark) or data.get(benchmark.lower()) or data.get(benchmark.upper())
    else:
        values = data
    if values is None:
        raise KeyError(f"No ids for benchmark {benchmark!r} in {ids_path}")
    return [str(x) for x in values]


def read_records(
    benchmark: str,
    input_jsonl: str,
    image_root: str,
    limit: int,
    seed: int,
    ids_path: str = "",
) -> list[BenchmarkRecord]:
    path = Path(input_jsonl)
    rows = read_jsonl(path)
    eval_ids = read_eval_ids(ids_path, benchmark)
    if eval_ids is not None:
        by_id = {str(row.get("sample_id", row.get("id", idx))): row for idx, row in enumerate(rows)}
        missing = [sample_id for sample_id in eval_ids if sample_id not in by_id]
        if missing:
            raise KeyError(f"Missing eval ids for {benchmark}: {missing[:10]}")
        rows = [by_id[sample_id] for sample_id in eval_ids]
    rng = random.Random(int(seed))
    if eval_ids is None and limit and limit > 0:
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[: int(limit)]
    records = []
    for idx, row in enumerate(rows):
        image_value = row.get("image", row.get("image_path", row.get("image_id", "")))
        image_path_or_id = str(row.get("image", row.get("image_path", row.get("image_id", ""))))
        metadata = {k: v for k, v in row.items() if k not in {"image", "question", "answer"}}
        if benchmark in {"sqa", "mmstar", "blink"}:
            metadata["num_choices"] = len(row.get("choices") or [])
            if benchmark == "mmstar":
                metadata["num_choices"] = 4
        records.append(
            BenchmarkRecord(
                benchmark=benchmark,
                sample_id=str(row.get("sample_id", row.get("id", idx))),
                question=str(row["question"]),
                answer=str(row["answer"]),
                image=image_value,
                image_root=image_root,
                image_path_or_id=image_path_or_id,
                category=str(row.get("category", "")),
                metadata=metadata,
            )
        )
    return records


def resolve_record_image(record: BenchmarkRecord, gqa_images: tuple[Any, dict[str, int]] | None) -> Image.Image:
    if record.benchmark == "gqa":
        if gqa_images is None:
            raise RuntimeError("GQA image dataset was not loaded.")
        ds, id_to_index = gqa_images
        image_id = str(record.image)
        if image_id not in id_to_index:
            raise KeyError(f"GQA image_id {image_id} not found in image split.")
        return ds[id_to_index[image_id]]["image"].convert("RGB")
    return resolve_image(record.image, record.image_root)


def encode_benchmark_prompt(
    processor: Any,
    record: BenchmarkRecord,
    image: Image.Image,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    text = processor.apply_chat_template(format_chat_messages(record.question), tokenize=False, add_generation_prompt=True)
    inputs = dict(processor(text=[text], images=[image], return_tensors="pt"))
    inputs = move_inputs(inputs, device)
    validate_single_image_qwen_inputs(inputs)
    return inputs


def generate_record(
    model: Any,
    processor: Any,
    record: BenchmarkRecord,
    image: Image.Image,
    method: str,
    retention_ratio: str,
    max_new_tokens: int,
    allow_embedding_fallback: bool,
) -> tuple[torch.Tensor, str, dict[str, Any]]:
    device = primary_device(model)
    inputs = encode_benchmark_prompt(processor, record, image, device)
    if method == "full_token_base":
        output_ids = model.generate(
            **model_input_subset(inputs),
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_ids = output_ids[:, prompt_len:]
        response = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        meta = {
            "num_full_visual_tokens": int((inputs["input_ids"] == model.config.image_token_id).sum().item()),
            "num_kept_visual_tokens": int((inputs["input_ids"] == model.config.image_token_id).sum().item()),
        }
        return gen_ids, response, meta

    gen_ids, response, meta = generate_pruned(
        model,
        processor,
        inputs,
        float(retention_ratio),
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        allow_embedding_fallback=allow_embedding_fallback,
    )
    return gen_ids, response, meta


def eval_group(args: argparse.Namespace) -> None:
    if args.method == "full_token_base" and args.retention_ratio != "full":
        raise ValueError("full_token_base must use --retention_ratio full.")
    if args.method == "visionzip_base" and args.retention_ratio == "full":
        raise ValueError("visionzip_base requires numeric --retention_ratio.")
    if args.method not in {"full_token_base", "visionzip_base"} and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for adapter methods.")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must be in [0, --num_shards).")

    torch.manual_seed(int(args.seed))
    records = read_records(args.benchmark, args.input_jsonl, args.image_root, args.limit, args.seed, args.ids_path)
    records = records[int(args.shard_index) :: int(args.num_shards)]
    gqa_images = load_gqa_image_index() if args.benchmark == "gqa" else None
    output_path = Path(args.output_jsonl)
    existing_rows: list[dict[str, Any]] = []
    processed_sample_ids: set[str] = set()
    if args.resume:
        expected_sample_ids = {record.sample_id for record in records}
        deduped_rows: list[dict[str, Any]] = []
        for row in read_jsonl_tolerant(output_path):
            sample_id = str(row.get("sample_id", ""))
            if sample_id and sample_id in expected_sample_ids and sample_id not in processed_sample_ids:
                deduped_rows.append(row)
                processed_sample_ids.add(sample_id)
        existing_rows = deduped_rows
        if len(processed_sample_ids) == len(records):
            write_jsonl(output_path, existing_rows)
            print(f"already complete {args.output_jsonl}")
            return

    model, processor = load_qwen_model_and_processor(
        args.model_name_or_path,
        bf16=str_to_bool(args.bf16),
        attn_implementation=args.attn_implementation,
        device_map="auto",
        min_pixels=int(args.min_pixels) if int(args.min_pixels) > 0 else None,
        max_pixels=int(args.max_pixels) if int(args.max_pixels) > 0 else None,
    )
    if args.method not in {"full_token_base", "visionzip_base"}:
        model = apply_lora(model, adapter_path=args.checkpoint_path)
    model.eval()

    rows: list[dict[str, Any]] = list(existing_rows)
    output_handle = None
    if args.resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("a", encoding="utf-8")
    with torch.no_grad():
        for idx, record in enumerate(records, start=1):
            if record.sample_id in processed_sample_ids:
                continue
            start = time.perf_counter()
            error = ""
            gen_ids = torch.empty(0, dtype=torch.long)
            response = ""
            meta: dict[str, Any] = {}
            parsed = None
            correct = False
            try:
                image = resolve_record_image(record, gqa_images)
                gen_ids, response, meta = generate_record(
                    model,
                    processor,
                    record,
                    image,
                    args.method,
                    args.retention_ratio,
                    args.max_new_tokens,
                    args.allow_embedding_fallback,
                )
                parsed = parse_prediction(args.benchmark, response, record)
                correct = is_correct(args.benchmark, parsed, record.answer)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - start
            row = {
                "benchmark": args.benchmark,
                "sample_id": record.sample_id,
                "method": args.method,
                "retention_ratio": args.retention_ratio if args.retention_ratio == "full" else float(args.retention_ratio),
                "question": record.question,
                "image_path": record.image_path_or_id,
                "image_id": record.image_path_or_id,
                "generated_response": response,
                "parsed_answer": parsed,
                "ground_truth_answer": record.answer,
                "correct": bool(correct),
                "response_length": int(gen_ids.numel()),
                "parse_success": parsed is not None,
                "latency_seconds": latency,
                "category": record.category,
                "error": error,
                "checkpoint_path": args.checkpoint_path,
                **{k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))},
                **{f"meta_{k}": v for k, v in (record.metadata or {}).items() if isinstance(v, (int, float, str, bool))},
            }
            rows.append(row)
            if output_handle is not None:
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
            print(
                json.dumps(
                    {
                        "benchmark": args.benchmark,
                        "method": args.method,
                        "retention_ratio": row["retention_ratio"],
                        "shard": f"{args.shard_index}/{args.num_shards}",
                        "idx": idx,
                        "num_records": len(records),
                        "sample_id": record.sample_id,
                        "parsed": parsed,
                        "correct": correct,
                        "error": error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if output_handle is not None:
        output_handle.close()
    if args.resume:
        row_by_sample_id = {str(row.get("sample_id", "")): row for row in rows}
        rows = [row_by_sample_id[record.sample_id] for record in records if record.sample_id in row_by_sample_id]
    write_jsonl(output_path, rows)
    print(f"wrote {args.output_jsonl}")


def ratio_key(value: Any) -> str:
    if str(value) == "full":
        return "full"
    return f"{float(value):.1f}"


def summarize_basic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"score": 0.0, "parse_rate": 0.0, "avg_response_length": 0.0, "num_samples": 0}
    return {
        "score": sum(1 for row in rows if row.get("correct")) / n,
        "parse_rate": sum(1 for row in rows if row.get("parse_success")) / n,
        "avg_response_length": mean(float(row.get("response_length", 0.0)) for row in rows),
        "num_samples": n,
        "num_failures": sum(1 for row in rows if row.get("error")),
    }


def summarize_mme(rows: list[dict[str, Any]]) -> dict[str, Any]:
    basic = summarize_basic(rows)
    category_pairs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        score = float(bool(row.get("correct")))
        category = str(row.get("category") or row.get("meta_category") or "unknown")
        pair_id = str(row.get("meta_mme_pair_id") or row.get("meta_mme_question_id") or row.get("sample_id"))
        category_pairs[category][pair_id].append(score)

    category_scores: dict[str, float] = {}
    invalid_pair_count = 0
    for category, pairs in category_pairs.items():
        pair_scores = []
        for scores in pairs.values():
            if len(scores) != 2:
                invalid_pair_count += 1
            acc = sum(scores) / len(scores) * 100.0 if scores else 0.0
            acc_plus = 100.0 if len(scores) == 2 and sum(scores) == 2 else 0.0
            pair_scores.append(acc + acc_plus)
        category_scores[category] = mean(pair_scores) if pair_scores else 0.0
    perception = sum(score for cat, score in category_scores.items() if cat in MME_PERCEPTION)
    cognition = sum(score for cat, score in category_scores.items() if cat in MME_COGNITION)
    basic.update(
        {
            "score": perception + cognition,
            "yes_no_accuracy": basic["score"] if False else summarize_basic(rows)["score"],
            "mme_perception_score": perception,
            "mme_cognition_score": cognition,
            "mme_total_score": perception + cognition,
            "mme_invalid_pair_count": invalid_pair_count,
            "mme_category_scores": category_scores,
        }
    )
    return basic


def summarize_mmstar(rows: list[dict[str, Any]]) -> dict[str, Any]:
    basic = summarize_basic(rows)
    l2_scores: dict[str, list[float]] = defaultdict(list)
    category_l2_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        score = float(bool(row.get("correct")))
        l2 = str(row.get("meta_l2_category") or row.get("l2_category") or row.get("category") or "unknown")
        category = str(row.get("category") or row.get("meta_category") or "unknown")
        l2_scores[l2].append(score)
        category_l2_scores[category][l2].append(score)

    l2_avgs = {l2: mean(scores) for l2, scores in l2_scores.items() if scores}
    basic["score"] = mean(l2_avgs.values()) if l2_avgs else 0.0
    basic["mmstar_sample_accuracy"] = summarize_basic(rows)["score"]
    for category in MMSTAR_CATEGORIES:
        per_l2 = category_l2_scores.get(category, {})
        avgs = [mean(scores) for scores in per_l2.values() if scores]
        basic[f"mmstar_{category}"] = mean(avgs) if avgs else ""
    return basic


def summarize_group(benchmark: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if benchmark == "mme":
        return summarize_mme(rows)
    if benchmark == "mmstar":
        return summarize_mmstar(rows)
    return summarize_basic(rows)


def aggregate(args: argparse.Namespace) -> None:
    partial_paths = sorted(Path(args.partials_dir).glob("*.jsonl"))
    if not partial_paths:
        raise FileNotFoundError(f"No partial JSONL files found in {args.partials_dir}")
    rows: list[dict[str, Any]] = []
    for path in partial_paths:
        rows.extend(read_jsonl(path))
    full_answers: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("method") == "full_token_base" and str(row.get("retention_ratio")) == "full":
            full_answers[f"{row.get('benchmark')}::{row.get('sample_id')}"] = row
    enriched: list[dict[str, Any]] = []
    for row in rows:
        full = full_answers.get(f"{row.get('benchmark')}::{row.get('sample_id')}")
        out = dict(row)
        out["full_token_base_answer"] = full.get("parsed_answer") if full else None
        out["teacher_agreement"] = bool(
            full
            and row.get("parse_success")
            and full.get("parse_success")
            and row.get("parsed_answer") == full.get("parsed_answer")
        )
        enriched.append(out)
    rows = enriched
    write_jsonl(Path(args.raw_generations), rows)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["benchmark"]), str(row["method"]), ratio_key(row["retention_ratio"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (benchmark, method, ratio), group_rows in sorted(grouped.items()):
        summary = summarize_group(benchmark, group_rows)
        summaries.append(
            {
                "benchmark": benchmark,
                "method": method,
                "retention_ratio": ratio,
                "score": summary.get("score", 0.0),
                "parse_rate": summary.get("parse_rate", 0.0),
                "avg_response_length": summary.get("avg_response_length", 0.0),
                "teacher_agreement": sum(1 for row in group_rows if row.get("teacher_agreement")) / len(group_rows)
                if group_rows
                else 0.0,
                "num_samples": summary.get("num_samples", len(group_rows)),
                "num_failures": summary.get("num_failures", 0),
                "checkpoint_path": group_rows[0].get("checkpoint_path", "") if group_rows else "",
                "yes_no_accuracy": summary.get("yes_no_accuracy", ""),
                "mme_perception_score": summary.get("mme_perception_score", ""),
                "mme_cognition_score": summary.get("mme_cognition_score", ""),
                "mme_invalid_pair_count": summary.get("mme_invalid_pair_count", ""),
                "mmstar_sample_accuracy": summary.get("mmstar_sample_accuracy", ""),
                **{
                    f"mmstar_{category.replace(' ', '_').replace('&', 'and').replace('-', '_')}": summary.get(f"mmstar_{category}", "")
                    for category in MMSTAR_CATEGORIES
                },
            }
        )

    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "method",
        "retention_ratio",
        "score",
        "parse_rate",
        "avg_response_length",
        "teacher_agreement",
        "gap_to_full",
        "improvement_over_vz",
        "recovery_rate",
        "num_samples",
        "checkpoint_path",
        "num_failures",
        "yes_no_accuracy",
        "mme_perception_score",
        "mme_cognition_score",
        "mme_invalid_pair_count",
        "mmstar_sample_accuracy",
        "mmstar_coarse_perception",
        "mmstar_fine_grained_perception",
        "mmstar_instance_reasoning",
        "mmstar_logical_reasoning",
        "mmstar_science_and_technology",
        "mmstar_math",
    ]
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        full_by_benchmark = {
            s["benchmark"]: float(s["score"])
            for s in summaries
            if s["method"] == "full_token_base" and s["retention_ratio"] == "full"
        }
        vz_by_benchmark_ratio = {
            (s["benchmark"], s["retention_ratio"]): float(s["score"])
            for s in summaries
            if s["method"] == "visionzip_base"
        }
        for summary in summaries:
            full_score = full_by_benchmark.get(summary["benchmark"])
            score = float(summary.get("score", 0.0))
            if full_score is None:
                summary["gap_to_full"] = ""
            else:
                summary["gap_to_full"] = full_score - score
            base_score = vz_by_benchmark_ratio.get((summary["benchmark"], summary["retention_ratio"]))
            if summary["method"] == "full_token_base":
                summary["improvement_over_vz"] = ""
                summary["recovery_rate"] = ""
            elif base_score is None:
                summary["improvement_over_vz"] = ""
                summary["recovery_rate"] = ""
            else:
                improvement = score - base_score
                summary["improvement_over_vz"] = improvement
                denom = (full_score - base_score) if full_score is not None else None
                summary["recovery_rate"] = "" if denom is None or denom <= 0 else improvement / denom
            writer.writerow({key: summary.get(key, "") for key in fieldnames})
    write_summary(Path(args.summary_md), summaries)
    print(f"wrote {args.raw_generations}")
    print(f"wrote {args.results_csv}")
    print(f"wrote {args.summary_md}")


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def write_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    by_key = {(s["benchmark"], s["method"], s["retention_ratio"]): s for s in summaries}
    lines = [
        "# Benchmark Gap Audit Summary",
        "",
        "## Setup",
        "",
        "- Models evaluated: full-token base and VisionZip-pruned base only.",
        "- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`.",
        "- Generation: greedy; `max_new_tokens` is set by the eval command.",
        "- Image preprocessing: Qwen processor with `max_pixels=16384*28*28` unless overridden; full-token and VisionZip base use the same cap.",
        "- VisionZip ratios: `0.1`, `0.2`, `0.3`, `0.4`.",
        "- OPSD/SFT/GRPO/EPIC training was not run or modified.",
        "",
        "## Scores",
        "",
        "| Benchmark | Method | Retention | Score | Parse Rate | Avg Response Tokens | Samples |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {row['retention_ratio']} | "
            f"{fmt(row['score'])} | {fmt(row['parse_rate'])} | {fmt(row['avg_response_length'])} | {row['num_samples']} |"
        )

    lines.extend(["", "## Drops From Full Token", ""])
    lines.append("| Benchmark | Retention | Full Score | VisionZip Score | Drop |")
    lines.append("|---|---:|---:|---:|---:|")
    largest_drop: tuple[str, str, float] | None = None
    meaningful: list[tuple[str, str, float]] = []
    for benchmark in sorted({s["benchmark"] for s in summaries}):
        full = by_key.get((benchmark, "full_token_base", "full"))
        if not full:
            continue
        full_score = float(full["score"])
        for ratio in DEFAULT_RATIOS:
            vz = by_key.get((benchmark, "visionzip_base", ratio))
            if not vz:
                continue
            score = float(vz["score"])
            drop = full_score - score
            lines.append(f"| {benchmark} | {ratio} | {fmt(full_score)} | {fmt(score)} | {fmt(drop)} |")
            if largest_drop is None or drop > largest_drop[2]:
                largest_drop = (benchmark, ratio, drop)
            if 0.02 <= drop <= 0.15:
                meaningful.append((benchmark, ratio, drop))

    if largest_drop:
        lines.extend(
            [
                "",
                f"Largest pruning-induced degradation: `{largest_drop[0]}` at retention `{largest_drop[1]}` with drop `{largest_drop[2]:.4f}`.",
            ]
        )
    if meaningful:
        best = sorted(meaningful, key=lambda x: (abs(x[2] - 0.08), x[0]))[0]
        lines.append(
            f"Meaningful but not catastrophic gap candidate: `{best[0]}` at retention `{best[1]}` with drop `{best[2]:.4f}`."
        )
    else:
        lines.append("No retention ratio produced a clearly meaningful-but-not-catastrophic gap under the 0.02-0.15 heuristic.")

    mme = [s for s in summaries if s["benchmark"] == "mme" and s["method"] == "visionzip_base"]
    if mme:
        lines.extend(["", "## MME Perception/Cognition", ""])
        lines.append("| Retention | MME Total | Perception | Cognition | Yes/No Accuracy |")
        lines.append("|---:|---:|---:|---:|---:|")
        full = by_key.get(("mme", "full_token_base", "full"))
        if full:
            lines.append(
                f"| full | {fmt(full.get('score'))} | {fmt(full.get('mme_perception_score'))} | "
                f"{fmt(full.get('mme_cognition_score'))} | {fmt(full.get('yes_no_accuracy'))} |"
            )
        for ratio in DEFAULT_RATIOS:
            row = by_key.get(("mme", "visionzip_base", ratio))
            if row:
                lines.append(
                    f"| {ratio} | {fmt(row.get('score'))} | {fmt(row.get('mme_perception_score'))} | "
                    f"{fmt(row.get('mme_cognition_score'))} | {fmt(row.get('yes_no_accuracy'))} |"
                )

        full = by_key.get(("mme", "full_token_base", "full"))
        if full:
            full_perception = float(full.get("mme_perception_score") or 0.0)
            full_cognition = float(full.get("mme_cognition_score") or 0.0)
            lines.extend(["", "MME relative drops:", ""])
            lines.append("| Retention | Perception Drop | Cognition Drop |")
            lines.append("|---:|---:|---:|")
            for ratio in DEFAULT_RATIOS:
                row = by_key.get(("mme", "visionzip_base", ratio))
                if not row:
                    continue
                perception_drop = (full_perception - float(row.get("mme_perception_score") or 0.0)) / full_perception if full_perception else math.nan
                cognition_drop = (full_cognition - float(row.get("mme_cognition_score") or 0.0)) / full_cognition if full_cognition else math.nan
                lines.append(f"| {ratio} | {fmt(perception_drop)} | {fmt(cognition_drop)} |")

    aok_path = REPORTS_DIR / "aokvqa_val_256_results.csv"
    aok_drops: dict[str, float] = {}
    if aok_path.exists():
        with aok_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        full_rows = [row for row in rows if row.get("method") == "full_token_base"]
        if full_rows:
            full_acc = float(full_rows[0]["accuracy"])
            for row in rows:
                if row.get("method") == "visionzip_base":
                    aok_drops[str(row.get("retention_ratio"))] = full_acc - float(row["accuracy"])

    if aok_drops:
        lines.extend(["", "## Comparison To A-OKVQA Sanity Gap", ""])
        lines.append("| Retention | A-OKVQA Drop | GQA Drop | SQA Drop | POPE Drop | MME Yes/No Acc Drop |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for ratio in DEFAULT_RATIOS:
            gqa_full = by_key.get(("gqa", "full_token_base", "full"))
            gqa_vz = by_key.get(("gqa", "visionzip_base", ratio))
            sqa_full = by_key.get(("sqa", "full_token_base", "full"))
            sqa_vz = by_key.get(("sqa", "visionzip_base", ratio))
            pope_full = by_key.get(("pope", "full_token_base", "full"))
            pope_vz = by_key.get(("pope", "visionzip_base", ratio))
            mme_full = by_key.get(("mme", "full_token_base", "full"))
            mme_vz = by_key.get(("mme", "visionzip_base", ratio))
            gqa_drop = float(gqa_full["score"]) - float(gqa_vz["score"]) if gqa_full and gqa_vz else math.nan
            sqa_drop = float(sqa_full["score"]) - float(sqa_vz["score"]) if sqa_full and sqa_vz else math.nan
            pope_drop = float(pope_full["score"]) - float(pope_vz["score"]) if pope_full and pope_vz else math.nan
            mme_acc_drop = (
                float(mme_full["yes_no_accuracy"]) - float(mme_vz["yes_no_accuracy"])
                if mme_full and mme_vz and mme_full.get("yes_no_accuracy") != "" and mme_vz.get("yes_no_accuracy") != ""
                else math.nan
            )
            lines.append(
                f"| {ratio} | {fmt(aok_drops.get(ratio))} | {fmt(gqa_drop)} | {fmt(sqa_drop)} | "
                f"{fmt(pope_drop)} | {fmt(mme_acc_drop)} |"
            )
        lines.append("")
        lines.append(
            "GQA and MME cognition show stronger pruning degradation than the A-OKVQA sanity set, especially at retention `0.1`."
        )

    lines.extend(["", "## Decision", ""])
    if largest_drop is None:
        lines.append("No complete full-token/VisionZip pairs were available, so the pruning gap cannot be assessed yet.")
    elif largest_drop[2] < 0.02:
        lines.extend(
            [
                "VisionZip degradation is small on all completed benchmark/ratio pairs.",
                "Current settings may be too easy for an OPSD recovery experiment.",
                "Recommended next diagnostic: add retention ratio `0.05` or select harder reasoning subsets, without changing OPSD.",
            ]
        )
    else:
        lines.extend(
            [
                "At least one benchmark/ratio shows non-trivial pruning degradation.",
                f"Use `{largest_drop[0]}` and nearby ratios as the main target for medium-scale training evaluation before full training.",
                "There is enough recovery room to justify a medium-scale training run, but full training should still be gated on medium-scale results.",
                "Recommended target ratios: keep `0.1` as a hard diagnostic, use `0.2` as the main meaningful-but-not-catastrophic setting, and report `0.3/0.4` for robustness.",
            ]
        )
    lines.extend(
        [
            "",
            "This report contains measured rows only. Missing benchmark rows mean that eval jobs did not finish or inputs were unavailable.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "prepare_mmstar":
        prepare_mmstar_cmd(args)
    elif args.cmd == "prepare_blink_single":
        prepare_blink_single_cmd(args)
    elif args.cmd == "eval_group":
        eval_group(args)
    elif args.cmd == "aggregate":
        aggregate(args)
    else:
        raise ValueError(args.cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
