#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vlmeval.smp import dump, load


VALID_UNKNOWN = "UNKNOWN"


class LocalQwenExtractor:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = os.environ.get("MMMU_QWEN_JUDGE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        self.max_new_tokens = int(os.environ.get("MMMU_QWEN_JUDGE_MAX_NEW_TOKENS", "8"))
        device = os.environ.get("MMMU_QWEN_JUDGE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        dtype_name = os.environ.get(
            "MMMU_QWEN_JUDGE_DTYPE", "bfloat16" if torch.cuda.is_available() else "float32"
        )
        torch_dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        device_map = "auto" if device == "auto" else {"": device}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        ).eval()
        self.input_device = next(self.model.parameters()).device

    def generate(self, prompt: str) -> str:
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You extract the final option selected in a model response. "
                    "Do not solve the question and do not judge correctness. "
                    "Return exactly one allowed uppercase option letter, or UNKNOWN when no option is selected."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.input_device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def available_options(row: pd.Series) -> dict[str, str]:
    options: dict[str, str] = {}
    for letter in string.ascii_uppercase:
        if letter not in row or pd.isna(row[letter]):
            continue
        value = str(row[letter]).strip()
        if value:
            options[letter] = value
    return options


def extract_answer_span(response: str) -> str | None:
    matches = re.findall(r"<answer\b[^>]*>(.*?)</answer\s*>", response, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def build_prompt(row: pd.Series, response: str, options: dict[str, str]) -> str:
    option_text = "\n".join(f"{letter}. {value}" for letter, value in options.items())
    answer_span = extract_answer_span(response)
    if answer_span is not None:
        return (
            f"Allowed options:\n{option_text}\n\n"
            "The model's final <answer> span is:\n"
            f"{answer_span}\n\n"
            "Map only this answer span to the selected option. The span is model output, not "
            "ground truth. Do not solve the underlying question and do not assess whether the "
            "choice is correct. "
            f"Return exactly one of: {', '.join(options)}, {VALID_UNKNOWN}."
        )
    return (
        f"Question:\n{row['question']}\n\n"
        f"Allowed options:\n{option_text}\n\n"
        f"Model response:\n{response}\n\n"
        "No complete <answer> tag was present. Extract the option ultimately selected by the "
        "model from its response. Do not solve the question yourself. "
        f"Return exactly one of: {', '.join(options)}, {VALID_UNKNOWN}."
    )


def parse_extraction(raw: str, options: set[str]) -> str:
    normalized = raw.strip().upper()
    if normalized in options:
        return normalized
    if normalized == VALID_UNKNOWN:
        return VALID_UNKNOWN
    matches = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", normalized)
    valid = [item for item in matches if item in options]
    if len(set(valid)) == 1:
        return valid[0]
    return VALID_UNKNOWN


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    result_file = args.result_file.expanduser().resolve()
    if not result_file.is_file():
        raise FileNotFoundError(result_file)

    output_dir = (args.output_dir or result_file.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result_file.stem
    cache_file = output_dir / f"{stem}_qwen7extract_cache.pkl"
    storage_file = output_dir / f"{stem}_qwen7extract.xlsx"
    score_file = output_dir / f"{stem}_qwen7extract_score.csv"
    summary_file = output_dir / f"{stem}_qwen7extract_summary.json"

    if args.overwrite:
        for path in (cache_file, storage_file, score_file, summary_file):
            path.unlink(missing_ok=True)
    if summary_file.is_file() and not args.overwrite:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("status") == "passed" and args.max_rows is None:
            print(json.dumps(summary, indent=2))
            return 0

    data = load(str(result_file)).sort_values(by="index").reset_index(drop=True)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        data = data.iloc[: args.max_rows].copy()

    cache = load(str(cache_file)) if cache_file.is_file() else {}
    extractor: LocalQwenExtractor | None = None
    prompt_hashes: list[str] = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="MMMU-Pro Qwen7 extraction"):
        key = str(row["index"])
        if key in cache:
            continue
        options = available_options(row)
        if not options:
            raise RuntimeError(f"No options for sample {key}")
        response = str(row.get("raw_prediction", row["prediction"]))
        prompt = build_prompt(row, response, options)
        prompt_hashes.append(sha256_text(prompt))
        if extractor is None:
            extractor = LocalQwenExtractor()
        raw_judgement = extractor.generate(prompt)
        cache[key] = {
            "extracted": parse_extraction(raw_judgement, set(options)),
            "raw_judgement": raw_judgement,
            "prompt_sha256": sha256_text(prompt),
        }
        if len(cache) % 20 == 0:
            dump(cache, str(cache_file))
    dump(cache, str(cache_file))

    missing = [str(index) for index in data["index"] if str(index) not in cache]
    if missing:
        raise RuntimeError(f"Missing cached judgements: {missing[:5]}")
    data["qwen_extracted"] = [cache[str(index)]["extracted"] for index in data["index"]]
    data["qwen_judge_raw"] = [cache[str(index)]["raw_judgement"] for index in data["index"]]
    data["qwen_hit"] = [
        str(prediction).strip().upper() == str(answer).strip().upper()
        for prediction, answer in zip(data["qwen_extracted"], data["answer"])
    ]
    dump(data, str(storage_file))

    score_row: dict[str, object] = {
        "split": "Overall",
        "Overall": 100.0 * float(data["qwen_hit"].mean()),
    }
    group_column = "subject" if "subject" in data else "category"
    if group_column in data:
        for name, group in data.groupby(group_column):
            score_row[str(name)] = 100.0 * float(group["qwen_hit"].mean())
    pd.DataFrame([score_row]).to_csv(score_file, index=False)

    model_path = os.environ.get("MMMU_QWEN_JUDGE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    summary = {
        "status": "passed",
        "dataset": "MMMU_Pro_4c",
        "rows": int(len(data)),
        "scope": "all",
        "qwen_model": model_path,
        "qwen_temperature": 0.0,
        "ground_truth_visible_to_extractor": False,
        "answer_tag_span_provided_to_extractor": True,
        "tagged_answer_prompt_excludes_reasoning": True,
        "qwen_extracted_rows": int((data["qwen_extracted"] != VALID_UNKNOWN).sum()),
        "qwen_unknown_rows": int((data["qwen_extracted"] == VALID_UNKNOWN).sum()),
        "qwen_score_percent": score_row["Overall"],
        "result_file": str(result_file),
        "storage_file": str(storage_file),
        "cache_file": str(cache_file),
        "score_file": str(score_file),
        "prompt_hash_count": len(set(prompt_hashes)),
    }
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
