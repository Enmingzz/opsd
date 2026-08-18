from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vlmeval.dataset.utils.yorn import MME_rating, POPE_rating, YOrN_Extraction
from vlmeval.smp import dump, load


class LocalQwenYesNoJudge:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = os.environ.get("QWEN_YN_JUDGE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        self.max_new_tokens = int(os.environ.get("QWEN_YN_JUDGE_MAX_NEW_TOKENS", "16"))
        device = os.environ.get("QWEN_YN_JUDGE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        dtype_name = os.environ.get("QWEN_YN_JUDGE_DTYPE", "bfloat16" if torch.cuda.is_available() else "float32")
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

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = messages[0]["content"] + "\n\n" + messages[1]["content"]

        inputs = self.tokenizer([text], return_tensors="pt").to(self.input_device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_semantic_yes_no_prompt(line: pd.Series) -> str:
    return (
        "Given a yes/no question and a model answer, decide whether the answer means Yes or No. "
        "If unclear, output Unknown. Return exactly one word: Yes, No, or Unknown.\n"
        f"Question: {line['question']}\n"
        f"Model answer: {line['prediction']}\n"
        "Output:"
    )


def parse_case(raw_case: str) -> tuple[str, str, str, Path]:
    parts = raw_case.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Bad --case {raw_case!r}; expected METHOD:RATIO:DATASET:RESULT_FILE")
    method, ratio, dataset, result_file = parts
    dataset = dataset.upper()
    if dataset not in {"MME", "POPE"}:
        raise ValueError(f"Bad dataset {dataset!r}; expected MME or POPE")
    path = Path(result_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_name = f"Qwen_{dataset}.xlsx"
    if path.name != expected_name:
        raise ValueError(f"Expected result file named {expected_name}, got {path.name}")
    return method, ratio, dataset, path


def load_cache(cache_file: Path) -> dict[str, dict[str, str]]:
    if not cache_file.exists():
        return {}
    raw = json.loads(cache_file.read_text())
    cache: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            judged = str(value.get("judged", "Unknown"))
            raw_text = str(value.get("raw", ""))
        else:
            judged = str(value)
            raw_text = str(value)
        cache[str(key)] = {"judged": judged, "raw": raw_text}
    return cache


def write_cache(cache_file: Path, cache: dict[str, dict[str, str]]) -> None:
    cache_file.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def score_case(
    *,
    method: str,
    ratio: str,
    dataset: str,
    result_file: Path,
    out_dir: Path,
    judge_holder: dict[str, LocalQwenYesNoJudge | None],
    force: bool,
) -> dict[str, object]:
    prefix = f"{method}_{ratio}_{dataset}_qwen_unknown"
    aux_file = out_dir / f"{prefix}_auxmatch.xlsx"
    score_file = out_dir / f"{prefix}_score.csv"
    cache_file = out_dir / f"{prefix}_judge_cache.json"

    if aux_file.exists() and score_file.exists() and not force:
        data = load(str(aux_file))
        score = load(str(score_file))
        return make_summary(method, ratio, dataset, result_file, aux_file, score_file, data, score, skipped=True)

    data = load(str(result_file)).sort_values(by="index").reset_index(drop=True)
    data["prediction"] = [str(x) for x in data["prediction"]]
    ans_map = {idx: YOrN_Extraction(pred) for idx, pred in zip(data["index"], data["prediction"])}
    unknown_before = int(sum(value == "Unknown" for value in ans_map.values()))

    cache = load_cache(cache_file)
    for idx, value in cache.items():
        numeric_idx = int(idx)
        judged = value.get("judged", "Unknown")
        if ans_map.get(numeric_idx) == "Unknown" and judged in {"Yes", "No", "Unknown"}:
            ans_map[numeric_idx] = judged

    unknown_rows = [data.iloc[i] for i in range(len(data)) if ans_map[data.iloc[i]["index"]] == "Unknown"]
    for line in tqdm(unknown_rows, desc=f"{method} {ratio} {dataset} qwen7yn"):
        idx = int(line["index"])
        key = str(idx)
        if key not in cache:
            if judge_holder["judge"] is None:
                judge_holder["judge"] = LocalQwenYesNoJudge()
            prompt = build_semantic_yes_no_prompt(line)
            raw = judge_holder["judge"].generate(prompt)
            judged = YOrN_Extraction(raw)
            cache[key] = {"judged": judged, "raw": raw}
            if len(cache) % 20 == 0:
                write_cache(cache_file, cache)
        ans_map[idx] = cache[key]["judged"]

    write_cache(cache_file, cache)
    data["extracted"] = [ans_map[idx] for idx in data["index"]]
    data["score"] = data["answer"] == data["extracted"]
    dump(data, str(aux_file))

    if dataset == "MME":
        score = MME_rating(str(aux_file))
    elif dataset == "POPE":
        score = POPE_rating(str(aux_file))
    else:
        raise AssertionError(dataset)
    dump(score, str(score_file))

    return make_summary(method, ratio, dataset, result_file, aux_file, score_file, data, score, skipped=False, unknown_before=unknown_before)


def make_summary(
    method: str,
    ratio: str,
    dataset: str,
    result_file: Path,
    aux_file: Path,
    score_file: Path,
    data: pd.DataFrame,
    score: pd.DataFrame,
    *,
    skipped: bool,
    unknown_before: int | None = None,
) -> dict[str, object]:
    unknown_after = int((data["extracted"] == "Unknown").sum())
    if unknown_before is None:
        unknown_before = unknown_after
    row: dict[str, object] = {
        "method": method,
        "ratio": ratio,
        "dataset": dataset,
        "n": int(len(data)),
        "unknown_before": int(unknown_before),
        "unknown_after": unknown_after,
        "fixed": int(unknown_before - unknown_after),
        "skipped_existing": skipped,
        "result_file": str(result_file),
        "aux_file": str(aux_file),
        "score_file": str(score_file),
    }
    if dataset == "MME":
        row["primary_score"] = float(score["perception"].iloc[0] + score["reasoning"].iloc[0])
    elif dataset == "POPE":
        overall = score[score["split"] == "Overall"].iloc[0]
        row["primary_score"] = float(overall["Overall"])
        row["pope_acc"] = float(overall["acc"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="METHOD:RATIO:DATASET:/path/to/Qwen_DATASET.xlsx")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_holder: dict[str, LocalQwenYesNoJudge | None] = {"judge": None}
    summaries = []
    for raw_case in args.case:
        method, ratio, dataset, result_file = parse_case(raw_case)
        summaries.append(
            score_case(
                method=method,
                ratio=ratio,
                dataset=dataset,
                result_file=result_file,
                out_dir=out_dir,
                judge_holder=judge_holder,
                force=args.force,
            )
        )

    summary = pd.DataFrame(summaries)
    summary_file = out_dir / f"{summary['method'].iloc[0]}_qwen_unknown_judge_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(summary.to_string(index=False))
    print(f"[posthoc] summary={summary_file}")


if __name__ == "__main__":
    main()
