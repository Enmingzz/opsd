from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from tqdm import tqdm

from vlmeval.smp import dump, load


class LocalQwenJudge:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = os.environ.get("MATH_QWEN_JUDGE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        self.max_new_tokens = int(os.environ.get("MATH_QWEN_JUDGE_MAX_NEW_TOKENS", "128"))
        device = os.environ.get("MATH_QWEN_JUDGE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        dtype_name = os.environ.get("MATH_QWEN_JUDGE_DTYPE", "bfloat16" if torch.cuda.is_available() else "float32")
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

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict math answer extraction and grading assistant. "
                    "Follow the requested output format exactly and do not add explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = messages[0]["content"] + "\n\n" + messages[1]["content"]

        inputs = self.tokenizer([text], return_tensors="pt").to(self.input_device)
        do_sample = float(temperature) > 0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=float(temperature), top_p=0.95)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if "Judgement:" in prompt:
            match = re.search(r"\b[01]\b", decoded)
            if match:
                return match.group(0)
        return decoded


def score_mathvision(result_file: Path) -> None:
    from vlmeval.dataset.utils.mathv import MATH_V_acc, MATH_V_auxeval, post_check

    storage = result_file.with_name(result_file.stem + "_qwen7judge.xlsx")
    tmp_file = result_file.with_name(result_file.stem + "_qwen7judge.pkl")
    score_file = result_file.with_name(result_file.stem + "_qwen7judge_score.csv")

    print(f"[score] dataset=MathVision result_file={result_file}")
    print(f"[score] storage={storage}")
    print(f"[score] qwen_judge_model={os.environ.get('MATH_QWEN_JUDGE_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')}")

    data = load(str(result_file)).sort_values(by="index")
    ans = load(str(tmp_file)) if tmp_file.exists() else {}
    judge = None

    for pos in tqdm(range(len(data)), desc="MathVision Qwen7 extraction"):
        line = data.iloc[pos]
        idx = line["index"]
        if idx in ans:
            continue
        prefetched = post_check(line, prefetch=True)
        if prefetched:
            ans[idx] = {"log": "Prefetch succeed", "res": prefetched}
        else:
            if judge is None:
                judge = LocalQwenJudge()
            ans[idx] = MATH_V_auxeval(judge, line)
        if len(ans) % 20 == 0:
            dump(ans, str(tmp_file))

    dump(ans, str(tmp_file))
    data["res"] = [ans[idx]["res"] for idx in data["index"]]
    data["log"] = [ans[idx]["log"] for idx in data["index"]]
    dump(data, str(storage))
    score = MATH_V_acc(str(storage))
    dump(score, str(score_file))
    print("[score] MathVision Qwen7 judge score:")
    print(score)


def score_mathverse(result_file: Path) -> None:
    from vlmeval.dataset.utils.mathverse import MathVerse_acc, MathVerse_auxeval_extract, MathVerse_auxeval_score

    storage_extract = result_file.with_name(result_file.stem + "_qwen7judge_extract.xlsx")
    tmp_file_extract = result_file.with_name(result_file.stem + "_qwen7judge_extract.pkl")
    storage_score = result_file.with_name(result_file.stem + "_qwen7judge_score.xlsx")
    tmp_file_score = result_file.with_name(result_file.stem + "_qwen7judge_score.pkl")
    score_file = result_file.with_name(result_file.stem + "_qwen7judge_score.csv")

    print(f"[score] dataset=MathVerse result_file={result_file}")
    print(f"[score] extract_storage={storage_extract}")
    print(f"[score] score_storage={storage_score}")
    print(f"[score] qwen_judge_model={os.environ.get('MATH_QWEN_JUDGE_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')}")

    if not storage_extract.exists():
        data = load(str(result_file)).sort_values(by="index")
        ans = load(str(tmp_file_extract)) if tmp_file_extract.exists() else {}
        judge = None
        for pos in tqdm(range(len(data)), desc="MathVerse Qwen7 extraction"):
            line = data.iloc[pos]
            idx = line["index"]
            if idx in ans:
                continue
            if judge is None:
                judge = LocalQwenJudge()
            ans[idx] = MathVerse_auxeval_extract(judge, line)
            if len(ans) % 20 == 0:
                dump(ans, str(tmp_file_extract))
        dump(ans, str(tmp_file_extract))
        data["extract"] = [ans[idx]["extract"] for idx in data["index"]]
        data["log_extract"] = [ans[idx]["log_extract"] for idx in data["index"]]
        dump(data, str(storage_extract))

    if not storage_score.exists():
        data = load(str(storage_extract)).sort_values(by="index")
        ans = load(str(tmp_file_score)) if tmp_file_score.exists() else {}
        judge = None
        for pos in tqdm(range(len(data)), desc="MathVerse Qwen7 scoring"):
            line = data.iloc[pos]
            idx = line["index"]
            if idx in ans:
                continue
            if judge is None:
                judge = LocalQwenJudge()
            ans[idx] = MathVerse_auxeval_score(judge, line)
            if len(ans) % 20 == 0:
                dump(ans, str(tmp_file_score))
        dump(ans, str(tmp_file_score))
        data["score"] = [ans[idx]["score"] for idx in data["index"]]
        data["log_score"] = [ans[idx]["log_score"] for idx in data["index"]]
        dump(data, str(storage_score))

    score = MathVerse_acc(str(storage_score))
    dump(score, str(score_file))
    print("[score] MathVerse Qwen7 judge score:")
    print(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["MathVerse_MINI_Vision_Only", "MathVision_MINI", "MathVision"])
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    result_file = Path(args.result_file).resolve()
    if not result_file.is_file():
        raise FileNotFoundError(result_file)

    if args.dataset == "MathVerse_MINI_Vision_Only":
        score_mathverse(result_file)
    elif args.dataset in ("MathVision_MINI", "MathVision"):
        score_mathvision(result_file)
    else:
        raise ValueError(args.dataset)


if __name__ == "__main__":
    main()
