from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from vlmeval.dataset.utils.mathvista import MathVista_acc, MathVista_auxeval, post_check
from vlmeval.smp import dump, load


class LocalQwenExtractor:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = os.environ.get(
            "MATH_QWEN_JUDGE_MODEL_PATH",
            os.environ.get("MATHVISTA_QWEN_JUDGE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
        )
        self.max_new_tokens = int(
            os.environ.get(
                "MATH_QWEN_JUDGE_MAX_NEW_TOKENS",
                os.environ.get("MATHVISTA_QWEN_JUDGE_MAX_NEW_TOKENS", "128"),
            )
        )
        device = os.environ.get(
            "MATH_QWEN_JUDGE_DEVICE",
            os.environ.get("MATHVISTA_QWEN_JUDGE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        dtype_name = os.environ.get(
            "MATH_QWEN_JUDGE_DTYPE",
            os.environ.get("MATHVISTA_QWEN_JUDGE_DTYPE", "bfloat16" if torch.cuda.is_available() else "float32"),
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

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You extract the final answer from a model response. "
                    "Return only the shortest extracted answer, with no explanation."
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
        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_mathvista(result_file: Path) -> None:
    storage = result_file.with_name(result_file.stem + "_qwen7judge.xlsx")
    tmp_file = result_file.with_name(result_file.stem + "_qwen7judge.pkl")
    score_file = result_file.with_name(result_file.stem + "_qwen7judge_score.csv")

    print(f"[score] dataset=MathVista_MINI result_file={result_file}")
    print(f"[score] storage={storage}")
    print(
        "[score] qwen_judge_model="
        f"{os.environ.get('MATH_QWEN_JUDGE_MODEL_PATH', os.environ.get('MATHVISTA_QWEN_JUDGE_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct'))}"
    )

    data = load(str(result_file)).sort_values(by="index")
    ans = load(str(tmp_file)) if tmp_file.exists() else {}
    judge = None

    for pos in tqdm(range(len(data)), desc="MathVista Qwen7 extraction"):
        line = data.iloc[pos]
        idx = line["index"]
        if idx in ans:
            continue
        prefetched = post_check(line, prefetch=True)
        if prefetched:
            ans[idx] = {"log": "Prefetch succeed", "res": prefetched}
        else:
            if judge is None:
                judge = LocalQwenExtractor()
            ans[idx] = MathVista_auxeval(judge, line)
        if len(ans) % 20 == 0:
            dump(ans, str(tmp_file))

    dump(ans, str(tmp_file))
    data["res"] = [ans[idx]["res"] for idx in data["index"]]
    data["log"] = [ans[idx]["log"] for idx in data["index"]]
    dump(data, str(storage))
    score = MathVista_acc(str(storage))
    dump(score, str(score_file))
    print("[score] MathVista_MINI Qwen7 judge score:")
    print(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    result_file = Path(args.result_file).resolve()
    if not result_file.is_file():
        raise FileNotFoundError(result_file)
    score_mathvista(result_file)


if __name__ == "__main__":
    main()
