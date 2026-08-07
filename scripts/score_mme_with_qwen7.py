from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from vlmeval.dataset.utils.yorn import MME_rating, YOrN_Extraction, YOrN_auxeval
from vlmeval.smp import dump, load


class LocalQwenYesNoJudge:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = os.environ.get("MME_QWEN_JUDGE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        self.max_new_tokens = int(os.environ.get("MME_QWEN_JUDGE_MAX_NEW_TOKENS", "16"))
        device = os.environ.get("MME_QWEN_JUDGE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        dtype_name = os.environ.get("MME_QWEN_JUDGE_DTYPE", "bfloat16" if torch.cuda.is_available() else "float32")
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
                "content": "You are a strict Yes/No answer normalizer. Return only Yes, No, or Unknown.",
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
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_file(result_file: Path, judge_holder: dict[str, LocalQwenYesNoJudge | None]) -> None:
    storage = result_file.with_name(result_file.stem + "_qwen7judge_auxmatch.xlsx")
    tmp_file = result_file.with_name(result_file.stem + "_qwen7judge_tmp.pkl")
    score_file = result_file.with_name(result_file.stem + "_qwen7judge_score.csv")

    print(f"[score] result_file={result_file}")
    print(f"[score] storage={storage}")
    print(f"[score] score_file={score_file}")
    print(f"[score] qwen_judge_model={os.environ.get('MME_QWEN_JUDGE_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')}")

    data = load(str(result_file)).sort_values(by="index")
    data["prediction"] = [str(x) for x in data["prediction"]]
    ans_map = {idx: YOrN_Extraction(pred) for idx, pred in zip(data["index"], data["prediction"])}
    initial_unknown = sum(v == "Unknown" for v in ans_map.values())

    cached = load(str(tmp_file)) if tmp_file.exists() else {}
    for idx, value in cached.items():
        if ans_map[idx] == "Unknown" and value != "Unknown":
            ans_map[idx] = value

    unknown_rows = [data.iloc[i] for i in range(len(data)) if ans_map[data.iloc[i]["index"]] == "Unknown"]
    for line in tqdm(unknown_rows, desc=f"MME Qwen7 yes/no {result_file.parent.parent.name}"):
        idx = line["index"]
        if idx in cached:
            ans_map[idx] = cached[idx]
            continue
        if judge_holder["judge"] is None:
            judge_holder["judge"] = LocalQwenYesNoJudge()
        ans = YOrN_auxeval(judge_holder["judge"], line)
        cached[idx] = ans
        ans_map[idx] = ans
        if len(cached) % 20 == 0:
            dump(cached, str(tmp_file))

    dump(cached, str(tmp_file))
    data["extracted"] = [ans_map[idx] for idx in data["index"]]
    data["score"] = data["answer"] == data["extracted"]
    dump(data, str(storage))

    score = MME_rating(str(storage))
    dump(score, str(score_file))

    final_unknown = int((data["extracted"] == "Unknown").sum())
    fixed = initial_unknown - final_unknown
    print(f"[score] unknown_before={initial_unknown} unknown_after={final_unknown} fixed={fixed}")
    print("[score] MME Qwen7 postprocessed score:")
    print(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", required=True, nargs="+")
    args = parser.parse_args()

    judge_holder: dict[str, LocalQwenYesNoJudge | None] = {"judge": None}
    for raw_result_file in args.result_file:
        result_file = Path(raw_result_file).resolve()
        if not result_file.is_file():
            raise FileNotFoundError(result_file)
        score_file(result_file, judge_holder)


if __name__ == "__main__":
    main()
