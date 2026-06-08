#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opsd.visionzip_aokvqa.aokvqa import FormattedAOKVQASample, normalize_aokvqa_sample, resolve_image
from opsd.visionzip_aokvqa.prompting import format_chat_messages, parse_final_answer
from opsd.visionzip_aokvqa.qwen_wrapper import (
    apply_lora,
    encode_prompt_text,
    extract_generated_logits,
    forward_pruned,
    generate_pruned,
    load_qwen_model_and_processor,
    primary_device,
)


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")
CONFIG_ROOT = ROOT / "configs" / "experiments"


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TRL GRPO training with VisionZip-pruned Qwen2.5-VL rollouts/forwards.")
    p.add_argument("--config", default=str(CONFIG_ROOT / "aokvqa_visionzip_grpo_trl_512.yaml"))
    p.add_argument("--output_dir", default="")
    p.add_argument("--selected_ids_path", default="")
    p.add_argument("--adapter_path", default="")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow_embedding_fallback", action="store_true")
    p.add_argument("--resume_from_checkpoint", default="")
    p.add_argument("--check_env", action="store_true")
    return p


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg.setdefault("output_dir", str(OUTPUT_ROOT / "checkpoints" / "grpo_trl_visionzip"))
    if args.max_steps is not None:
        set_nested(cfg, "training.max_steps", int(args.max_steps))
    if args.limit is not None:
        set_nested(cfg, "dataset.limit", int(args.limit))
    if args.smoke:
        set_nested(cfg, "training.max_steps", 1)
        set_nested(cfg, "dataset.limit", 16)
        set_nested(cfg, "training.logging_steps", 1)
        set_nested(cfg, "training.save_steps", 1)
        cfg["smoke"] = True
    if args.allow_embedding_fallback:
        set_nested(cfg, "pruning.allow_embedding_fallback", True)
    if args.adapter_path:
        set_nested(cfg, "training.adapter_path", args.adapter_path)
    if args.selected_ids_path:
        set_nested(cfg, "dataset.selected_ids_path", args.selected_ids_path)
    if args.resume_from_checkpoint:
        set_nested(cfg, "training.resume_from_checkpoint", args.resume_from_checkpoint)
    return cfg


def apply_selected_ids(dataset: list[FormattedAOKVQASample], ids_path: str | Path | None) -> list[FormattedAOKVQASample]:
    if not ids_path:
        return dataset
    wanted = [str(x) for x in json.loads(Path(ids_path).read_text(encoding="utf-8"))]
    by_id = {sample.sample_id: sample for sample in dataset}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise KeyError(f"Selected training ids missing from dataset: {missing[:10]}")
    return [by_id[sample_id] for sample_id in wanted]


def sample_retention_ratio(cfg: dict[str, Any], rng: random.Random) -> float:
    ratios = [float(x) for x in get_nested(cfg, "pruning.train_retention_ratios", [0.05, 0.1, 0.2, 0.3])]
    weights_raw = get_nested(cfg, "pruning.train_retention_ratio_weights", None)
    if weights_raw is None:
        return float(rng.choice(ratios))
    weights = [float(x) for x in weights_raw]
    if len(weights) != len(ratios):
        raise ValueError("pruning.train_retention_ratio_weights must match pruning.train_retention_ratios.")
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("pruning.train_retention_ratio_weights must be non-negative and sum to a positive value.")
    return float(rng.choices(ratios, weights=weights, k=1)[0])


class LazyAOKVQAGRPOTrainDataset(torch.utils.data.Dataset):
    def __init__(self, hf_splits: list[Any], rows: list[dict[str, Any]], image_root: str = "") -> None:
        self.hf_splits = hf_splits
        self.rows = rows
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        split_index = int(row.pop("_split_index"))
        row_index = int(row.pop("_row_index"))
        image_value = self.hf_splits[split_index][row_index]["image"]
        row["image"] = resolve_image(image_value, image_root=self.image_root)
        return row


def build_dataset(cfg: dict[str, Any]):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("TRL GRPO training requires the `datasets` package.") from exc

    selected_ids_path = get_nested(cfg, "dataset.selected_ids_path", "")
    wanted_ids: list[str] = []
    wanted_set: set[str] | None = None
    if selected_ids_path:
        wanted_ids = [str(x) for x in json.loads(Path(selected_ids_path).read_text(encoding="utf-8"))]
        wanted_set = set(wanted_ids)

    dataset_name = get_nested(cfg, "dataset.name", "HuggingFaceM4/A-OKVQA")
    image_root = get_nested(cfg, "dataset.image_root", "")
    hf_splits: list[Any] = []
    rows: list[dict[str, Any]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    rng = random.Random(int(get_nested(cfg, "training.seed", 42)))

    for split in list(get_nested(cfg, "dataset.use_splits", ["train", "validation"])):
        ds = load_dataset(dataset_name, split=split)
        split_index = len(hf_splits)
        hf_splits.append(ds)
        metadata_ds = ds.remove_columns([name for name in ("image",) if name in ds.column_names])
        for idx, record in enumerate(metadata_ds):
            meta = dict(record)
            meta["image"] = {"split_index": split_index, "row_index": idx}
            sample = normalize_aokvqa_sample(meta, index=len(rows))
            if wanted_set is not None and sample.sample_id not in wanted_set:
                continue
            row = {
                "prompt": format_chat_messages(sample.prompt),
                "prompt_text": sample.prompt,
                "sample_id": sample.sample_id,
                "question": sample.question,
                "options": list(sample.options),
                "correct_index": int(sample.correct_index),
                "correct_letter": sample.correct_letter,
                "reasoning": sample.reasoning,
                "retention_ratio": sample_retention_ratio(cfg, rng),
                "_split_index": split_index,
                "_row_index": idx,
            }
            rows.append(row)
            rows_by_id[sample.sample_id] = row

    if wanted_ids:
        missing = [sample_id for sample_id in wanted_ids if sample_id not in rows_by_id]
        if missing:
            raise KeyError(f"Selected training ids missing from dataset: {missing[:10]}")
        rows = [rows_by_id[sample_id] for sample_id in wanted_ids]
    else:
        rng.shuffle(rows)
        limit = int(get_nested(cfg, "dataset.limit", 0) or 0)
        if limit and limit > 0:
            rows = rows[:limit]

    if not rows:
        raise ValueError("Training dataset is empty.")
    return LazyAOKVQAGRPOTrainDataset(hf_splits=hf_splits, rows=rows, image_root=image_root)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(completion or "")


def reward_correct_answer(completions, correct_letter=None, **_: Any) -> list[float]:
    letters = correct_letter or []
    return [1.0 if parse_final_answer(completion_to_text(c)) == str(gold).strip().upper() else 0.0 for c, gold in zip(completions, letters)]


def reward_final_answer_format(completions, **_: Any) -> list[float]:
    return [1.0 if parse_final_answer(completion_to_text(c)) is not None else 0.0 for c in completions]


def require_trl():
    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "TRL is required for official GRPO. Install it in an isolated environment, e.g. `pip install trl==0.24.0`, "
            f"then rerun this entrypoint. Original import error: {exc!r}"
        ) from exc
    return GRPOConfig, GRPOTrainer


def check_environment() -> int:
    import importlib.util

    checks: dict[str, Any] = {
        "python": sys.executable,
        "trl": bool(importlib.util.find_spec("trl")),
        "datasets": bool(importlib.util.find_spec("datasets")),
        "transformers": bool(importlib.util.find_spec("transformers")),
        "peft": bool(importlib.util.find_spec("peft")),
        "accelerate": bool(importlib.util.find_spec("accelerate")),
    }
    if checks["trl"]:
        try:
            _, trainer_cls = require_trl()
            checks["trl_version"] = __import__("trl").__version__
            checks["grpo_generate_signature"] = str(inspect.signature(trainer_cls._generate))
            checks["has_prunable_logprob_hook"] = hasattr(trainer_cls, "_get_per_token_logps_and_entropies")
        except Exception as exc:
            checks["trl_error"] = repr(exc)
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if checks["trl"] and checks.get("has_prunable_logprob_hook") else 1


def build_visionzip_trainer_class(base_cls):
    generate_param_count = len(inspect.signature(base_cls._generate).parameters)
    legacy_generate_api = generate_param_count >= 3

    class VisionZipGRPOTrainer(base_cls):
        def __init__(self, *args: Any, visionzip_cfg: dict[str, Any], **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.visionzip_cfg = visionzip_cfg
            self._visionzip_generation_rows: list[dict[str, Any]] = []
            self._visionzip_last_generate_state: dict[str, Any] = {}
            self._visionzip_active_rows: torch.Tensor | None = None
            self._visionzip_active_ratios: torch.Tensor | None = None

        def _generate_and_score_completions(self, inputs):
            self._visionzip_generation_rows = list(inputs)
            output = super()._generate_and_score_completions(inputs)
            state = self._visionzip_last_generate_state
            device = output["prompt_ids"].device
            if state:
                output["visionzip_row_index"] = torch.tensor(state["row_indices"], device=device, dtype=torch.long)
                output["retention_ratio"] = torch.tensor(state["retention_ratios"], device=device, dtype=torch.float32)
            for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes", "token_type_ids", "num_images"):
                output.pop(key, None)
            return output

        def _generate(self, prompts, images=None):
            if not self._visionzip_generation_rows:
                raise RuntimeError("VisionZip GRPO generation rows were not initialized.")
            device = self.accelerator.device
            mode = "train" if self.model.training else "eval"
            model = self.accelerator.unwrap_model(self.model)

            prompt_ids: list[list[int]] = []
            completion_ids: list[list[int]] = []
            retention_ratios: list[float] = []
            row_indices: list[int] = []
            texts: list[str] = []
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    for row_index, row in enumerate(self._visionzip_generation_rows[: len(prompts)]):
                        ratio = float(row.get("retention_ratio", 0.1))
                        sample = self._sample_from_row(row)
                        prompt_inputs = encode_prompt_text(
                            self.processing_class,
                            sample,
                            str(row["prompt_text"]),
                            image_root=get_nested(self.visionzip_cfg, "dataset.image_root", ""),
                            device=device,
                        )
                        gen_ids, text, _meta = generate_pruned(
                            model,
                            self.processing_class,
                            prompt_inputs,
                            ratio,
                            max_new_tokens=int(get_nested(self.visionzip_cfg, "generation.max_new_tokens", self.max_completion_length)),
                            do_sample=True,
                            temperature=float(getattr(self, "temperature", get_nested(self.visionzip_cfg, "generation.temperature", 1.2))),
                            top_p=float(getattr(self, "top_p", get_nested(self.visionzip_cfg, "generation.top_p", 1.0))),
                            allow_embedding_fallback=bool(get_nested(self.visionzip_cfg, "pruning.allow_embedding_fallback", False)),
                            manual_decode=bool(get_nested(self.visionzip_cfg, "generation.manual_pruned_generate", True)),
                        )
                        ids = gen_ids.reshape(-1).detach().cpu().tolist()
                        if not ids:
                            ids = [int(getattr(self, "eos_token_id", self.processing_class.tokenizer.eos_token_id))]
                        prompt_ids.append(prompt_inputs["input_ids"][0].detach().cpu().tolist())
                        completion_ids.append([int(x) for x in ids])
                        retention_ratios.append(ratio)
                        row_indices.append(row_index)
                        texts.append(text)
            finally:
                if was_training:
                    model.train()

            self._visionzip_last_generate_state = {
                "retention_ratios": retention_ratios,
                "row_indices": row_indices,
            }
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
            prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
            total_completion_tokens = self.accelerator.gather(completion_lengths).sum()
            if mode == "train":
                self.state.num_input_tokens_seen += (self.accelerator.gather(prompt_lengths).sum() + total_completion_tokens).item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
            self._metrics[mode]["completions/mean_length"].append(self.accelerator.gather(completion_lengths).float().mean().item())
            self._metrics[mode]["completions/min_length"].append(self.accelerator.gather(completion_lengths).float().min().item())
            self._metrics[mode]["completions/max_length"].append(self.accelerator.gather(completion_lengths).float().max().item())
            logprobs = None
            if legacy_generate_api:
                return prompt_ids, completion_ids, total_completion_tokens, logprobs, {}
            completions = [[{"role": "assistant", "content": text}] for text in texts]
            return prompt_ids, completion_ids, None, completions, total_completion_tokens, logprobs, {}, None, []

        def _compute_loss(self, model, inputs):
            self._visionzip_active_rows = inputs.get("visionzip_row_index")
            self._visionzip_active_ratios = inputs.get("retention_ratio")
            try:
                return super()._compute_loss(model, inputs)
            finally:
                self._visionzip_active_rows = None
                self._visionzip_active_ratios = None

        def _get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=None,
            compute_entropy=False,
            **_: Any,
        ):
            all_logps: list[torch.Tensor] = []
            all_entropies: list[torch.Tensor] = []
            for row_pos in range(int(input_ids.size(0))):
                row_index = self._row_index(row_pos)
                ratio = self._retention_ratio(row_pos)
                row = self._visionzip_generation_rows[row_index]
                sample = self._sample_from_row(row)
                row_input_ids = input_ids[row_pos : row_pos + 1]
                row_attention_mask = attention_mask[row_pos : row_pos + 1]
                prompt_len = int(row_input_ids.shape[1]) - int(logits_to_keep)
                prompt_inputs = encode_prompt_text(
                    self.processing_class,
                    sample,
                    str(row["prompt_text"]),
                    image_root=get_nested(self.visionzip_cfg, "dataset.image_root", ""),
                    device=row_input_ids.device,
                )
                full_inputs = dict(prompt_inputs)
                full_inputs["input_ids"] = row_input_ids
                full_inputs["attention_mask"] = row_attention_mask
                if "mm_token_type_ids" in prompt_inputs:
                    full_inputs["mm_token_type_ids"] = self._extend_mm_token_type_ids(
                        prompt_inputs["mm_token_type_ids"],
                        row_input_ids,
                        prompt_len,
                    )
                outputs, pruned = forward_pruned(
                    model,
                    full_inputs,
                    ratio,
                    prompt_len=prompt_len,
                    allow_embedding_fallback=bool(get_nested(self.visionzip_cfg, "pruning.allow_embedding_fallback", False)),
                )
                logits = extract_generated_logits(
                    outputs.logits,
                    int(pruned["metadata"]["student_prompt_len"]),
                    int(logits_to_keep),
                ).unsqueeze(0)
                logits = logits / float(getattr(self, "temperature", 1.0))
                completion_ids = row_input_ids[:, -int(logits_to_keep) :].to(device=logits.device, dtype=torch.long)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                all_logps.append(log_probs.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1))
                if compute_entropy:
                    with torch.no_grad():
                        probs = F.softmax(logits.float(), dim=-1)
                        all_entropies.append(-(probs * log_probs).sum(dim=-1))
            logps = torch.cat(all_logps, dim=0)
            entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
            return logps, entropies

        def _row_index(self, row_pos: int) -> int:
            if self._visionzip_active_rows is not None:
                return int(self._visionzip_active_rows[row_pos].detach().cpu().item())
            state_rows = self._visionzip_last_generate_state.get("row_indices", [])
            return int(state_rows[row_pos])

        def _retention_ratio(self, row_pos: int) -> float:
            if self._visionzip_active_ratios is not None:
                return float(self._visionzip_active_ratios[row_pos].detach().cpu().item())
            state_ratios = self._visionzip_last_generate_state.get("retention_ratios", [])
            return float(state_ratios[row_pos])

        @staticmethod
        def _sample_from_row(row: dict[str, Any]) -> FormattedAOKVQASample:
            return FormattedAOKVQASample(
                sample_id=str(row["sample_id"]),
                image=row["image"],
                question=str(row.get("question", "")),
                options=list(row.get("options", ["", "", "", ""])),
                correct_index=int(row.get("correct_index", 0)),
                correct_letter=str(row.get("correct_letter", "")),
                reasoning=str(row.get("reasoning", "")),
                prompt=str(row["prompt_text"]),
                target="",
                raw={},
            )

        @staticmethod
        def _extend_mm_token_type_ids(prompt_mm: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
            prompt_mm = prompt_mm.to(device=input_ids.device)
            if int(prompt_mm.shape[1]) > prompt_len:
                prompt_mm = prompt_mm[:, -prompt_len:]
            if int(prompt_mm.shape[1]) < prompt_len:
                pad = prompt_mm.new_zeros((1, prompt_len - int(prompt_mm.shape[1])))
                prompt_mm = torch.cat([pad, prompt_mm], dim=1)
            tail = prompt_mm.new_zeros((1, int(input_ids.shape[1]) - prompt_len))
            return torch.cat([prompt_mm, tail], dim=1)

    return VisionZipGRPOTrainer


def make_grpo_config(config_cls, cfg: dict[str, Any]):
    output_dir = str(cfg.get("output_dir", OUTPUT_ROOT / "checkpoints" / "grpo_trl_visionzip"))
    kwargs = {
        "output_dir": output_dir,
        "max_steps": int(get_nested(cfg, "training.max_steps", 1000)),
        "per_device_train_batch_size": int(get_nested(cfg, "training.per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(get_nested(cfg, "training.per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(get_nested(cfg, "training.gradient_accumulation_steps", 8)),
        "learning_rate": float(get_nested(cfg, "training.learning_rate", 1e-6)),
        "weight_decay": float(get_nested(cfg, "training.weight_decay", 0.0)),
        "bf16": bool(get_nested(cfg, "training.bf16", True)),
        "logging_steps": int(get_nested(cfg, "training.logging_steps", 10)),
        "save_steps": int(get_nested(cfg, "training.save_steps", 100)),
        "save_strategy": str(get_nested(cfg, "training.save_strategy", "steps")),
        "report_to": get_nested(cfg, "training.report_to", "none"),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(get_nested(cfg, "training.dataloader_num_workers", 0)),
        "gradient_checkpointing": bool(get_nested(cfg, "training.gradient_checkpointing", True)),
        "num_generations": int(get_nested(cfg, "grpo.num_generations", 8)),
        "generation_batch_size": int(get_nested(cfg, "grpo.generation_batch_size", 8)),
        "max_prompt_length": int(get_nested(cfg, "grpo.max_prompt_length", 2048)),
        "max_completion_length": int(get_nested(cfg, "generation.max_new_tokens", 512)),
        "temperature": float(get_nested(cfg, "generation.temperature", 1.2)),
        "top_p": float(get_nested(cfg, "generation.top_p", 1.0)),
        "beta": float(get_nested(cfg, "grpo.beta", 0.0)),
        "loss_type": str(get_nested(cfg, "grpo.loss_type", "grpo")),
        "scale_rewards": str(get_nested(cfg, "grpo.scale_rewards", "group")),
        "num_iterations": int(get_nested(cfg, "grpo.num_iterations", 2)),
        "reward_weights": list(get_nested(cfg, "grpo.reward_weights", [1.0, 0.1])),
        "use_vllm": bool(get_nested(cfg, "grpo.use_vllm", False)),
        "seed": int(get_nested(cfg, "training.seed", 42)),
    }
    valid = set(inspect.signature(config_cls.__init__).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in valid}
    return config_cls(**filtered)


def train(cfg: dict[str, Any]) -> Path:
    config_cls, trainer_cls = require_trl()
    output_dir = Path(str(cfg.get("output_dir", OUTPUT_ROOT / "checkpoints" / "grpo_trl_visionzip")))
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device_map: Any = None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        stagger = float(os.environ.get("OPSD_GRPO_DDP_STAGGER_LOAD_SECONDS", "0"))
        if stagger > 0:
            time.sleep(float(local_rank) * stagger)
        device_map = {"": local_rank}
    elif get_nested(cfg, "training.device_map", None):
        device_map = get_nested(cfg, "training.device_map")

    model, processor = load_qwen_model_and_processor(
        str(get_nested(cfg, "base_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        bf16=bool(get_nested(cfg, "training.bf16", True)),
        attn_implementation=str(get_nested(cfg, "training.attn_implementation", "flash_attention_2")),
        device_map=device_map,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if bool(get_nested(cfg, "training.use_lora", True)):
        model = apply_lora(
            model,
            r=int(get_nested(cfg, "training.lora_r", 64)),
            alpha=int(get_nested(cfg, "training.lora_alpha", 128)),
            dropout=float(get_nested(cfg, "training.lora_dropout", 0.05)),
            target_modules=list(get_nested(cfg, "training.target_modules", [])) or None,
            adapter_path=str(get_nested(cfg, "training.adapter_path", "")),
        )
    model.to(primary_device(model))

    dataset = build_dataset(cfg)
    args = make_grpo_config(config_cls, cfg)
    trainer = build_visionzip_trainer_class(trainer_cls)(
        model=model,
        reward_funcs=[reward_correct_answer, reward_final_answer_format],
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        visionzip_cfg=cfg,
    )
    resume_from = str(get_nested(cfg, "training.resume_from_checkpoint", "") or "") or None
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(output_dir))
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_env:
        return check_environment()
    cfg = resolve_config(args)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
