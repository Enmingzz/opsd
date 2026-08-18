# LLaVA-CoT 20k LLM-only retraining (2026-08-03)

This directory contains the clean 20k SFT and progressive OPSD retraining runs.
It does not replace or modify prior checkpoints or evaluation outputs.

## Shared contract

- Base: `Qwen/Qwen2.5-VL-7B-Instruct`
- Data: the verified, decontaminated, exact-order 20k LLaVA-CoT JSONL
- Image budget: `min_pixels=max_pixels=1080*28*28`
- Target cap: 512 Qwen tokens in the verified dataset
- Scope: LLM decoder LoRA only, 392 tensors / 40,370,176 parameters
- LoRA: rank 16, alpha 32, dropout 0
- Optimizer: AdamW, fixed learning rate `2e-5`, no weight decay
- Effective batch size: `4 GPUs * 1 sample * 8 accumulation = 32`
- Reasoning target format and clean Armen-aligned environment match the controlled runs

## Progressive OPSD

The 20,000-sample horizon is optimizer-aligned:

| Global indices | Retention | Samples |
|---|---:|---:|
| 0-4,991 | 40% | 4,992 |
| 4,992-9,983 | 30% | 4,992 |
| 9,984-14,975 | 20% | 4,992 |
| 14,976-19,999 | 10% | 5,024 |

The run uses the original forward-KL OPSD objective, EMA teacher decay 0.9999,
no ground-truth access for the teacher, and KV-cached greedy student rollouts.
It is split at the optimizer boundary 9,984 because a full 20k run exceeds the
24-hour Slurm limit. Stage 2 restores LoRA, optimizer, EMA, and all rank RNG
states from stage 1. Both configs retain `training.max_steps=20000`, so the
curriculum horizon is not compressed at the segment boundary.

## SFT

SFT starts from the same base model and uses all 20k examples. Ratios are
deterministically paired per sample with the original random OPSD namespace;
the first 9,984 assignments are checked against the original OPSD logs.

## Validation and launch

```bash
python experiments/llm_only/lcot20k_retrain_dropout0_20260803/validate_configs.py
```

The launch order is a combined 32-sample, 4-GPU smoke test, then independent
SFT and progressive stage-1 jobs, followed by progressive stage 2 with an
`afterok` dependency. Every completed segment is audited for data order,
ratio assignment, finite loss, fixed LR, optimizer step, LoRA scope, EMA
state, and checkpoint completeness.

Estimated storage is about 47 GB total: evaluation snapshots are about 155 MB
each, progressive resumable checkpoints about 617 MB each, and SFT resumable
checkpoints about 463 MB each.
