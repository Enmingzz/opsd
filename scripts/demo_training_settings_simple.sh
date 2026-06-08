#!/bin/bash
# Compact OPSD training settings for demo slides.
set -euo pipefail

cat <<'EOF'
OPSD Training Settings
======================

Base VLM: Qwen2.5-VL-7B-Instruct
Teacher: full-token Qwen2.5-VL, frozen
Student: same Qwen2.5-VL + LoRA adapter

Pruner: DivPrune-Lite
Training retention ratios: 10%, 20%, 30%, 40%
Ratio sampling: uniform, 25% each
Student input: drop visual tokens before LLM

Loss: pure KL distillation only
CE loss: disabled
Distillation mode: teacher rollout
Temperature: 2.0
Max generated tokens: 32

LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Target modules: q/k/v/o projections + MLP up/down/gate

GPU: 4 x NVIDIA L40S
Per-GPU micro batch size: 1
Gradient accumulation steps: 4
Effective global batch size: 16

Training steps: 3000
Checkpoint interval: every 500 steps
Learning rate: 2e-5
Weight decay: 0.0
Precision: bf16
Attention: FlashAttention-2

Training data: LLaVA-Instruct subset, 20K samples
Validation data: LLaVA-Instruct subset, 1K samples
Evaluation ratios: 5%, 15%, 25%, 35%, 45%
EOF
