#!/bin/bash
# Print the OPSD training setup used in the current divprune_lite experiments.
# Default mode is dry-run for slides/demo. Add --run to actually launch training.
set -euo pipefail

RUN=false
if [[ "${1:-}" == "--run" ]]; then
  RUN=true
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash opsd/scripts/demo_opsd_training_params.sh
  bash opsd/scripts/demo_opsd_training_params.sh --run

Default prints the OPSD training parameters and command without starting training.
EOF
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  exit 2
fi

REPO_ROOT=/project/6101803/enmingzz
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-7B-Instruct
TRAIN_JSONL=/scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl
VAL_JSONL=/scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl
IMAGE_ROOT=/scratch/xxluo/mscoco/train2017
OUTPUT_DIR=/scratch/enmingzz/temp/opsd_runs/demo_opsd_divprune_multiratio

KEEP_RATIOS=0.10,0.20,0.30,0.40
RATIO_SAMPLING_PROBS=0.25,0.25,0.25,0.25

cat <<EOF
OPSD Qwen2.5-VL training setup
==============================

Model:
  base model              ${MODEL_NAME_OR_PATH}
  student adaptation      PEFT LoRA adapter
  teacher                 same base model, full visual tokens, frozen/no_grad
  precision               bf16
  attention               flash_attention_2, fallback handled by training script

Pruning:
  pruner                  divprune_lite
  location                after Qwen2.5-VL vision encoder, before LLM
  student input mode      drop_tokens
  training retention      ${KEEP_RATIOS}
  sampling probability    ${RATIO_SAMPLING_PROBS}
  divprune grid floor     true, grid_size=4
  vscan_merge_dropped     false

Distillation:
  distill mode            teacher_rollout
  loss                    kl_only
  KL direction            KL(p_teacher || p_student)
  temperature             2.0
  kd_topk                 0, full-vocab KL
  CE baseline             disabled
  gold_prefix_debug       not used
  max_new_tokens          32

LoRA:
  r                       16
  alpha                   32
  dropout                 0.05
  target modules          q/k/v/o + gate/up/down projections

Optimization:
  lr                      2e-5
  weight_decay            0.0
  grad_accum_steps        16
  max_steps               3000
  save_every              500
  eval_every              500
  log_every               10
  seed                    42

Data:
  train_jsonl             ${TRAIN_JSONL}
  val_jsonl               ${VAL_JSONL}
  image_root              ${IMAGE_ROOT}
  output_dir              ${OUTPUT_DIR}
EOF

CMD=(
  python opsd/scripts/train_qwen25vl_prune_distill.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --train_jsonl "${TRAIN_JSONL}"
  --val_jsonl "${VAL_JSONL}"
  --image_root "${IMAGE_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --student_input_mode drop_tokens
  --pruner divprune_lite
  --divprune_grid_floor true
  --divprune_grid_size 4
  --divprune_chunk_size 8192
  --keep_ratios "${KEEP_RATIOS}"
  --ratio_sampling_probs "${RATIO_SAMPLING_PROBS}"
  --sample_budget_each_step true
  --distill_mode teacher_rollout
  --loss kl_only
  --enable_ce_baseline false
  --ce_alpha 0.0
  --kd_alpha 1.0
  --temperature 2.0
  --kd_topk 0
  --max_new_tokens 32
  --teacher_confidence_weighting true
  --filter_teacher_wrong false
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --lr 2e-5
  --weight_decay 0.0
  --gradient_accumulation_steps 16
  --max_steps 3000
  --save_every 500
  --eval_every 500
  --log_every 10
  --bf16 true
  --attn_implementation flash_attention_2
  --seed 42
)

echo
echo "Training command"
echo "================"
printf 'cd %q\n' "${REPO_ROOT}"
printf 'source %q\n' "/project/6101803/enmingzz/env/vsi-official.sh"
echo "export HF_HOME=/scratch/enmingzz/hf_cache"
echo "export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache"
echo "export TOKENIZERS_PARALLELISM=false"
printf '%q ' "${CMD[@]}"
echo

if [[ "${RUN}" == "true" ]]; then
  cd "${REPO_ROOT}"
  source /project/6101803/enmingzz/env/vsi-official.sh
  export HF_HOME=/scratch/enmingzz/hf_cache
  export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
  export TOKENIZERS_PARALLELISM=false
  "${CMD[@]}"
else
  echo
  echo "Dry-run only. Add --run to start training."
fi
