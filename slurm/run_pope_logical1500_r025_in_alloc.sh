#!/bin/bash
# Evaluate logical step_1500 OPSD on POPE at divprune_lite retention 25%.
set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false

OUT_DIR=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_1500/pope_r025
ADAPTER=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_1500/divprune_kl_multiratio_010_020_030_040_resume_from_step1000_to_step1500/step_500
EVAL_JSONL=/scratch/enmingzz/temp/opsd_data/pope_full.jsonl
IMAGE_ROOT=/scratch/enmingzz/temp/opsd_data/pope_images

if [[ ! -d "${ADAPTER}" ]]; then
  echo "Missing adapter: ${ADAPTER}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_JSONL}" ]]; then
  echo "Missing POPE jsonl: ${EVAL_JSONL}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/shards"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local out="${OUT_DIR}/shards/eval_pope_r025_shard${shard}.jsonl"
  local log="${OUT_DIR}/shards/eval_pope_r025_shard${shard}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python opsd/scripts/eval_qwen25vl_pruned_student.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_path "${ADAPTER}" \
    --eval_jsonl "${EVAL_JSONL}" \
    --image_root "${IMAGE_ROOT}" \
    --output_jsonl "${out}" \
    --keep_ratio 0.25 \
    --pruner divprune_lite \
    --student_input_mode drop_tokens \
    --max_new_tokens 8 \
    --num_shards 4 \
    --shard_index "${shard}" \
    --skip_full_token_base true \
    --attn_implementation flash_attention_2 \
    > "${log}" 2>&1
}

echo "Starting POPE r=0.25 logical step_1500 eval at $(date)"
run_shard 0 0 &
pid0=$!
run_shard 1 1 &
pid1=$!
run_shard 2 2 &
pid2=$!
run_shard 3 3 &
pid3=$!
wait "${pid0}"
wait "${pid1}"
wait "${pid2}"
wait "${pid3}"

cat \
  "${OUT_DIR}/shards/eval_pope_r025_shard0.jsonl" \
  "${OUT_DIR}/shards/eval_pope_r025_shard1.jsonl" \
  "${OUT_DIR}/shards/eval_pope_r025_shard2.jsonl" \
  "${OUT_DIR}/shards/eval_pope_r025_shard3.jsonl" \
  > "${OUT_DIR}/eval_pope_r025_full.jsonl"

python opsd/scripts/score_pope_eval.py \
  --eval_jsonl "${OUT_DIR}/eval_pope_r025_full.jsonl" \
  --output_dir "${OUT_DIR}/score"

echo "POPE r=0.25 logical step_1500 eval complete at $(date)"
