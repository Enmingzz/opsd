#!/bin/bash
# Re-run the slow MME middle shard as four smaller shards inside an existing 4xL40S allocation.
set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false

OUT_DIR=/scratch/enmingzz/temp/opsd_runs/real_single_r025/mme
EVAL_JSONL=/scratch/enmingzz/temp/opsd_data/mme_test.jsonl
IMAGE_ROOT=/scratch/enmingzz/temp/opsd_data/mme_images
ADAPTER=/scratch/enmingzz/temp/opsd_runs/real_single_r025/divprune_kl_single_r025_ddp4/step_1000
mkdir -p "${OUT_DIR}/shards"
rm -f "${OUT_DIR}/shards"/eval_step1000_r025_mme_shard2*.jsonl

run_subshard() {
  local name="$1"
  local gpu="$2"
  local start="$3"
  local limit="$4"
  local out="${OUT_DIR}/shards/eval_step1000_r025_mme_${name}.jsonl"
  local log="${OUT_DIR}/shards/eval_step1000_r025_mme_${name}.log"
  echo "Starting ${name}: start=${start} limit=${limit} gpu=${gpu} at $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" python opsd/scripts/eval_qwen25vl_pruned_student.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_path "${ADAPTER}" \
    --eval_jsonl "${EVAL_JSONL}" \
    --image_root "${IMAGE_ROOT}" \
    --output_jsonl "${out}" \
    --keep_ratio 0.25 \
    --pruner divprune_lite \
    --student_input_mode drop_tokens \
    --max_new_tokens 16 \
    --start_index "${start}" \
    --limit "${limit}" \
    --attn_implementation flash_attention_2 \
    > "${log}" 2>&1
  echo "Finished ${name} at $(date)"
}

run_subshard shard2a 0 1188 148 &
pid0=$!
run_subshard shard2b 1 1336 148 &
pid1=$!
run_subshard shard2c 2 1484 148 &
pid2=$!
run_subshard shard2d 3 1632 150 &
pid3=$!

wait "${pid0}"
wait "${pid1}"
wait "${pid2}"
wait "${pid3}"

cat \
  "${OUT_DIR}/shards/eval_step1000_r025_mme_shard0.jsonl" \
  "${OUT_DIR}/shards/eval_step1000_r025_mme_shard1.jsonl" \
  "${OUT_DIR}/shards/eval_step1000_r025_mme_shard2a.jsonl" \
  "${OUT_DIR}/shards/eval_step1000_r025_mme_shard2b.jsonl" \
  "${OUT_DIR}/shards/eval_step1000_r025_mme_shard2c.jsonl" \
  "${OUT_DIR}/shards/eval_step1000_r025_mme_shard2d.jsonl" \
  "${OUT_DIR}/shards/eval_step1000_r025_mme_shard3.jsonl" \
  > "${OUT_DIR}/eval_step1000_r025_mme_full.jsonl"

python opsd/scripts/score_mme_eval.py \
  --eval_jsonl "${OUT_DIR}/eval_step1000_r025_mme_full.jsonl" \
  --output_dir "${OUT_DIR}/score_step1000_r025_full"

echo "Recovered MME full eval complete at $(date)"
