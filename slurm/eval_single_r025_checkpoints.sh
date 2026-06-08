#!/bin/bash
# Adjust --account/--gres if your cluster uses different names.
#SBATCH --job-name=opsd_r025_eval
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/enmingzz/temp/opsd_logs/%x_%j.out

set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false

OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/real_single_r025
TRAIN_DIR="${OUT_ROOT}/divprune_kl_single_r025"
EVAL_DIR="${OUT_ROOT}/evals"
mkdir -p "${EVAL_DIR}" /scratch/enmingzz/temp/opsd_logs

for step in 250 500 750 1000; do
  CKPT="${TRAIN_DIR}/step_${step}"
  if [[ ! -d "${CKPT}" ]]; then
    echo "Missing checkpoint: ${CKPT}" >&2
    exit 2
  fi
  python opsd/scripts/eval_qwen25vl_pruned_student.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_path "${CKPT}" \
    --eval_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
    --image_root /scratch/xxluo/mscoco/train2017 \
    --output_jsonl "${EVAL_DIR}/eval_step${step}_r025_limit200.jsonl" \
    --keep_ratio 0.25 \
    --pruner divprune_lite \
    --student_input_mode drop_tokens \
    --max_new_tokens 32 \
    --limit 200 \
    --attn_implementation flash_attention_2
done
