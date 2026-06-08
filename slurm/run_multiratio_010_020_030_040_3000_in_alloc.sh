#!/bin/bash
# Run a 4-GPU OPSD multi-ratio training job inside an existing 4xL40S allocation.
set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_3000
TRAIN_DIR="${OUT_ROOT}/divprune_kl_multiratio_010_020_030_040"
LOG_DIR=/scratch/enmingzz/temp/opsd_logs

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

archive_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    local archived="${path}_restart_$(date +%Y%m%d_%H%M%S)"
    echo "Archiving existing ${path} to ${archived}"
    mv "${path}" "${archived}"
  fi
}

archive_if_exists "${TRAIN_DIR}"

echo "Starting OPSD multi-ratio 0.10/0.20/0.30/0.40 training at $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  opsd/scripts/train_qwen25vl_prune_distill.py \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_single_r025.yaml \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /scratch/xxluo/mscoco/train2017 \
  --output_dir "${TRAIN_DIR}" \
  --keep_ratios 0.10,0.20,0.30,0.40 \
  --ratio_sampling_probs 0.25,0.25,0.25,0.25 \
  --max_steps 3000 \
  --save_every 500 \
  --eval_every 500 \
  --gradient_accumulation_steps 4 \
  --attn_implementation flash_attention_2

echo "OPSD multi-ratio training complete at $(date)"
