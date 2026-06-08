#!/bin/bash
# Continue OPSD with true on-policy KL from the logical step_1500 teacher-rollout adapter.
set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/scratch/enmingzz/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}
BASE_ADAPTER=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_1500/divprune_kl_multiratio_010_020_030_040_resume_from_step1000_to_step1500/step_500
OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_onpolicy_from_logical1500
TRAIN_DIR="${OUT_ROOT}/divprune_kl_onpolicy_010_020_030_040_from_logical1500"

mkdir -p "${OUT_ROOT}" /scratch/enmingzz/temp/opsd_logs

if [[ ! -d "${BASE_ADAPTER}" ]]; then
  echo "Missing base adapter checkpoint: ${BASE_ADAPTER}" >&2
  exit 2
fi

if [[ -e "${TRAIN_DIR}" ]]; then
  archived="${TRAIN_DIR}_restart_$(date +%Y%m%d_%H%M%S)"
  echo "Archiving existing ${TRAIN_DIR} to ${archived}"
  mv "${TRAIN_DIR}" "${archived}"
fi

echo "Starting true on-policy OPSD continuation at $(date)"
echo "Base adapter: ${BASE_ADAPTER}"
echo "Output dir: ${TRAIN_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  opsd/scripts/train_qwen25vl_prune_distill.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_single_r025.yaml \
  --adapter_path "${BASE_ADAPTER}" \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /scratch/xxluo/mscoco/train2017 \
  --output_dir "${TRAIN_DIR}" \
  --keep_ratios 0.10,0.20,0.30,0.40 \
  --ratio_sampling_probs 0.25,0.25,0.25,0.25 \
  --distill_mode on_policy \
  --loss kl_only \
  --enable_ce_baseline false \
  --ce_alpha 0.0 \
  --kd_alpha 1.0 \
  --max_steps 1000 \
  --save_every 500 \
  --eval_every 500 \
  --gradient_accumulation_steps 4 \
  --max_new_tokens 32 \
  --attn_implementation flash_attention_2 \
  --ddp_timeout_minutes 60

echo "True on-policy OPSD continuation complete at $(date)"
echo "Adapter checkpoints are under: ${TRAIN_DIR}"
