#!/bin/bash
# Resume the 0.10/0.20/0.30/0.40 OPSD run from the step_1000 LoRA adapter.
# The output step_2000 in RESUME_DIR is the logical 3000-step checkpoint.
set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_3000
BASE_TRAIN_DIR="${OUT_ROOT}/divprune_kl_multiratio_010_020_030_040"
BASE_ADAPTER="${BASE_TRAIN_DIR}/step_1000"
RESUME_DIR="${OUT_ROOT}/divprune_kl_multiratio_010_020_030_040_resume_from_step1000_to_step3000"
LOG_DIR=/scratch/enmingzz/temp/opsd_logs

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

if [[ ! -d "${BASE_ADAPTER}" ]]; then
  echo "Missing base adapter checkpoint: ${BASE_ADAPTER}" >&2
  exit 2
fi

if [[ -e "${RESUME_DIR}" ]]; then
  archived="${RESUME_DIR}_restart_$(date +%Y%m%d_%H%M%S)"
  echo "Archiving existing ${RESUME_DIR} to ${archived}"
  mv "${RESUME_DIR}" "${archived}"
fi

echo "Resuming OPSD multi-ratio training from ${BASE_ADAPTER} at $(date)"
echo "Writing resumed run to ${RESUME_DIR}"
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  opsd/scripts/train_qwen25vl_prune_distill.py \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_single_r025.yaml \
  --adapter_path "${BASE_ADAPTER}" \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /scratch/xxluo/mscoco/train2017 \
  --output_dir "${RESUME_DIR}" \
  --keep_ratios 0.10,0.20,0.30,0.40 \
  --ratio_sampling_probs 0.25,0.25,0.25,0.25 \
  --max_steps 2000 \
  --save_every 500 \
  --eval_every 500 \
  --gradient_accumulation_steps 4 \
  --attn_implementation flash_attention_2 \
  --ddp_timeout_minutes 60

echo "OPSD multi-ratio resume complete at $(date)"
echo "Logical step_3000 adapter: ${RESUME_DIR}/step_2000"
