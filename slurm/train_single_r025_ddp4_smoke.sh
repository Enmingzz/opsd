#!/bin/bash
# Adjust --account/--gres if your cluster uses different names.
#SBATCH --job-name=opsd_r025_ddp_smoke
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/enmingzz/temp/opsd_logs/%x_%j.out

set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/real_single_r025
mkdir -p "${OUT_ROOT}" /scratch/enmingzz/temp/opsd_logs

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  opsd/scripts/train_qwen25vl_prune_distill.py \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_single_r025.yaml \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava2k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava200_val.jsonl \
  --image_root /scratch/xxluo/mscoco/train2017 \
  --output_dir "${OUT_ROOT}/divprune_kl_single_r025_ddp4_smoke20" \
  --max_steps 20 \
  --save_every 10 \
  --eval_every 10 \
  --gradient_accumulation_steps 1 \
  --max_new_tokens 8 \
  --log_every 1 \
  --attn_implementation flash_attention_2
