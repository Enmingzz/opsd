#!/bin/bash
# Adjust --account/--gres if your cluster uses different names.
#SBATCH --job-name=opsd_r025
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

python opsd/scripts/train_qwen25vl_prune_distill.py \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_single_r025.yaml \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root "${OPSD_IMAGE_ROOT:-/path/to/coco/train2017}" \
  --output_dir /scratch/enmingzz/temp/opsd_runs/divprune_kl_single_r025 \
  --max_steps 1000 \
  --save_every 500 \
  --eval_every 500
