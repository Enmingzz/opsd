#!/bin/bash
# Adjust --account/--gres if your cluster uses different names.
#SBATCH --job-name=opsd_sweep
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

python opsd/scripts/sweep_pruned_baselines.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --eval_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root "${OPSD_IMAGE_ROOT:-/path/to/coco/train2017}" \
  --output_dir /scratch/enmingzz/temp/opsd_runs/baseline_sweep_llava1k \
  --pruners random,grid,divprune_lite,vscan_stage1 \
  --keep_ratios 1.0,0.5,0.25,0.125 \
  --student_input_mode drop_tokens \
  --max_new_tokens 32 \
  --limit 1000 \
  --seed 42 \
  --attn_implementation flash_attention_2
