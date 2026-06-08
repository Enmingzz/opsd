#!/bin/bash
# Adjust --account/--gres if your cluster uses different names.
#SBATCH --job-name=opsd_stress
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/enmingzz/temp/opsd_logs/%x_%j.out

set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
# OPSD is under /project/6101803/enmingzz/opsd in this workspace.
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false

python opsd/tests/stress_test_normal_resolution.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --attn_implementation flash_attention_2
