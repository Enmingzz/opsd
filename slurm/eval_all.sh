#!/bin/bash
# Adjust --account/--gres if your cluster uses different names.
#SBATCH --job-name=opsd_eval
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/enmingzz/temp/opsd_logs/%x_%j.out

set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false

for ratio in 0.5 0.25 0.125; do
  python opsd/scripts/eval_qwen25vl_pruned_student.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_path /scratch/enmingzz/temp/opsd_runs/divprune_kl_multiratio/step_1000 \
    --eval_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
    --image_root "${OPSD_IMAGE_ROOT:-/path/to/coco/train2017}" \
    --output_jsonl "/scratch/enmingzz/temp/opsd_runs/divprune_kl_multiratio/eval_r${ratio}.jsonl" \
    --keep_ratio "${ratio}" \
    --pruner divprune_lite \
    --student_input_mode drop_tokens \
    --max_new_tokens 32 \
    --limit 1000 \
    --attn_implementation flash_attention_2
done
