#!/bin/bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --job-name=armen_test4g
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_armen51682_qwen25vl_hires_reasoning_armen_test_4gpu_btaati/slurm-%A_%a.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_armen51682_qwen25vl_hires_reasoning_armen_test_4gpu_btaati/slurm-%A_%a.err

set -euo pipefail

mkdir -p /scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_armen51682_qwen25vl_hires_reasoning_armen_test_4gpu_btaati

TAGS=(base vz_r005 vz_r010 vz_r020 vz_r030)
ENABLE_VISIONZIP=(False True True True True)
VISIONZIP_RATIOS=(0.0 1.0 0.95 0.85 0.75)

export ARMEN_RUN_TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
export EVAL_NAME="armen51682_qwen25vl_hires_reasoning_armen_test_4gpu_${ARMEN_RUN_TAG}_mme"
export enable_visionzip="${ENABLE_VISIONZIP[$SLURM_ARRAY_TASK_ID]}"
export visionzip_ratio="${VISIONZIP_RATIOS[$SLURM_ARRAY_TASK_ID]}"
export use_kv_cache=True
export EVAL_NPROC_PER_NODE=4
export MODEL_NAME=Qwen

/bin/bash /project/6101803/enmingzz/opsd/scripts/eval_armen_qwen25vl_reasoning_mme.sh
