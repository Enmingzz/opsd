#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --array=0-7
#SBATCH --job-name=sft_epic_reason_eval
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_armen51682_qwen25vl_hires_reasoning_sft_epic_vz_ratios_mme_mmstar_pope_gigor/slurm-%A_%a.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_armen51682_qwen25vl_hires_reasoning_sft_epic_vz_ratios_mme_mmstar_pope_gigor/slurm-%A_%a.err

set -euo pipefail

mkdir -p /scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_armen51682_qwen25vl_hires_reasoning_sft_epic_vz_ratios_mme_mmstar_pope_gigor

SFT_ADAPTER="/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/checkpoints/aokvqa_hires_reasoning_ebs32_4l40s_armenvzfix_20260614_162619/04_sft_reasoning_ebs32/final"
EPIC_ADAPTER="/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/checkpoints/aokvqa_hires_reasoning_ebs32_4l40s_armenvzfix_20260614_162619/05_epic_tcd_reasoning_ebs32/final"

TASK_ADAPTERS=(sft sft sft sft epic epic epic epic)
TASK_RATIOS=(r005 r010 r020 r030 r005 r010 r020 r030)
TASK_VISIONZIP_RATIOS=(1.00 0.95 0.85 0.75 1.00 0.95 0.85 0.75)

export ADAPTER_TAG="${TASK_ADAPTERS[$SLURM_ARRAY_TASK_ID]}"
export RATIO_TAG="${TASK_RATIOS[$SLURM_ARRAY_TASK_ID]}"
export visionzip_ratio="${TASK_VISIONZIP_RATIOS[$SLURM_ARRAY_TASK_ID]}"

case "${ADAPTER_TAG}" in
  sft)
    export adapter_path="${SFT_ADAPTER}"
    ;;
  epic)
    export adapter_path="${EPIC_ADAPTER}"
    ;;
  *)
    echo "Unknown ADAPTER_TAG=${ADAPTER_TAG}" >&2
    exit 1
    ;;
esac

export RUN_GROUP="armen51682_qwen25vl_hires_reasoning_sft_epic_vz_ratios_mme_mmstar_pope_gigor_${SLURM_ARRAY_JOB_ID}"
export EVAL_NAME="${RUN_GROUP}/${ADAPTER_TAG}_${RATIO_TAG}"
export EVAL_DATASETS="MME MMStar POPE"
export EVAL_NPROC_PER_NODE=4
export MODEL_NAME=Qwen
export model_path="Qwen/Qwen2.5-VL-7B-Instruct"
export enable_thinking=True
export enable_visionzip=True
export temperature=0.000001
export num_return_sequences=1
export use_kv_cache=True
export MASTER_PORT="$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))"

/bin/bash /project/6101803/enmingzz/opsd/scripts/eval_armen51682_qwen25vl_reasoning_adapters_ratios.sh
