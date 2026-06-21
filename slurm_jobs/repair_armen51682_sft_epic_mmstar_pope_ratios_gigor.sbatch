#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
#SBATCH --job-name=repair_mmstar_pope
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/repair_armen51682_sft_epic_mmstar_pope_ratios_gigor/slurm-%A_%a.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/repair_armen51682_sft_epic_mmstar_pope_ratios_gigor/slurm-%A_%a.err

set -euo pipefail

PROJECT_ROOT="/project/6101803/enmingzz"
REPO_ROOT="${PROJECT_ROOT}/opsd"
ARMEN_VLMEVALKIT_ROOT="${REPO_ROOT}/third_party/VLMEvalKit_armen51682"
OUT_ROOT="/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning"
OPENCV_ROOT="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/gcc12/cuda12.2/opencv/4.11.0"
TOKENIZERS_QWEN25_ROOT="${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}"
HF_HUB034_ROOT="${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}"

mkdir -p /scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/repair_armen51682_sft_epic_mmstar_pope_ratios_gigor

source "${PROJECT_ROOT}/env/vsi-official.sh"

export HF_HOME="/scratch/enmingzz/hf_cache"
export TRANSFORMERS_CACHE="/scratch/enmingzz/hf_cache"
export LMUData="/scratch/enmingzz/vlmevalkit_data"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset MMEVAL_ROOT

SANITIZED_PYTHONPATH=""
if [[ -n "${PYTHONPATH:-}" ]]; then
  IFS=':' read -r -a PYTHONPATH_PARTS <<< "${PYTHONPATH}"
  for path in "${PYTHONPATH_PARTS[@]}"; do
    if [[ -z "${path}" || "${path}" == /scratch/enmingzz/temp/qwen25_bootstrap* ]]; then
      continue
    fi
    SANITIZED_PYTHONPATH="${SANITIZED_PYTHONPATH:+${SANITIZED_PYTHONPATH}:}${path}"
  done
fi

export PYTHONPATH="${ARMEN_VLMEVALKIT_ROOT}:${ARMEN_VLMEVALKIT_ROOT}/transformers/src:${ARMEN_VLMEVALKIT_ROOT}/internvl:${HF_HUB034_ROOT}:${TOKENIZERS_QWEN25_ROOT}:${PROJECT_ROOT}:${OPENCV_ROOT}/lib/python3.11/site-packages${SANITIZED_PYTHONPATH:+:${SANITIZED_PYTHONPATH}}"
export LD_LIBRARY_PATH="${OPENCV_ROOT}/lib64:${OPENCV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

SFT_ADAPTER="/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/checkpoints/aokvqa_hires_reasoning_ebs32_4l40s_armenvzfix_20260614_162619/04_sft_reasoning_ebs32/final"
EPIC_ADAPTER="/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/checkpoints/aokvqa_hires_reasoning_ebs32_4l40s_armenvzfix_20260614_162619/05_epic_tcd_reasoning_ebs32/final"
OLD_RUN_ROOT="${OUT_ROOT}/eval_vlmevalkit/armen51682_qwen25vl_hires_reasoning_sft_epic_vz_ratios_mme_mmstar_pope_gigor_3959824"

TASK_ADAPTERS=(sft sft sft sft epic epic epic epic)
TASK_RATIOS=(r005 r010 r020 r030 r005 r010 r020 r030)
TASK_VISIONZIP_RATIOS=(1.00 0.95 0.85 0.75 1.00 0.95 0.85 0.75)

ADAPTER_TAG="${TASK_ADAPTERS[$SLURM_ARRAY_TASK_ID]}"
RATIO_TAG="${TASK_RATIOS[$SLURM_ARRAY_TASK_ID]}"
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

export MODEL_NAME=Qwen
export model_path="Qwen/Qwen2.5-VL-7B-Instruct"
export enable_thinking=True
export enable_visionzip=True
export temperature=0.000001
export num_return_sequences=1
export use_kv_cache=True

PRED_ROOT="${OLD_RUN_ROOT}/${ADAPTER_TAG}_${RATIO_TAG}/Qwen/T20260615_G51682a6b"

cd "${ARMEN_VLMEVALKIT_ROOT}"

echo "[$(date --iso-8601=seconds)] repair ${ADAPTER_TAG}_${RATIO_TAG}"
echo "pred_root=${PRED_ROOT}"
echo "adapter_path=${adapter_path}"
echo "visionzip_ratio=${visionzip_ratio}"

python "${REPO_ROOT}/scripts/repair_armen51682_mmstar_pope_from_shards.py" \
  --pred-root "${PRED_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --world-size 4 \
  --fill-pope-rank 1
