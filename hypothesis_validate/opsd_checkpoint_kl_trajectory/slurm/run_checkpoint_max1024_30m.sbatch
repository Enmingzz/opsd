#!/usr/bin/env bash
#SBATCH --job-name=opsd-kl1024
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/logs/opsd_checkpoint_kl_metric_noocr_max1024_20260729/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/logs/opsd_checkpoint_kl_metric_noocr_max1024_20260729/%x-%j.err

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT_LABEL ADAPTER_PATH_OR_BASE" >&2
  exit 2
fi

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
CONFIG=${OPSD_ROOT}/hypothesis_validate/opsd_checkpoint_kl_trajectory/config_max1024.json
LABEL="$1"
ADAPTER_PATH="$2"

mkdir -p /scratch/enmingzz/outputs/llm_only/logs/opsd_checkpoint_kl_metric_noocr_max1024_20260729
mkdir -p /scratch/enmingzz/cache/opsd_checkpoint_kl_max1024_tmp
cd "${OPSD_ROOT}"
source "${PROJECT_ROOT}/env/vsi-official.sh"

export HF_HOME=${HF_HOME:-/home/enmingzz/scratch/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}
export HF_HUB034_ROOT=${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}
export TOKENIZERS_QWEN25_ROOT=${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}
export ARMEN_TRANSFORMERS_SRC=/project/6101803/enmingzz/ckpt_eval_trainenv/VLMEvalKit_armen51682/transformers/src
export VISIONZIP_QWEN25VL_ROOT=${PROJECT_ROOT}/VisionZip/Qwen2_5_VL
export OPSD_PRUNING_METHOD=visionzip
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/scratch/enmingzz/cache/opsd_checkpoint_kl_max1024_tmp
export PYTHONPATH=${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}

echo "date=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-}"
echo "label=${LABEL}"
echo "adapter=${ADAPTER_PATH}"
echo "git=$(git rev-parse HEAD)"
nvidia-smi

python hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/run_checkpoint.py \
  --config "${CONFIG}" \
  --checkpoint-label "${LABEL}" \
  --adapter-path "${ADAPTER_PATH}"
