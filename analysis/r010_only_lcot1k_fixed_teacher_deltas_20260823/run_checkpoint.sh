#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 STEP OUTPUT_DIR [LIMIT] [OFFSET]" >&2
  exit 2
fi

STEP=$1
OUTPUT_DIR=$2
LIMIT=${3:-1000}
OFFSET=${4:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
ROOT=/project/6101803/enmingzz/opsd
PROJECT=/project/6101803/enmingzz
PADDED=$(printf '%06d' "${STEP}")
ADAPTER=/scratch/enmingzz/outputs/llm_only/opsd_random_r010_only_dropout0_20260815/run/eval_snapshots/step_${PADDED}

cd "${ROOT}"
source "${PROJECT}/env/vsi-official.sh"
export HF_HOME=${HF_HOME:-/scratch/enmingzz/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}
export HF_HUB034_ROOT=${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}
export TOKENIZERS_QWEN25_ROOT=${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}
export ARMEN_TRANSFORMERS_SRC=/project/6101803/enmingzz/ckpt_eval_trainenv/VLMEvalKit_armen51682/transformers/src
export VISIONZIP_QWEN25VL_ROOT=${PROJECT}/VisionZip/Qwen2_5_VL
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OPSD_PRUNING_METHOD=visionzip
export PYTHONPATH=${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}
export TMPDIR=/scratch/enmingzz/cache/r010_lcot1k_fixed_teacher_deltas_tmp
mkdir -p "${TMPDIR}" "${OUTPUT_DIR}"

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-direct} step=${STEP}"
echo "adapter=${ADAPTER} output=${OUTPUT_DIR} limit=${LIMIT} offset=${OFFSET} max_new_tokens=${MAX_NEW_TOKENS}"
nvidia-smi

exec python analysis/r010_only_lcot1k_fixed_teacher_deltas_20260823/run_checkpoint.py \
  --step "${STEP}" \
  --adapter-path "${ADAPTER}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --min-pixels $((1280 * 28 * 28)) \
  --max-pixels $((4096 * 28 * 28)) \
  --seed 42
