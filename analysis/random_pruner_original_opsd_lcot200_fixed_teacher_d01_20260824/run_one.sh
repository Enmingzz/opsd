#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 STEP" >&2
  exit 2
fi

STEP=$1
case "${STEP}" in
  0|1024|2048|3072) ;;
  *) echo "STEP must be one of 0, 1024, 2048, 3072" >&2; exit 2 ;;
esac

PROJECT=/project/6101803/enmingzz
OPSD_ROOT=${PROJECT}/opsd
HERE=${OPSD_ROOT}/analysis/random_pruner_original_opsd_lcot200_fixed_teacher_d01_20260824
SCORER=${OPSD_ROOT}/analysis/r010_only_lcot1k_fixed_teacher_deltas_20260823/run_checkpoint.py
CHECKPOINT_ROOT=/scratch/enmingzz/outputs/llm_only/random_pruner_group_angle_T015_dropout0_matched_20260814/original_control/train_9984/eval_snapshots
PADDED=$(printf '%06d' "${STEP}")
ADAPTER=${CHECKPOINT_ROOT}/step_${PADDED}
OUTPUT=${HERE}/outputs/step_${PADDED}

if [[ ! -s "${ADAPTER}/adapter_model.safetensors" ]]; then
  echo "Missing adapter: ${ADAPTER}/adapter_model.safetensors" >&2
  exit 1
fi

cd "${OPSD_ROOT}"
source "${PROJECT}/env/vsi-official.sh"
export HF_HOME=/scratch/enmingzz/.cache/huggingface
export HF_HUB_CACHE=/scratch/enmingzz/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/enmingzz/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/enmingzz/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB034_ROOT=${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}
export TOKENIZERS_QWEN25_ROOT=${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}
export ARMEN_TRANSFORMERS_SRC=${PROJECT}/ckpt_eval_trainenv/VLMEvalKit_armen51682/transformers/src
export VISIONZIP_QWEN25VL_ROOT=${PROJECT}/VisionZip/Qwen2_5_VL
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OPSD_PRUNING_METHOD=random
export OPSD_RANDOM_PRUNER_SEED=42
export PYTHONPATH=${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}
export TMPDIR=/scratch/enmingzz/cache/random_pruner_lcot200_fixed_teacher_d01_tmp
mkdir -p "${TMPDIR}" "${OUTPUT}"

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-direct} step=${STEP}"
echo "adapter=${ADAPTER} output=${OUTPUT}"
nvidia-smi

exec python "${SCORER}" \
  --step "${STEP}" \
  --adapter-path "${ADAPTER}" \
  --output-dir "${OUTPUT}" \
  --limit 200 \
  --offset 0 \
  --max-new-tokens 1024 \
  --min-pixels $((1280 * 28 * 28)) \
  --max-pixels $((4096 * 28 * 28)) \
  --seed 42 \
  --teacher-mode fixed_base \
  --pruning-method random \
  --random-pruner-seed 42 \
  --deltas 0.01
