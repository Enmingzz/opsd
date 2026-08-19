#!/usr/bin/env bash
set -euo pipefail

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
EXP_ROOT="${OPSD_ROOT}/experiments/llm_only/opsd_r010_projection_mass_va_group_dropout0_20260818"
CONFIG_PATH="${EXP_ROOT}/configs/smoke4.yaml"
OUTPUT_ROOT=/scratch/enmingzz/outputs/llm_only/opsd_r010_projection_mass_va_group_dropout0_20260818
OUT_DIR="${OUTPUT_ROOT}/smoke4"
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing smoke output: ${OUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"
cd "${OPSD_ROOT}"

source "${PROJECT_ROOT}/env/vsi-official.sh"
export HF_HOME="${HF_HOME:-/home/enmingzz/scratch/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB034_ROOT="${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}"
export TOKENIZERS_QWEN25_ROOT="${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}"
export ARMEN_TRANSFORMERS_SRC="${ARMEN_TRANSFORMERS_SRC:-${OPSD_ROOT}/third_party/VLMEvalKit_armen51682/transformers/src}"
export VISIONZIP_QWEN25VL_ROOT="${VISIONZIP_QWEN25VL_ROOT:-${PROJECT_ROOT}/VisionZip/Qwen2_5_VL}"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}"

python "${EXP_ROOT}/validate_config.py" --config "${CONFIG_PATH}" \
  --expected-steps 4 --expected-top-fraction 0.10 \
  --expected-high-group-lambda 0.30 \
  --output "${OUT_DIR}/preflight.json" --overwrite
python visionzip_aokvqa/train.py --config "${CONFIG_PATH}" --output_dir "${OUT_DIR}"
python experiments/scripts/verify_lora_scope.py \
  --config "${CONFIG_PATH}" --expected-scope language_decoder_only \
  --adapter-path "${OUT_DIR}/final" \
  --output "${OUT_DIR}/final_scope_verification.json" --overwrite
