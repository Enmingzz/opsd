#!/usr/bin/env bash
set -euo pipefail

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
EXP_ROOT=${OPSD_ROOT}/experiments/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824
CONFIG=${EXP_ROOT}/configs/smoke4.yaml
OUT_DIR=${1:-/scratch/enmingzz/outputs/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824/smoke4}
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"
cd "${OPSD_ROOT}"
source "${PROJECT_ROOT}/env/vsi-official.sh"
export HF_HOME=${HF_HOME:-/home/enmingzz/scratch/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}
export HF_HUB034_ROOT=/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2
export TOKENIZERS_QWEN25_ROOT=/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only
export ARMEN_TRANSFORMERS_SRC=${OPSD_ROOT}/third_party/VLMEvalKit_armen51682/transformers/src
export VISIONZIP_QWEN25VL_ROOT=${PROJECT_ROOT}/VisionZip/Qwen2_5_VL
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 PYTHONPATH=${PROJECT_ROOT}

python "${EXP_ROOT}/validate_config.py" --config "${CONFIG}" --expected-steps 4 \
  --output "${OUT_DIR}/preflight.json" --overwrite
python visionzip_aokvqa/train.py --config "${CONFIG}" --output_dir "${OUT_DIR}"
python "${EXP_ROOT}/audit_smoke.py" --run-dir "${OUT_DIR}"
python experiments/scripts/verify_lora_scope.py \
  --config "${CONFIG}" --expected-scope language_decoder_only \
  --output "${OUT_DIR}/lora_scope_preflight.json" --overwrite
echo "smoke_output=${OUT_DIR}"
