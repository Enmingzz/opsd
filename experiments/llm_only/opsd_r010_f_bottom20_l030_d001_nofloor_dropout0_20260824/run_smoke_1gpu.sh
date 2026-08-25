#!/usr/bin/env bash
set -euo pipefail

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
EXP_ROOT=${OPSD_ROOT}/experiments/llm_only/opsd_r010_f_bottom20_l030_d001_nofloor_dropout0_20260824
OUT_DIR=/scratch/enmingzz/outputs/llm_only/opsd_r010_f_bottom20_l030_d001_nofloor_dropout0_20260824/smoke4
CONFIG=${EXP_ROOT}/configs/smoke4.yaml
mkdir -p "${OUT_DIR}"
cd "${OPSD_ROOT}"
source "${PROJECT_ROOT}/env/vsi-official.sh"
export PYTHONPATH=${PROJECT_ROOT}
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB034_ROOT=/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2
export TOKENIZERS_QWEN25_ROOT=/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only
export ARMEN_TRANSFORMERS_SRC=${OPSD_ROOT}/third_party/VLMEvalKit_armen51682/transformers/src
export VISIONZIP_QWEN25VL_ROOT=${PROJECT_ROOT}/VisionZip/Qwen2_5_VL
python "${EXP_ROOT}/validate_config.py" --config "${CONFIG}" --expected-steps 4 \
  --output "${OUT_DIR}/preflight.json" --overwrite
exec python visionzip_aokvqa/train.py --config "${CONFIG}" --output_dir "${OUT_DIR}"
