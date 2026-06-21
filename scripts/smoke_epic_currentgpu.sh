#!/usr/bin/env bash
set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/project/6101803/enmingzz/opsd}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/6101803/enmingzz}"
OUT_DIR="${1:-/scratch/enmingzz/temp/outputs/vsi_bench/checkpoints/_smoke_epic_debug_currentgpu}"
CONFIG_PATH="${2:-${OPSD_ROOT}/configs/visionzip_aokvqa/_tmp_epic_tcd_reasoning_smoke_device0.yaml}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

cd "${OPSD_ROOT}"

source "${PROJECT_ROOT}/env/vsi-official.sh"
export HF_HOME="${HF_HOME:-/home/enmingzz/scratch/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB034_ROOT="${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}"
export TOKENIZERS_QWEN25_ROOT="${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}"
export VSI_OFFICIAL_SITE_PACKAGES="${VSI_OFFICIAL_SITE_PACKAGES:-/scratch/enmingzz/temp/venvs/vsi-official/lib/python3.11/site-packages}"
export ARMEN_TRANSFORMERS_SRC="${ARMEN_TRANSFORMERS_SRC:-${OPSD_ROOT}/third_party/VLMEvalKit_armen51682/transformers/src}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

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
export PYTHONPATH="${PROJECT_ROOT}:${ARMEN_TRANSFORMERS_SRC}:${HF_HUB034_ROOT}:${TOKENIZERS_QWEN25_ROOT}:${VSI_OFFICIAL_SITE_PACKAGES}${SANITIZED_PYTHONPATH:+:${SANITIZED_PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export VISIONZIP_QWEN25VL_ROOT="${VISIONZIP_QWEN25VL_ROOT:-${OPSD_ROOT}/third_party/VisionZip/Qwen2_5_VL}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) smoke_out=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

python - <<'PY'
from opsd.visionzip_aokvqa.qwen_wrapper import import_qwen25_modules

import_qwen25_modules()

import datasets
import flash_attn
import huggingface_hub
import PIL
import pyarrow
import tokenizers
import torch
import transformers

print(
    {
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__,
        "flash_attn": getattr(flash_attn, "__version__", "ok"),
        "PIL": PIL.__version__,
        "tokenizers": tokenizers.__version__,
        "hf_hub": huggingface_hub.__version__,
        "pyarrow": pyarrow.__version__,
        "datasets": datasets.__version__,
    }
)
PY

python visionzip_aokvqa/train.py \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUT_DIR}" \
  --max_steps 1 \
  --limit 1 \
  --gradient_accumulation_steps 1 \
  --max_new_tokens "${MAX_NEW_TOKENS}"

echo "[$(date --iso-8601=seconds)] EPIC smoke completed"
