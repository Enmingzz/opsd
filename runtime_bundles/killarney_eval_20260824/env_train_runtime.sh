#!/usr/bin/env bash
set -euo pipefail

export CKPT_EVAL_ROOT="${CKPT_EVAL_ROOT:-/project/6101803/enmingzz/ckpt_eval_trainenv}"
export PROJECT_ROOT="${PROJECT_ROOT:-/project/6101803/enmingzz}"
export VLM_ROOT="${VLM_ROOT:-${CKPT_EVAL_ROOT}/VLMEvalKit_armen51682}"
export HF_HUB034_ROOT="${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}"
export TOKENIZERS_QWEN25_ROOT="${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}"
export OPENCV_ROOT="${OPENCV_ROOT:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/gcc12/cuda12.2/opencv/4.11.0}"
export VSI_OFFICIAL_SITE_PACKAGES="${VSI_OFFICIAL_SITE_PACKAGES:-/scratch/enmingzz/temp/venvs/vsi-official/lib/python3.11/site-packages}"

source "${PROJECT_ROOT}/env/vsi-official.sh"

export HF_HOME="${HF_HOME:-/scratch/enmingzz/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export LMUData="${LMUData:-/scratch/enmingzz/vlmevalkit_data}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-14400}"
unset MMEVAL_ROOT

SANITIZED_PYTHONPATH=""
if [[ -n "${PYTHONPATH:-}" ]]; then
  IFS=':' read -r -a PYTHONPATH_PARTS <<< "${PYTHONPATH}"
  for path in "${PYTHONPATH_PARTS[@]}"; do
    case "${path}" in
      ""|/scratch/enmingzz/temp/qwen25_bootstrap*|/scratch/enmingzz/temp/pydeps_armen_clean*|/project/6101803/enmingzz/vlm/official_thinking_in_space*|*/VLMEvalKit_armen51682*)
        continue
        ;;
    esac
    SANITIZED_PYTHONPATH="${SANITIZED_PYTHONPATH:+${SANITIZED_PYTHONPATH}:}${path}"
  done
fi

export PYTHONPATH="${VLM_ROOT}:${VLM_ROOT}/transformers/src:${VLM_ROOT}/internvl:${PROJECT_ROOT}:${HF_HUB034_ROOT}:${TOKENIZERS_QWEN25_ROOT}:${VSI_OFFICIAL_SITE_PACKAGES}:${OPENCV_ROOT}/lib/python3.11/site-packages${SANITIZED_PYTHONPATH:+:${SANITIZED_PYTHONPATH}}"
export LD_LIBRARY_PATH="${OPENCV_ROOT}/lib64:${OPENCV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LMUData}" "${HF_HOME}"
