#!/usr/bin/env bash

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/project/6101803/enmingzz/vlm}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/enmingzz/temp}"

DATA_ROOT="${SCRATCH_ROOT}/datasets/vsi_bench"
VIDEO_ROOT="${DATA_ROOT}/videos"
OUT_ROOT="${SCRATCH_ROOT}/outputs/vsi_bench"
LOG_ROOT="${SCRATCH_ROOT}/logs/vsi_bench"
CACHE_ROOT="${SCRATCH_ROOT}/cache"
TMPDIR="${SCRATCH_ROOT}/tmp"
VENV_ROOT="${SCRATCH_ROOT}/venvs/vsi-fastv"
VENV_VIDEO_ROOT="${SCRATCH_ROOT}/venvs/vsi-video"
VENV_OFFICIAL_ROOT="${SCRATCH_ROOT}/venvs/vsi-official"

export CODE_ROOT
export SCRATCH_ROOT
export DATA_ROOT
export VIDEO_ROOT
export OUT_ROOT
export LOG_ROOT
export CACHE_ROOT
export TMPDIR
export VENV_ROOT
export VENV_VIDEO_ROOT
export VENV_OFFICIAL_ROOT

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export VSIBENCH_VIDEO_ROOT="${HF_HOME}/vsibench"
export REVSI_VIDEO_ROOT="${HF_HOME}/revsi_official"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/pycache"

mkdir -p \
  "${DATA_ROOT}/raw" \
  "${VIDEO_ROOT}" \
  "${OUT_ROOT}" \
  "${LOG_ROOT}" \
  "${HF_DATASETS_CACHE}" \
  "${VSIBENCH_VIDEO_ROOT}" \
  "${REVSI_VIDEO_ROOT}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${TMPDIR}" \
  "$(dirname "${VENV_ROOT}")" \
  "$(dirname "${VENV_VIDEO_ROOT}")" \
  "$(dirname "${VENV_OFFICIAL_ROOT}")"
