#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/project/6101803/enmingzz"
REPO_ROOT="${PROJECT_ROOT}/opsd"
ARMEN_VLMEVALKIT_ROOT="${REPO_ROOT}/third_party/VLMEvalKit_armen51682"
OUT_ROOT="/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning"
OPENCV_ROOT="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/gcc12/cuda12.2/opencv/4.11.0"
TOKENIZERS_QWEN25_ROOT="${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}"
HF_HUB034_ROOT="${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}"

MODEL_NAME="${MODEL_NAME:-Qwen}"
ADAPTER_TAG="${ADAPTER_TAG:?Set ADAPTER_TAG, for example sft or epic}"
RATIO_TAG="${RATIO_TAG:?Set RATIO_TAG, for example r005}"
EVAL_DATASETS="${EVAL_DATASETS:-MME MMStar POPE}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}} % 20000) + ${SLURM_ARRAY_TASK_ID:-0}))}"
RUN_GROUP="${RUN_GROUP:-armen51682_qwen25vl_hires_reasoning_adapters_ratios_${SLURM_ARRAY_JOB_ID:-manual}}"
EVAL_NAME="${EVAL_NAME:-${RUN_GROUP}/${ADAPTER_TAG}_${RATIO_TAG}}"
WORK_DIR="${OUT_ROOT}/eval_vlmevalkit/${EVAL_NAME}"
LOG_DIR="${OUT_ROOT}/logs/full/eval_${RUN_GROUP}"
export MODEL_NAME

source "${PROJECT_ROOT}/env/vsi-official.sh"

export HF_HOME="/scratch/enmingzz/hf_cache"
export TRANSFORMERS_CACHE="/scratch/enmingzz/hf_cache"
export LMUData="/scratch/enmingzz/vlmevalkit_data"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export DIST_TIMEOUT="${DIST_TIMEOUT:-14400}"
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

export model_path="${model_path:-Qwen/Qwen2.5-VL-7B-Instruct}"
export adapter_path="${adapter_path:?Set adapter_path to a PEFT adapter directory}"
export enable_thinking="${enable_thinking:-True}"
export temperature="${temperature:-0.000001}"
export num_return_sequences="${num_return_sequences:-1}"
export use_kv_cache="${use_kv_cache:-True}"
export enable_visionzip="${enable_visionzip:-True}"
export visionzip_ratio="${visionzip_ratio:?Set visionzip_ratio}"

export PYTHONPATH="${ARMEN_VLMEVALKIT_ROOT}:${ARMEN_VLMEVALKIT_ROOT}/transformers/src:${ARMEN_VLMEVALKIT_ROOT}/internvl:${HF_HUB034_ROOT}:${TOKENIZERS_QWEN25_ROOT}:${PROJECT_ROOT}:${OPENCV_ROOT}/lib/python3.11/site-packages${SANITIZED_PYTHONPATH:+:${SANITIZED_PYTHONPATH}}"
export LD_LIBRARY_PATH="${OPENCV_ROOT}/lib64:${OPENCV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${LMUData}"

if [[ ! -f "${adapter_path}/adapter_model.safetensors" ]]; then
  echo "Missing adapter_model.safetensors under adapter_path=${adapter_path}" >&2
  exit 1
fi

read -r -a DATASET_ARGS <<< "${EVAL_DATASETS}"

cd "${ARMEN_VLMEVALKIT_ROOT}"

echo "[$(date --iso-8601=seconds)] VLMEvalKit=${ARMEN_VLMEVALKIT_ROOT}"
echo "[$(date --iso-8601=seconds)] commit=$(git rev-parse HEAD)"
echo "[$(date --iso-8601=seconds)] work_dir=${WORK_DIR}"
echo "[$(date --iso-8601=seconds)] datasets=${EVAL_DATASETS}"
echo "[$(date --iso-8601=seconds)] model_name=${MODEL_NAME}"
echo "[$(date --iso-8601=seconds)] adapter_tag=${ADAPTER_TAG}"
echo "[$(date --iso-8601=seconds)] ratio_tag=${RATIO_TAG}"
echo "[$(date --iso-8601=seconds)] model_path=${model_path}"
echo "[$(date --iso-8601=seconds)] adapter_path=${adapter_path}"
echo "[$(date --iso-8601=seconds)] enable_thinking=${enable_thinking}"
echo "[$(date --iso-8601=seconds)] enable_visionzip=${enable_visionzip}"
echo "[$(date --iso-8601=seconds)] visionzip_ratio=${visionzip_ratio}"
echo "[$(date --iso-8601=seconds)] temperature=${temperature}"
echo "[$(date --iso-8601=seconds)] use_kv_cache=${use_kv_cache}"
echo "[$(date --iso-8601=seconds)] eval_nproc_per_node=${EVAL_NPROC_PER_NODE}"
echo "[$(date --iso-8601=seconds)] master_port=${MASTER_PORT}"
echo "[$(date --iso-8601=seconds)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
from __future__ import annotations

import inspect
import os
from pathlib import Path

import cv2
import pyarrow
import tokenizers
import transformers
from vlmeval.config import supported_VLM
from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

model_name = os.environ["MODEL_NAME"]
kwargs = supported_VLM[model_name].keywords
adapter_path = Path(os.environ["adapter_path"])
print("cv2", cv2.__version__)
print("pyarrow", pyarrow.__version__)
print("transformers", transformers.__file__, transformers.__version__)
print("tokenizers", tokenizers.__file__, tokenizers.__version__)
print("Qwen2VLChat_default_max_new_tokens", inspect.signature(Qwen2VLChat.__init__).parameters["max_new_tokens"].default)
print("model_kwargs", {k: kwargs.get(k) for k in [
    "model_path",
    "adapter_path",
    "min_pixels",
    "max_pixels",
    "use_custom_prompt",
    "enable_thinking",
    "enable_visionzip",
    "visionzip_ratio",
    "temperature",
    "use_kv_cache",
    "num_return_sequences",
]})
assert "VLMEvalKit_armen51682/transformers/src" in transformers.__file__
assert adapter_path.joinpath("adapter_model.safetensors").is_file()
assert kwargs.get("model_path") == os.environ["model_path"]
assert kwargs.get("adapter_path") == os.environ["adapter_path"]
assert kwargs.get("min_pixels") == 1280 * 28 * 28
assert kwargs.get("max_pixels") == 16384 * 28 * 28
assert kwargs.get("use_custom_prompt") is False
assert kwargs.get("enable_thinking") is True
assert kwargs.get("enable_visionzip") == (os.environ["enable_visionzip"].lower() in ("1", "true", "yes"))
assert abs(float(kwargs.get("visionzip_ratio")) - float(os.environ["visionzip_ratio"])) < 1e-12
assert abs(float(kwargs.get("temperature")) - float(os.environ["temperature"])) < 1e-12
assert kwargs.get("use_kv_cache") == (os.environ["use_kv_cache"].lower() in ("1", "true", "yes"))
assert inspect.signature(Qwen2VLChat.__init__).parameters["max_new_tokens"].default == 2048
PY

if [[ "${EVAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[$(date --iso-8601=seconds)] preflight-only finished"
  exit 0
fi

if [[ "${EVAL_NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun \
    --nnodes=1 \
    --nproc-per-node="${EVAL_NPROC_PER_NODE}" \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    run.py \
      --data "${DATASET_ARGS[@]}" \
      --model "${MODEL_NAME}" \
      --mode all \
      --work-dir "${WORK_DIR}"
else
  python run.py \
    --data "${DATASET_ARGS[@]}" \
    --model "${MODEL_NAME}" \
    --mode all \
    --work-dir "${WORK_DIR}"
fi

echo "[$(date --iso-8601=seconds)] finished"
