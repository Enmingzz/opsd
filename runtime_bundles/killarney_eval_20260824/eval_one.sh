#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_train_runtime.sh"

OUT_ROOT="${CKPT_EVAL_OUT_ROOT:-/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning}"
MODEL_NAME="${MODEL_NAME:-Qwen}"
ADAPTER_TAG="${ADAPTER_TAG:?Set ADAPTER_TAG, for example sft or opsd_ema}"
RATIO_TAG="${RATIO_TAG:?Set RATIO_TAG, for example r030}"
adapter_path="${adapter_path:-}"
case "${adapter_path}" in
  none|None|null|NULL|-)
    adapter_path=""
    ;;
esac
visionzip_ratio="${visionzip_ratio:?Set visionzip_ratio, for example 0.75}"

export MODEL_NAME
export model_path="${model_path:-Qwen/Qwen2.5-VL-7B-Instruct}"
export adapter_path
export enable_thinking="${enable_thinking:-True}"
export enable_visionzip="${enable_visionzip:-True}"
export temperature="${temperature:-0.0}"
export num_return_sequences="${num_return_sequences:-1}"
export use_kv_cache="${use_kv_cache:-True}"
export visionzip_ratio
export VLMEVAL_STRICT_ERRORS="${VLMEVAL_STRICT_ERRORS:-1}"
export VISIONZIP_ATTN_CHUNK="${VISIONZIP_ATTN_CHUNK:-128}"
export MMSTAR_QWEN_JUDGE="${MMSTAR_QWEN_JUDGE:-0}"
export MMSTAR_QWEN_JUDGE_MODEL_PATH="${MMSTAR_QWEN_JUDGE_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export MMSTAR_QWEN_JUDGE_DEVICE="${MMSTAR_QWEN_JUDGE_DEVICE:-cuda:0}"
export MMSTAR_QWEN_JUDGE_DTYPE="${MMSTAR_QWEN_JUDGE_DTYPE:-bfloat16}"
export MMSTAR_QWEN_JUDGE_SCOPE="${MMSTAR_QWEN_JUDGE_SCOPE:-misses}"
export MMSTAR_QWEN_JUDGE_ALLOW_OPEN_ANSWER="${MMSTAR_QWEN_JUDGE_ALLOW_OPEN_ANSWER:-1}"
export MMSTAR_QWEN_JUDGE_STRICT="${MMSTAR_QWEN_JUDGE_STRICT:-1}"

EVAL_DATASETS="${EVAL_DATASETS:-MME MMStar POPE}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}} % 20000) + ${SLURM_ARRAY_TASK_ID:-0}))}"
RUN_GROUP="${RUN_GROUP:-ckpt_trainenv_manual_$(date +%Y%m%d_%H%M%S)}"
EVAL_NAME="${EVAL_NAME:-${RUN_GROUP}/${ADAPTER_TAG}_${RATIO_TAG}}"
WORK_DIR="${WORK_DIR:-${OUT_ROOT}/eval_vlmevalkit_trainenv/${EVAL_NAME}}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs/full/eval_${RUN_GROUP}}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${LMUData}"

if [[ -n "${adapter_path}" ]]; then
  if [[ ! -f "${adapter_path}/adapter_model.safetensors" && ! -f "${adapter_path}/adapter_model.bin" ]]; then
    echo "Missing PEFT adapter weights under adapter_path=${adapter_path}" >&2
    exit 1
  fi
  if [[ ! -f "${adapter_path}/adapter_config.json" ]]; then
    echo "Missing adapter_config.json under adapter_path=${adapter_path}" >&2
    exit 1
  fi
fi

read -r -a DATASET_ARGS <<< "${EVAL_DATASETS}"

cd "${VLM_ROOT}"

echo "[$(date --iso-8601=seconds)] ckpt_eval_root=${CKPT_EVAL_ROOT}"
echo "[$(date --iso-8601=seconds)] vlmevalkit=${VLM_ROOT}"
echo "[$(date --iso-8601=seconds)] commit=$(git rev-parse HEAD)"
echo "[$(date --iso-8601=seconds)] tokenizers_qwen25_root=${TOKENIZERS_QWEN25_ROOT}"
echo "[$(date --iso-8601=seconds)] work_dir=${WORK_DIR}"
echo "[$(date --iso-8601=seconds)] datasets=${EVAL_DATASETS}"
echo "[$(date --iso-8601=seconds)] model_name=${MODEL_NAME}"
echo "[$(date --iso-8601=seconds)] adapter_tag=${ADAPTER_TAG}"
echo "[$(date --iso-8601=seconds)] ratio_tag=${RATIO_TAG}"
echo "[$(date --iso-8601=seconds)] model_path=${model_path}"
echo "[$(date --iso-8601=seconds)] adapter_path=${adapter_path:-<none>}"
echo "[$(date --iso-8601=seconds)] enable_thinking=${enable_thinking}"
echo "[$(date --iso-8601=seconds)] enable_visionzip=${enable_visionzip}"
echo "[$(date --iso-8601=seconds)] visionzip_ratio=${visionzip_ratio}"
echo "[$(date --iso-8601=seconds)] temperature=${temperature}"
echo "[$(date --iso-8601=seconds)] use_kv_cache=${use_kv_cache}"
echo "[$(date --iso-8601=seconds)] eval_nproc_per_node=${EVAL_NPROC_PER_NODE}"
echo "[$(date --iso-8601=seconds)] vlmeval_strict_errors=${VLMEVAL_STRICT_ERRORS}"
echo "[$(date --iso-8601=seconds)] visionzip_attn_chunk=${VISIONZIP_ATTN_CHUNK}"
echo "[$(date --iso-8601=seconds)] mmstar_qwen_judge=${MMSTAR_QWEN_JUDGE}"
echo "[$(date --iso-8601=seconds)] mmstar_qwen_judge_model_path=${MMSTAR_QWEN_JUDGE_MODEL_PATH}"
echo "[$(date --iso-8601=seconds)] mmstar_qwen_judge_device=${MMSTAR_QWEN_JUDGE_DEVICE}"
echo "[$(date --iso-8601=seconds)] mmstar_qwen_judge_dtype=${MMSTAR_QWEN_JUDGE_DTYPE}"
echo "[$(date --iso-8601=seconds)] mmstar_qwen_judge_scope=${MMSTAR_QWEN_JUDGE_SCOPE}"
echo "[$(date --iso-8601=seconds)] mmstar_qwen_judge_allow_open_answer=${MMSTAR_QWEN_JUDGE_ALLOW_OPEN_ANSWER}"
echo "[$(date --iso-8601=seconds)] mmstar_qwen_judge_strict=${MMSTAR_QWEN_JUDGE_STRICT}"
echo "[$(date --iso-8601=seconds)] master_port=${MASTER_PORT}"
echo "[$(date --iso-8601=seconds)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import cv2
import accelerate
import flash_attn
import huggingface_hub
import peft
import PIL
import pyarrow
import tokenizers
import torch
import transformers
from vlmeval.config import supported_VLM
from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

def as_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y")

model_name = os.environ["MODEL_NAME"]
kwargs = supported_VLM[model_name].keywords
adapter_path_raw = os.environ.get("adapter_path", "")
adapter_path = Path(adapter_path_raw).resolve() if adapter_path_raw else None
vlm_root = Path(os.environ["VLM_ROOT"]).resolve()
tokenizers_root = Path(os.environ["TOKENIZERS_QWEN25_ROOT"]).resolve()
bootstrap = Path("/scratch/enmingzz/temp/qwen25_bootstrap")

loaded = {
    "cv2": getattr(cv2, "__file__", ""),
    "accelerate": accelerate.__file__,
    "flash_attn": getattr(flash_attn, "__file__", ""),
    "huggingface_hub": huggingface_hub.__file__,
    "peft": peft.__file__,
    "PIL": PIL.__file__,
    "pyarrow": pyarrow.__file__,
    "tokenizers": tokenizers.__file__,
    "torch": torch.__file__,
    "transformers": transformers.__file__,
}
versions = {
    "cv2": cv2.__version__,
    "accelerate": accelerate.__version__,
    "flash_attn": getattr(flash_attn, "__version__", "ok"),
    "huggingface_hub": huggingface_hub.__version__,
    "peft": peft.__version__,
    "PIL": PIL.__version__,
    "pyarrow": pyarrow.__version__,
    "tokenizers": tokenizers.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
}
print("versions", versions)
print("loaded", loaded)
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

assert not any(str(Path(p)).startswith(str(bootstrap)) for p in sys.path if p), sys.path
for package, path in loaded.items():
    assert not str(path).startswith(str(bootstrap)), (package, path)
assert str(transformers.__file__).startswith(str(vlm_root / "transformers" / "src")), transformers.__file__
assert "/scratch/enmingzz/temp/venvs/vsi-official" in str(PIL.__file__), PIL.__file__
assert str(tokenizers.__file__).startswith(str(tokenizers_root)), tokenizers.__file__
assert tokenizers.__version__ == "0.22.2", tokenizers.__version__
assert PIL.__version__ == "9.5.0.post2", PIL.__version__
assert pyarrow.__version__ == "23.0.1", pyarrow.__version__
assert huggingface_hub.__version__ == "0.34.3", huggingface_hub.__version__
assert transformers.__version__ == "4.57.0", transformers.__version__
if adapter_path is not None:
    assert adapter_path.joinpath("adapter_config.json").is_file()
    assert adapter_path.joinpath("adapter_model.safetensors").is_file() or adapter_path.joinpath("adapter_model.bin").is_file()
assert kwargs.get("model_path") == os.environ["model_path"]
if adapter_path is not None:
    assert Path(kwargs.get("adapter_path")).resolve() == adapter_path
else:
    assert kwargs.get("adapter_path") == ""
assert kwargs.get("min_pixels") == 1280 * 28 * 28
assert kwargs.get("max_pixels") == 4096 * 28 * 28
assert kwargs.get("use_custom_prompt") is False
assert kwargs.get("enable_thinking") == as_bool(os.environ["enable_thinking"])
assert kwargs.get("enable_visionzip") == as_bool(os.environ["enable_visionzip"])
assert abs(float(kwargs.get("visionzip_ratio")) - float(os.environ["visionzip_ratio"])) < 1e-12
assert abs(float(kwargs.get("temperature")) - float(os.environ["temperature"])) < 1e-12
assert kwargs.get("use_kv_cache") == as_bool(os.environ["use_kv_cache"])
assert inspect.signature(Qwen2VLChat.__init__).parameters["max_new_tokens"].default == 2048
PY

if [[ "${EVAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[$(date --iso-8601=seconds)] preflight-only finished"
  exit 0
fi

run_vlmeval() {
  local dataset_args=("$@")
  local reuse_args=()
  if [[ "${VLMEVAL_REUSE:-0}" == "1" ]]; then
    reuse_args+=(--reuse)
  fi
  if [[ "${EVAL_NPROC_PER_NODE}" -gt 1 ]]; then
    python -m torch.distributed.run \
      --nnodes=1 \
      --nproc-per-node="${EVAL_NPROC_PER_NODE}" \
      --master-addr=127.0.0.1 \
      --master-port="${MASTER_PORT}" \
      run.py \
        --data "${dataset_args[@]}" \
        --model "${MODEL_NAME}" \
        --mode all \
        --work-dir "${WORK_DIR}" \
        "${reuse_args[@]}"
  else
    python run.py \
      --data "${dataset_args[@]}" \
      --model "${MODEL_NAME}" \
      --mode all \
      --work-dir "${WORK_DIR}" \
      "${reuse_args[@]}"
  fi
}

if [[ "${EVAL_SPLIT_DATASETS:-1}" == "1" && "${#DATASET_ARGS[@]}" -gt 1 ]]; then
  BASE_PORT="${MASTER_PORT}"
  for dataset_idx in "${!DATASET_ARGS[@]}"; do
    export MASTER_PORT="$((BASE_PORT + dataset_idx))"
    echo "[$(date --iso-8601=seconds)] start dataset=${DATASET_ARGS[dataset_idx]} master_port=${MASTER_PORT}"
    run_vlmeval "${DATASET_ARGS[dataset_idx]}"
    echo "[$(date --iso-8601=seconds)] done dataset=${DATASET_ARGS[dataset_idx]}"
  done
else
  run_vlmeval "${DATASET_ARGS[@]}"
fi

echo "[$(date --iso-8601=seconds)] finished"
