#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CONFIG_PATH OUTPUT_DIR" >&2
  exit 2
fi

CONFIG_PATH="$(realpath "$1")"
OUT_DIR="$2"
OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
EXP_ROOT="${OPSD_ROOT}/experiments/llm_only/opsd_random_r010_only_dropout0_20260815"
mapfile -t RUN_META < <(python - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
maximum = int(cfg["training"]["max_steps"])
checkpointing = cfg.get("checkpointing", {})
print(str(checkpointing.get("resume_from", "") or ""))
print(maximum)
print(int(checkpointing.get("stop_at_step", maximum) or maximum))
PY
)
RESUME_FROM="${RUN_META[0]}"
MAX_STEPS="${RUN_META[1]}"
STOP_AT_STEP="${RUN_META[2]}"

if [[ -n "${RESUME_FROM}" ]]; then
  [[ -d "${OUT_DIR}" ]] || { echo "Resume output does not exist: ${OUT_DIR}" >&2; exit 1; }
  [[ -f "${RESUME_FROM}/COMPLETE" ]] || { echo "Incomplete resume checkpoint: ${RESUME_FROM}" >&2; exit 1; }
  [[ "$(realpath "${RESUME_FROM}/../..")" == "$(realpath "${OUT_DIR}")" ]] || {
    echo "Resume checkpoint is not owned by output directory: ${RESUME_FROM}" >&2
    exit 1
  }
  [[ ! -e "${OUT_DIR}/final" ]] || { echo "Refusing to resume a completed run: ${OUT_DIR}" >&2; exit 1; }
else
  if [[ -e "${OUT_DIR}" ]]; then
    echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
    exit 1
  fi
  mkdir -p "${OUT_DIR}"
fi
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
export OPSD_DDP_TIMEOUT_MINUTES="${OPSD_DDP_TIMEOUT_MINUTES:-120}"
export OPSD_DDP_STAGGER_LOAD_SECONDS="${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONPATH="${PROJECT_ROOT}"

python "${EXP_ROOT}/validate_config.py" --config "${CONFIG_PATH}" \
  --output "${OUT_DIR}/preflight_step_${STOP_AT_STEP}.json" --overwrite
printf 'start=%s\nhost=%s\njob=%s\ngit=%s\nconfig=%s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-none}" \
  "$(git rev-parse HEAD)" "${CONFIG_PATH}" | tee -a "${OUT_DIR}/launch_metadata.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv \
  | tee "${OUT_DIR}/gpu_start_step_${STOP_AT_STEP}.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 \
  > "${OUT_DIR}/gpu_monitor_step_${STOP_AT_STEP}.csv" &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

torchrun --nproc-per-node=4 --master-port="$((32000 + RANDOM % 1000))" \
  visionzip_aokvqa/train.py --config "${CONFIG_PATH}" --output_dir "${OUT_DIR}"

cleanup
trap - EXIT
if [[ "${STOP_AT_STEP}" -eq "${MAX_STEPS}" ]]; then
  python experiments/scripts/verify_lora_scope.py \
    --config "${CONFIG_PATH}" --expected-scope language_decoder_only \
    --adapter-path "${OUT_DIR}/final" --output "${OUT_DIR}/final_scope_verification.json" --overwrite
  python "${EXP_ROOT}/audit_training.py" \
    --run-dir "${OUT_DIR}" --expected-steps "${MAX_STEPS}" --overwrite
else
  SEGMENT="${OUT_DIR}/segment_complete_step_$(printf '%06d' "${STOP_AT_STEP}").json"
  CHECKPOINT="${OUT_DIR}/resume_checkpoints/step_$(printf '%06d' "${STOP_AT_STEP}")"
  [[ -f "${SEGMENT}" ]] || { echo "Missing segment marker: ${SEGMENT}" >&2; exit 1; }
  [[ -f "${CHECKPOINT}/COMPLETE" ]] || { echo "Missing resumable checkpoint: ${CHECKPOINT}" >&2; exit 1; }
fi
echo "end=$(date --iso-8601=seconds)" | tee -a "${OUT_DIR}/launch_metadata.txt"
