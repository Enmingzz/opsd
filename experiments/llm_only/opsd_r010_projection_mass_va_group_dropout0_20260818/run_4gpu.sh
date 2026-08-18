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
EXP_ROOT="${OPSD_ROOT}/experiments/llm_only/opsd_r010_projection_mass_va_group_dropout0_20260818"
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
export OPSD_DDP_TIMEOUT_MINUTES="${OPSD_DDP_TIMEOUT_MINUTES:-120}"
export OPSD_DDP_STAGGER_LOAD_SECONDS="${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONPATH="${PROJECT_ROOT}"

python "${EXP_ROOT}/validate_config.py" --config "${CONFIG_PATH}" \
  --expected-steps 10240 --output "${OUT_DIR}/preflight.json" --overwrite

if [[ -L "${OUT_DIR}/final" && -f "${OUT_DIR}/final/COMPLETE" ]]; then
  echo "Final checkpoint is already complete: ${OUT_DIR}/final"
  exit 0
fi

RESUME_ARGS=()
LATEST_COMPLETE="$(find "${OUT_DIR}/resume_checkpoints" -mindepth 2 -maxdepth 2 \
  -name COMPLETE -printf '%h\n' 2>/dev/null | sort | tail -1 || true)"
if [[ -n "${LATEST_COMPLETE}" ]]; then
  echo "Resuming from ${LATEST_COMPLETE}"
  RESUME_ARGS=(--resume_from_checkpoint "${LATEST_COMPLETE}")
fi

printf 'start=%s\nhost=%s\njob=%s\ngit=%s\nconfig=%s\nresume=%s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-none}" \
  "$(git rev-parse HEAD)" "${CONFIG_PATH}" "${LATEST_COMPLETE:-none}" \
  | tee -a "${OUT_DIR}/launch_metadata.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv \
  | tee "${OUT_DIR}/gpu_start.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 \
  >> "${OUT_DIR}/gpu_monitor.csv" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

torchrun --nproc-per-node=4 --master-port="$((32000 + RANDOM % 1000))" \
  visionzip_aokvqa/train.py --config "${CONFIG_PATH}" --output_dir "${OUT_DIR}" \
  "${RESUME_ARGS[@]}"

cleanup
trap - EXIT
python experiments/scripts/verify_lora_scope.py \
  --config "${CONFIG_PATH}" --expected-scope language_decoder_only \
  --adapter-path "${OUT_DIR}/final" \
  --output "${OUT_DIR}/final_scope_verification.json" --overwrite
python experiments/llm_only/opsd_random_r010_only_dropout0_20260815/audit_training.py \
  --run-dir "${OUT_DIR}" --expected-steps 10240 --overwrite
echo "end=$(date --iso-8601=seconds)" | tee -a "${OUT_DIR}/launch_metadata.txt"
