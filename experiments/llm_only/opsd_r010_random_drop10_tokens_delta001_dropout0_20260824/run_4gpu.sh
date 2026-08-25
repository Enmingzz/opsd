#!/usr/bin/env bash
set -euo pipefail

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
EXP_ROOT=${OPSD_ROOT}/experiments/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824
OUTPUT_ROOT=/scratch/enmingzz/outputs/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824
CONFIG=${EXP_ROOT}/configs/train_10240.yaml
OUT_DIR=${OUTPUT_ROOT}/run
mkdir -p "${OUT_DIR}"
cd "${OPSD_ROOT}"
sha256sum --check "${EXP_ROOT}/code_checksums.sha256"

source "${PROJECT_ROOT}/env/vsi-official.sh"
export HF_HOME=${HF_HOME:-/home/enmingzz/scratch/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}
export HF_HUB034_ROOT=/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2
export TOKENIZERS_QWEN25_ROOT=/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only
export ARMEN_TRANSFORMERS_SRC=${OPSD_ROOT}/third_party/VLMEvalKit_armen51682/transformers/src
export VISIONZIP_QWEN25VL_ROOT=${PROJECT_ROOT}/VisionZip/Qwen2_5_VL
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
export OPSD_DDP_TIMEOUT_MINUTES=120 OPSD_DDP_STAGGER_LOAD_SECONDS=15 OMP_NUM_THREADS=2
export PYTHONPATH=${PROJECT_ROOT}

python "${EXP_ROOT}/validate_config.py" --config "${CONFIG}" --expected-steps 10240 \
  --output "${OUT_DIR}/preflight.json" --overwrite
if [[ -L "${OUT_DIR}/final" && -f "${OUT_DIR}/final/COMPLETE" ]]; then
  echo "Final checkpoint already complete: ${OUT_DIR}/final"
  exit 0
fi
RESUME_ARGS=()
LATEST_COMPLETE=$(find "${OUT_DIR}/resume_checkpoints" -mindepth 2 -maxdepth 2 \
  -name COMPLETE -printf '%h\n' 2>/dev/null | sort | tail -1 || true)
if [[ -n "${LATEST_COMPLETE}" ]]; then
  echo "Resuming from ${LATEST_COMPLETE}"
  RESUME_ARGS=(--resume_from_checkpoint "${LATEST_COMPLETE}")
fi

printf 'start=%s\nhost=%s\njob=%s\ngit=%s\nconfig=%s\nresume=%s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-none}" \
  "$(git rev-parse HEAD)" "${CONFIG}" "${LATEST_COMPLETE:-none}" \
  | tee -a "${OUT_DIR}/launch_metadata.txt"
sha256sum visionzip_aokvqa/train.py visionzip_aokvqa/native_budget_weighting.py "${CONFIG}" \
  | tee "${OUT_DIR}/input_checksums.sha256"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv | tee "${OUT_DIR}/gpu_start.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 \
  >> "${OUT_DIR}/gpu_monitor.csv" &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

torchrun --nproc-per-node=4 --master-port="$((32000 + RANDOM % 1000))" \
  visionzip_aokvqa/train.py --config "${CONFIG}" --output_dir "${OUT_DIR}" \
  "${RESUME_ARGS[@]}"

cleanup
trap - EXIT
python experiments/scripts/verify_lora_scope.py \
  --config "${CONFIG}" --expected-scope language_decoder_only \
  --adapter-path "${OUT_DIR}/final" \
  --output "${OUT_DIR}/final_scope_verification.json" --overwrite
python "${EXP_ROOT}/audit_training.py" --run-dir "${OUT_DIR}" \
  --expected-steps 10240 --overwrite
echo "end=$(date --iso-8601=seconds)" | tee -a "${OUT_DIR}/launch_metadata.txt"
