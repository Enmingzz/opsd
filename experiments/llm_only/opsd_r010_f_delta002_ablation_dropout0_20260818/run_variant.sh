#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -ge 2 && "$#" -le 3 ]] || {
  echo "Usage: $0 {global_f_affine|global_f_curriculum} {smoke|train} [OUTPUT_DIR]" >&2
  exit 2
}

VARIANT="$1"
STAGE="$2"
case "${VARIANT}" in global_f_affine|global_f_curriculum) ;; *) exit 2 ;; esac
case "${STAGE}" in smoke|train) ;; *) exit 2 ;; esac

ROOT=/project/6101803/enmingzz/opsd
PROJECT=/project/6101803/enmingzz
EXP=${ROOT}/experiments/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818
CONFIG=${EXP}/configs/${STAGE}_${VARIANT}.yaml
OUTPUT_ROOT=/scratch/enmingzz/outputs/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818
DEFAULT_OUT=${OUTPUT_ROOT}/${VARIANT}/${STAGE}
[[ "${STAGE}" == train ]] && DEFAULT_OUT=${OUTPUT_ROOT}/${VARIANT}/run
OUT=${3:-${DEFAULT_OUT}}

RESUME_CHECKPOINT=""
if [[ -e "${OUT}" ]]; then
  if [[ "${STAGE}" != train ]]; then
    echo "Refusing to overwrite existing output: ${OUT}" >&2
    exit 1
  fi
  if [[ -f "${OUT}/final/COMPLETE" ]]; then
    echo "Training output is already complete: ${OUT}"
    exit 0
  fi
  if [[ -d "${OUT}/resume_checkpoints" ]]; then
    RESUME_CHECKPOINT=$(find "${OUT}/resume_checkpoints" -mindepth 2 -maxdepth 2 \
      -type f -name COMPLETE -printf '%h\n' 2>/dev/null | sort -V | tail -n 1 || true)
  fi
  if [[ -z "${RESUME_CHECKPOINT}" ]]; then
    echo "Existing output has no complete resumable checkpoint: ${OUT}" >&2
    exit 1
  fi
  echo "Resuming ${VARIANT} from ${RESUME_CHECKPOINT}"
else
  mkdir -p "${OUT}"
fi

cd "${ROOT}"
source "${PROJECT}/env/vsi-official.sh"
export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB034_ROOT=/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2
export TOKENIZERS_QWEN25_ROOT=/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only
export ARMEN_TRANSFORMERS_SRC=/project/6101803/enmingzz/ckpt_eval_trainenv/VLMEvalKit_armen51682/transformers/src
export VISIONZIP_QWEN25VL_ROOT=${PROJECT}/VisionZip/Qwen2_5_VL
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
export OPSD_DDP_TIMEOUT_MINUTES=120 OPSD_DDP_STAGGER_LOAD_SECONDS=15 OMP_NUM_THREADS=2
export OPSD_PRUNING_METHOD=visionzip PYTHONPATH=${PROJECT}

python "${EXP}/validate_config.py" \
  --config "${CONFIG}" --output "${OUT}/preflight.json" --overwrite
printf 'start=%s\nhost=%s\njob=%s\ngit=%s\nvariant=%s\nstage=%s\nconfig=%s\nstudent_ratio=0.10\nprobe_ratio=0.12\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-none}" \
  "$(git rev-parse HEAD)" "${VARIANT}" "${STAGE}" "${CONFIG}" \
  | tee "${OUT}/launch_metadata.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv \
  | tee "${OUT}/gpu_start.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 \
  > "${OUT}/gpu_monitor.csv" &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

if [[ "${STAGE}" == smoke ]]; then
  python visionzip_aokvqa/train.py --config "${CONFIG}" --output_dir "${OUT}"
else
  TRAIN_ARGS=(--config "${CONFIG}" --output_dir "${OUT}")
  [[ -n "${RESUME_CHECKPOINT}" ]] && TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
  torchrun --nproc-per-node=4 --master-port="$((32000 + RANDOM % 1000))" \
    visionzip_aokvqa/train.py "${TRAIN_ARGS[@]}"
fi
cleanup
trap - EXIT

python experiments/scripts/verify_lora_scope.py \
  --config "${CONFIG}" --expected-scope language_decoder_only \
  --adapter-path "${OUT}/final" \
  --output "${OUT}/final_scope_verification.json" --overwrite
if [[ "${STAGE}" == train ]]; then
  python experiments/llm_only/opsd_random_r010_only_dropout0_20260815/audit_training.py \
    --run-dir "${OUT}" --expected-steps 10240 --overwrite
fi
echo "end=$(date --iso-8601=seconds)" | tee -a "${OUT}/launch_metadata.txt"
