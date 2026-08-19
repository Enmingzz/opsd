#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -eq 1 ]] || { echo "Usage: $0 {trajectory_top20|trajectory_bottom80}" >&2; exit 2; }
VARIANT=$1
case "${VARIANT}" in trajectory_top20|trajectory_bottom80) ;; *) exit 2 ;; esac

ROOT=/project/6101803/enmingzz/opsd
PROJECT=/project/6101803/enmingzz
EXP=${ROOT}/experiments/llm_only/opsd_r010_f_trajectory_partition_delta002_dropout0_20260817
CONFIG=${EXP}/configs/train_${VARIANT}.yaml
OUT=/scratch/enmingzz/outputs/llm_only/opsd_r010_f_trajectory_partition_delta002_dropout0_20260817/${VARIANT}/run
RESUME_CHECKPOINT=""
if [[ -e "${OUT}" ]]; then
  if [[ -f "${OUT}/final/COMPLETE" ]]; then echo "Already complete: ${OUT}"; exit 0; fi
  RESUME_CHECKPOINT=$(find "${OUT}/resume_checkpoints" -mindepth 2 -maxdepth 2 -type f -name COMPLETE -printf '%h\n' 2>/dev/null | sort -V | tail -n 1 || true)
  [[ -n "${RESUME_CHECKPOINT}" ]] || { echo "Partial output has no resumable checkpoint: ${OUT}" >&2; exit 1; }
else
  mkdir -p "${OUT}"
fi

cd "${ROOT}"
source "${PROJECT}/env/vsi-official.sh"
export HF_HOME=/scratch/enmingzz/hf_cache TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export HF_HUB034_ROOT=/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2
export TOKENIZERS_QWEN25_ROOT=/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only
export ARMEN_TRANSFORMERS_SRC=/project/6101803/enmingzz/ckpt_eval_trainenv/VLMEvalKit_armen51682/transformers/src
export VISIONZIP_QWEN25VL_ROOT=${PROJECT}/VisionZip/Qwen2_5_VL
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
export OPSD_DDP_TIMEOUT_MINUTES=120 OPSD_DDP_STAGGER_LOAD_SECONDS=15 OMP_NUM_THREADS=2
export OPSD_PRUNING_METHOD=visionzip PYTHONPATH=${PROJECT}

python "${EXP}/validate_config.py" --config "${CONFIG}" --output "${OUT}/preflight.json" --overwrite
printf 'start=%s\nhost=%s\njob=%s\ngit=%s\nvariant=%s\nconfig=%s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-none}" \
  "$(git rev-parse HEAD)" "${VARIANT}" "${CONFIG}" | tee "${OUT}/launch_metadata.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv | tee "${OUT}/gpu_start.csv"

ARGS=(--config "${CONFIG}" --output_dir "${OUT}")
if [[ -n "${RESUME_CHECKPOINT}" ]]; then ARGS+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}"); fi
torchrun --nproc-per-node=4 --master-port="$((32000 + RANDOM % 1000))" \
  visionzip_aokvqa/train.py "${ARGS[@]}"
python experiments/scripts/verify_lora_scope.py --config "${CONFIG}" \
  --expected-scope language_decoder_only --adapter-path "${OUT}/final" \
  --output "${OUT}/final_scope_verification.json" --overwrite
python experiments/llm_only/opsd_random_r010_only_dropout0_20260815/audit_training.py \
  --run-dir "${OUT}" --expected-steps 10240 --overwrite
echo "end=$(date --iso-8601=seconds)" | tee -a "${OUT}/launch_metadata.txt"
