#!/usr/bin/env bash
set -euo pipefail

# Eval the current official-VisionZip first4 run with VLMEvalKit on AutoDL.
# Defaults: 12 workers over 4 GPUs, three eval workers per GPU.

BASE_ROOT="${BASE_ROOT:-/root/autodl-tmp/opsd_eval}"
OPSD_ROOT="${OPSD_ROOT:-${BASE_ROOT}/opsd}"
OUT_ROOT="${OUT_ROOT:-${BASE_ROOT}/outputs/visionzip_aokvqa_reasoning}"
RUN_ID="${RUN_ID:-aokvqa_first4_officialvz_20260610_154040}"
CKPT_ROOT="${CKPT_ROOT:-${OUT_ROOT}/checkpoints/${RUN_ID}}"
LAUNCHER="${LAUNCHER:-${OPSD_ROOT}/scripts/run_vlmevalkit_autodl_12worker.sh}"

DATASETS="${DATASETS:-MME MMStar POPE}"
RATIOS="${RATIOS:-r030 r010 r005}"
NPROC_PER_NODE="${NPROC_PER_NODE:-12}"
RUN_NAME="${RUN_NAME:-first4_officialvz_mme_mmstar_pope_$(date +%Y%m%d_%H%M%S)}"
WAIT_FOR_FINALS="${WAIT_FOR_FINALS:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
WAIT_TRIES="${WAIT_TRIES:-720}"

required_finals=(
  "${CKPT_ROOT}/01_sft_official_gbs8/final"
  "${CKPT_ROOT}/02_epic_tcd_full_teacher_official_gbs8/final"
  "${CKPT_ROOT}/03_ema_opsd_nogt_official_gbs8/final"
  "${CKPT_ROOT}/04_freeze_sft_teacher_opsd_nogt_official_gbs8/final"
)

if [[ "${WAIT_FOR_FINALS}" == "1" ]]; then
  for final_dir in "${required_finals[@]}"; do
    tries=0
    while [[ ! -d "${final_dir}" ]]; do
      tries=$((tries + 1))
      if (( tries > WAIT_TRIES )); then
        echo "Timed out waiting for ${final_dir}" >&2
        exit 1
      fi
      echo "Waiting for ${final_dir} (${tries}/${WAIT_TRIES})"
      sleep "${WAIT_SECONDS}"
    done
  done
else
  for final_dir in "${required_finals[@]}"; do
    [[ -d "${final_dir}" ]] || { echo "Missing ${final_dir}" >&2; exit 1; }
  done
fi

models=(opsd_qwen25vl_official_7b_flashattn2_mnt32)

for ratio in ${RATIOS}; do
  models+=("opsd_qwen25vl_visionzip_flashattn2_${ratio}")
done

for adapter in sft epic ema_nogt freeze_sftteacher_nogt; do
  for ratio in ${RATIOS}; do
    models+=("opsd_first4_officialvz_${adapter}_${ratio}")
  done
done

MODELS="${models[*]}" \
DATASETS="${DATASETS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
RUN_NAME="${RUN_NAME}" \
bash "${LAUNCHER}"
