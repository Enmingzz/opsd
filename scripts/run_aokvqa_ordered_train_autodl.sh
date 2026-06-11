#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/opsd_eval/opsd

export PATH=/root/miniconda3/bin:${PATH}
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/root/autodl-tmp/opsd_eval:/root/autodl-tmp/opsd_eval/VLMEvalKit:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OPSD_DDP_TIMEOUT_MINUTES=${OPSD_DDP_TIMEOUT_MINUTES:-120}
export OPSD_DDP_STAGGER_LOAD_SECONDS=${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}

RUN_GROUP=${RUN_GROUP:-aokvqa_ordered_$(date +%Y%m%d_%H%M%S)}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29611}
BASE_ROOT=/root/autodl-tmp/opsd_eval/outputs/visionzip_aokvqa_reasoning
BASE_OUT=${BASE_ROOT}/checkpoints/${RUN_GROUP}
LOGDIR=${BASE_ROOT}/logs/train/${RUN_GROUP}

mkdir -p "${BASE_OUT}" "${LOGDIR}"
printf '%s\n' "${RUN_GROUP}" > "${BASE_ROOT}/latest_aokvqa_ordered_train_run.txt"

run_stage() {
  local stage_id="$1"
  local name="$2"
  local config="$3"
  local port="$4"
  local out_dir="${BASE_OUT}/${stage_id}_${name}"
  local log_path="${LOGDIR}/${stage_id}_${name}.log"

  echo "[$(date --iso-8601=seconds)] start ${stage_id}_${name}" | tee -a "${LOGDIR}/sequence.log"
  torchrun \
    --nproc-per-node=4 \
    --master-port="${port}" \
    visionzip_aokvqa/train.py \
    --config "${config}" \
    --output_dir "${out_dir}" \
    2>&1 | tee "${log_path}"
  echo "[$(date --iso-8601=seconds)] done ${stage_id}_${name}" | tee -a "${LOGDIR}/sequence.log"
}

run_stage "01" "opsd_nogt_ema_decay9999_gbs8" "configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_gbs8.yaml" "${MASTER_PORT_BASE}"
run_stage "02" "epic_tcd_full_teacher_gbs8" "configs/visionzip_aokvqa/aokvqa_epic_tcd_gbs8.yaml" "$((MASTER_PORT_BASE + 1))"
run_stage "03" "sft_gbs8" "configs/visionzip_aokvqa/aokvqa_sft_gbs8.yaml" "$((MASTER_PORT_BASE + 2))"
run_stage "04" "opsd_nogt_freeze_teacher_gbs8" "configs/visionzip_aokvqa/aokvqa_opsd_nogt_freeze_teacher_gbs8.yaml" "$((MASTER_PORT_BASE + 3))"
