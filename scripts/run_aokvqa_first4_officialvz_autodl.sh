#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT="${BASE_ROOT:-/root/autodl-tmp/opsd_eval}"
OPSD_ROOT="${OPSD_ROOT:-${BASE_ROOT}/opsd}"
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-${BASE_ROOT}/VLMEvalKit}"
OUT_ROOT="${OUT_ROOT:-${BASE_ROOT}/outputs/visionzip_aokvqa_reasoning}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

cd "${OPSD_ROOT}"

if [[ -d "${CONDA_BIN}" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi
export HF_HOME="${HF_HOME:-${BASE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${BASE_ROOT}:${VLMEVALKIT_ROOT}:${PYTHONPATH:-}"
export VISIONZIP_QWEN25VL_ROOT="${VISIONZIP_QWEN25VL_ROOT:-${BASE_ROOT}/VisionZip/Qwen2_5_VL}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OPSD_DDP_TIMEOUT_MINUTES=${OPSD_DDP_TIMEOUT_MINUTES:-120}
export OPSD_DDP_STAGGER_LOAD_SECONDS=${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}

RUN_GROUP=${RUN_GROUP:-aokvqa_first4_officialvz_$(date +%Y%m%d_%H%M%S)}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29721}
BASE_OUT=${OUT_ROOT}/checkpoints/${RUN_GROUP}
LOGDIR=${OUT_ROOT}/logs/train/${RUN_GROUP}
GENERATED_CONFIG_DIR=${LOGDIR}/generated_configs

mkdir -p "${BASE_OUT}" "${LOGDIR}" "${GENERATED_CONFIG_DIR}"
printf '%s\n' "${RUN_GROUP}" > "${OUT_ROOT}/latest_aokvqa_first4_officialvz_train_run.txt"

{
  printf 'stage\tname\tconfig\n'
  printf '01\tsft_official_gbs8\tconfigs/visionzip_aokvqa/aokvqa_sft_gbs8.yaml\n'
  printf '02\tepic_tcd_full_teacher_official_gbs8\tconfigs/visionzip_aokvqa/aokvqa_epic_tcd_gbs8.yaml\n'
  printf '03\tema_opsd_nogt_official_gbs8\tconfigs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_gbs8.yaml\n'
  printf '04\tfreeze_sft_teacher_opsd_nogt_official_gbs8\t%s/aokvqa_opsd_sft_teacher_freeze_nogt_gbs8.yaml\n' "${GENERATED_CONFIG_DIR}"
} > "${LOGDIR}/manifest.tsv"

run_stage() {
  local stage_id="$1"
  local name="$2"
  local config="$3"
  local port="$4"
  local out_dir="${BASE_OUT}/${stage_id}_${name}"
  local log_path="${LOGDIR}/${stage_id}_${name}.log"

  echo "[$(date --iso-8601=seconds)] start ${stage_id}_${name}" | tee -a "${LOGDIR}/sequence.log"
  torchrun \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --master-port="${port}" \
    visionzip_aokvqa/train.py \
    --config "${config}" \
    --output_dir "${out_dir}" \
    2>&1 | tee "${log_path}"
  echo "[$(date --iso-8601=seconds)] done ${stage_id}_${name}" | tee -a "${LOGDIR}/sequence.log"
}

make_freeze_sft_teacher_config() {
  local template="configs/visionzip_aokvqa/aokvqa_opsd_sft_teacher_freeze_nogt_gbs8.yaml"
  local config="${GENERATED_CONFIG_DIR}/aokvqa_opsd_sft_teacher_freeze_nogt_gbs8.yaml"
  local teacher_adapter="${BASE_OUT}/01_sft_official_gbs8/final"

  python - "${template}" "${config}" "${teacher_adapter}" <<'PY'
import sys
from pathlib import Path

import yaml

template = Path(sys.argv[1])
out = Path(sys.argv[2])
teacher_adapter = sys.argv[3]
cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
cfg.setdefault("opsd", {})["teacher_adapter_path"] = teacher_adapter
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
print(out)
PY
}

run_stage "01" "sft_official_gbs8" "configs/visionzip_aokvqa/aokvqa_sft_gbs8.yaml" "${MASTER_PORT_BASE}"
run_stage "02" "epic_tcd_full_teacher_official_gbs8" "configs/visionzip_aokvqa/aokvqa_epic_tcd_gbs8.yaml" "$((MASTER_PORT_BASE + 1))"
run_stage "03" "ema_opsd_nogt_official_gbs8" "configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_gbs8.yaml" "$((MASTER_PORT_BASE + 2))"
FREEZE_SFT_CONFIG=$(make_freeze_sft_teacher_config)
run_stage "04" "freeze_sft_teacher_opsd_nogt_official_gbs8" "${FREEZE_SFT_CONFIG}" "$((MASTER_PORT_BASE + 3))"
