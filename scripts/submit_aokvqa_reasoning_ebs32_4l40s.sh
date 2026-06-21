#!/usr/bin/env bash
set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/project/6101803/enmingzz/opsd}"
OUT_ROOT="${OUT_ROOT:-/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning}"
VENV_ROOT="${VENV_ROOT:-/scratch/enmingzz/temp/venvs/vsi-official}"
ACCOUNT="${ACCOUNT:-aip-gigor}"
PARTITION="${PARTITION:-gpubase_l40s_b4}"
TIME_LIMIT="${TIME_LIMIT:-16:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-400G}"
DRY_RUN="${DRY_RUN:-0}"
RUN_GROUP="${RUN_GROUP:-aokvqa_hires_reasoning_ebs32_4l40s_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${OUT_ROOT}/checkpoints/${RUN_GROUP}"
LOGDIR="${OUT_ROOT}/logs/train/${RUN_GROUP}"
GENERATED_CONFIG_DIR="${LOGDIR}/generated_configs"
SBATCH_SCRIPT="${OPSD_ROOT}/slurm_jobs/train_aokvqa_reasoning_ebs32_4l40s.sbatch"

cd "${OPSD_ROOT}"
mkdir -p "${BASE_OUT}" "${LOGDIR}" "${GENERATED_CONFIG_DIR}"
printf '%s\n' "${RUN_GROUP}" > "${OUT_ROOT}/latest_aokvqa_reasoning_ebs32_4l40s_run.txt"

make_teacher_config() {
  local template="$1"
  local output="$2"
  local teacher_adapter="$3"

  "${VENV_ROOT}/bin/python" - "${template}" "${output}" "${teacher_adapter}" <<'PY'
import sys
from pathlib import Path

import yaml

template = Path(sys.argv[1])
output = Path(sys.argv[2])
teacher_adapter = sys.argv[3]
cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
cfg.setdefault("opsd", {})["teacher_adapter_path"] = teacher_adapter
output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
print(output)
PY
}

submit_stage() {
  local stage_name="$1"
  local config_path="$2"
  local dependency="${3:-}"
  local job_name="aok_${stage_name}"
  local args=(
    --parsable
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --gres="gpu:l40s:4"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM}"
    --time="${TIME_LIMIT}"
    --job-name="${job_name:0:50}"
    --output="${LOGDIR}/slurm-${stage_name}-%j.out"
    --error="${LOGDIR}/slurm-${stage_name}-%j.err"
  )
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency="${dependency}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf 'DRY_RUN sbatch'
      printf ' %q' "${args[@]}" "${SBATCH_SCRIPT}" "${config_path}" "${stage_name}" "${RUN_GROUP}"
      printf '\n'
    } >&2
    local fake_job_id="dry_${stage_name}"
    printf '%s\t%s\t%s\t%s\n' "${stage_name}" "${fake_job_id}" "${config_path}" "${dependency}" >> "${LOGDIR}/submitted_jobs.tsv"
    printf '%s\n' "${fake_job_id}"
    return
  fi
  local raw
  raw=$(sbatch "${args[@]}" "${SBATCH_SCRIPT}" "${config_path}" "${stage_name}" "${RUN_GROUP}")
  local job_id="${raw%%;*}"
  printf '%s\t%s\t%s\t%s\n' "${stage_name}" "${job_id}" "${config_path}" "${dependency}" >> "${LOGDIR}/submitted_jobs.tsv"
  printf '%s\n' "${job_id}"
}

SFT_STAGE="04_sft_reasoning_ebs32"
SFT_CONFIG="configs/visionzip_aokvqa/aokvqa_sft_reasoning_ebs32_4l40s.yaml"
SFT_OUT="${BASE_OUT}/${SFT_STAGE}/final"

FREEZE_CONFIG="${GENERATED_CONFIG_DIR}/aokvqa_opsd_nogt_freeze_sft_teacher_reasoning_ebs32_4l40s.yaml"
SFT_EMA_CONFIG="${GENERATED_CONFIG_DIR}/aokvqa_opsd_nogt_sft_ema_teacher_reasoning_ebs32_4l40s.yaml"
make_teacher_config \
  "configs/visionzip_aokvqa/aokvqa_opsd_nogt_freeze_sft_teacher_reasoning_ebs32_4l40s.yaml" \
  "${FREEZE_CONFIG}" \
  "${SFT_OUT}" >/dev/null
make_teacher_config \
  "configs/visionzip_aokvqa/aokvqa_opsd_nogt_sft_ema_teacher_reasoning_ebs32_4l40s.yaml" \
  "${SFT_EMA_CONFIG}" \
  "${SFT_OUT}" >/dev/null

{
  printf 'stage\tjob_id\tconfig\tdependency\n'
} > "${LOGDIR}/submitted_jobs.tsv"

SFT_JOB=$(submit_stage "${SFT_STAGE}" "${SFT_CONFIG}")
OPSD_EMA_JOB=$(submit_stage "01_opsd_ema_teacher_nogt_ebs32" "configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_reasoning_ebs32_4l40s.yaml")
FREEZE_JOB=$(submit_stage "02_freeze_sft_teacher_opsd_nogt_ebs32" "${FREEZE_CONFIG}" "afterok:${SFT_JOB}")
SFT_EMA_JOB=$(submit_stage "03_sft_ema_teacher_opsd_nogt_ebs32" "${SFT_EMA_CONFIG}" "afterok:${SFT_JOB}")
EPIC_JOB=$(submit_stage "05_epic_tcd_reasoning_ebs32" "configs/visionzip_aokvqa/aokvqa_epic_tcd_reasoning_ebs32_4l40s.yaml")

cat <<EOF
RUN_GROUP=${RUN_GROUP}
LOGDIR=${LOGDIR}
BASE_OUT=${BASE_OUT}
SFT_JOB=${SFT_JOB}
OPSD_EMA_JOB=${OPSD_EMA_JOB}
FREEZE_JOB=${FREEZE_JOB}
SFT_EMA_JOB=${SFT_EMA_JOB}
EPIC_JOB=${EPIC_JOB}
EOF
