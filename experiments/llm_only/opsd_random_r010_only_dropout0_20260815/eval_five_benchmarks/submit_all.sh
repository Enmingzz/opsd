#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
ACCOUNT="${ACCOUNT:-aip-btaati}"
PARTITION="${PARTITION:-gpubase_l40s_b3}"
MANIFEST="${MANIFEST:-${OUT_ROOT}/submission.tsv}"
mkdir -p "$(dirname "${MANIFEST}")"
printf 'submitted_at\tjob_id\taccount\ttask\tratio\ttime_limit\n' > "${MANIFEST}"

for task in mme mmstar mathvista mathverse mmmupro; do
  case "${task}" in
    mme|mmstar) walltime=04:00:00 ;;
    mathvista|mathverse) walltime=05:00:00 ;;
    mmmupro) walltime=06:00:00 ;;
  esac
  for ratio in noprune r010 r020 r030; do
    job_id="$(sbatch --parsable \
      --account="${ACCOUNT}" --partition="${PARTITION}" --time="${walltime}" \
      --export="ALL,CHECKPOINT_STEP=${CHECKPOINT_STEP},TRAIN_RUN_ROOT=${TRAIN_RUN_ROOT},TRAIN_CONFIG=${TRAIN_CONFIG},ADAPTER_PATH=${ADAPTER_PATH},MERGED_MODEL=${MERGED_MODEL},OUT_ROOT=${OUT_ROOT}" \
      --job-name="r10-${task:0:3}-${ratio#r}" \
      "${HERE}/eval_one_case.sbatch" "${task}" "${ratio}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${job_id}" "${ACCOUNT}" "${task}" "${ratio}" "${walltime}" \
      >> "${MANIFEST}"
  done
done
echo "submission_manifest=${MANIFEST}"
