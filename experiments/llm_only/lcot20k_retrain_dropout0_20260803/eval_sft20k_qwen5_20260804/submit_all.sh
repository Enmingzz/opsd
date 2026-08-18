#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd/experiments/llm_only/lcot20k_retrain_dropout0_20260803/eval_sft20k_qwen5_20260804
SCRIPT="${ROOT}/eval_one_case.sbatch"
MANIFEST="${ROOT}/submission.tsv"
ACCOUNT="${ACCOUNT:-aip-gigor}"
PARTITION="${PARTITION:-gpubase_l40s_b3}"

printf 'submitted_at\tjob_id\taccount\ttask\tratio\ttime_limit\n' > "${MANIFEST}"
for task in mme mmstar mathvista mathverse mmmupro; do
  case "${task}" in
    mme) walltime=04:00:00; short=mme ;;
    mmstar) walltime=04:00:00; short=mms ;;
    mathvista) walltime=04:00:00; short=mvi ;;
    mathverse) walltime=05:00:00; short=mve ;;
    mmmupro) walltime=06:00:00; short=mmm ;;
  esac
  for ratio in noprune r010 r020 r030; do
    job_id="$(sbatch \
      --parsable \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --time="${walltime}" \
      --job-name="sf20-${short}-${ratio#r}" \
      "${SCRIPT}" "${task}" "${ratio}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${job_id}" "${ACCOUNT}" "${task}" "${ratio}" "${walltime}" \
      >> "${MANIFEST}"
    echo "submitted job=${job_id} task=${task} ratio=${ratio} time=${walltime}"
  done
done
