#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd/hypothesis_validate/opsd_checkpoint_kl_trajectory
SBATCH=${ROOT}/slurm/run_sft_checkpoint_max1024_30m.sbatch
MANIFEST=${ROOT}/outputs/slurm_jobs_sft_metric_noocr_max1024_20260729.tsv
CHECKPOINT_ROOT=/scratch/enmingzz/outputs/llm_only/checkpoints/llm_only_random_decontam_v1_20260713/sft

mkdir -p "${ROOT}/outputs"
printf 'checkpoint_label\tcheckpoint_step\tadapter_path\tjob_id\tsubmitted_at\n' > "${MANIFEST}"

submit() {
  local label="$1"
  local step="$2"
  local adapter="$3"
  local job_id
  job_id=$(sbatch --parsable --job-name="sft-kl1024-${label}" "${SBATCH}" "${label}" "${adapter}")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${step}" "${adapter}" "${job_id}" "$(date --iso-8601=seconds)" >> "${MANIFEST}"
  echo "${label}: ${job_id}"
}

submit step_0 0 __BASE__
for step in 1024 2048 3072 4096 5120 6144 7168 8192 9216; do
  submit "step_${step}" "${step}" "${CHECKPOINT_ROOT}/step_${step}"
done
submit step_9984 9984 "${CHECKPOINT_ROOT}/final"

echo "manifest=${MANIFEST}"
