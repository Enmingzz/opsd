#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd/hypothesis_validate/opsd_checkpoint_kl_trajectory
CONFIG_ROOT=${ROOT}/configs/ratio_pairs
SBATCH=${ROOT}/slurm/run_ratio_pair_checkpoint_max1024_30m.sbatch
MANIFEST=${ROOT}/outputs/slurm_jobs_ratio_pairs_88_20260729.tsv

mkdir -p "${ROOT}/outputs"
printf 'pair\tmethod\tcheckpoint_label\tcheckpoint_step\tconfig\tadapter_path\tjob_id\tsubmitted_at\n' > "${MANIFEST}"

submit() {
  local pair="$1"
  local method="$2"
  local label="$3"
  local step="$4"
  local config="$5"
  local adapter="$6"
  local job_id
  job_id=$(sbatch --parsable --job-name="kl-${pair}-${method}-${step}" \
    "${SBATCH}" "${config}" "${label}" "${adapter}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${pair}" "${method}" "${label}" "${step}" "${config}" "${adapter}" \
    "${job_id}" "$(date --iso-8601=seconds)" >> "${MANIFEST}"
  echo "${pair} ${method} ${label}: ${job_id}"
}

for pair in r015_r0175 r015_r020 r0175_r020 r020_r0225; do
  for method in opsd sft; do
    config=${CONFIG_ROOT}/config_${method}_${pair}_max1024.json
    submit "${pair}" "${method}" step_0 0 "${config}" __BASE__
    for step in 1024 2048 3072 4096 5120 6144 7168 8192 9216; do
      submit "${pair}" "${method}" "step_${step}" "${step}" "${config}" "step_${step}"
    done
    submit "${pair}" "${method}" step_9984 9984 "${config}" final
  done
done

row_count=$(( $(wc -l < "${MANIFEST}") - 1 ))
if [[ "${row_count}" -ne 88 ]]; then
  echo "Expected 88 submitted jobs, found ${row_count}" >&2
  exit 1
fi
echo "manifest=${MANIFEST}"
echo "submitted_jobs=${row_count}"
