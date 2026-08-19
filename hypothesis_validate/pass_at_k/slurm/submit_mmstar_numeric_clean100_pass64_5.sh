#!/usr/bin/env bash
# Submit five independent numeric-clean100 pass@64 conditions.

set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd
SWEEP_ROOT="${ROOT}/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_base_qwen25vl7b_t07_p095_20260722"
SBATCH_FILE="${ROOT}/hypothesis_validate/pass_at_k/slurm/run_mmstar_numeric_clean100_pass64_1gpu.sbatch"
SAMPLE_FILE="${ROOT}/hypothesis_validate/manual_review/mmstar_numeric_open_ended_clean100_seed42/samples.jsonl"
NUMERIC_VALIDATION="${ROOT}/hypothesis_validate/manual_review/mmstar_numeric_open_ended_clean100_seed42/numeric_validation_report.json"
SMOKE_VALIDATION="${ROOT}/hypothesis_validate/pass_at_k/_smoke/mmstar_numeric_r020_sample2_n1/raw_validation.json"
ACCOUNT="${ACCOUNT:-aip-gigor}"
PARTITION="${PARTITION:-gpubase_l40s_b3}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MANIFEST="${SWEEP_ROOT}/submission_manifest.tsv"
TMP_MANIFEST="${MANIFEST}.tmp.$$"
SAMPLE_SHA256=$(sha256sum "${SAMPLE_FILE}" | awk '{print $1}')
submitted_jobs=()
committed=0

cleanup() {
    if [[ "${committed}" -eq 0 && "${#submitted_jobs[@]}" -gt 0 ]]; then
        echo "Submission failed; cancelling ${#submitted_jobs[@]} partially submitted jobs." >&2
        scancel "${submitted_jobs[@]}" || true
    fi
    rm -f "${TMP_MANIFEST}"
}
trap cleanup EXIT INT TERM

if [[ ! -s "${SAMPLE_FILE}" || ! -s "${SBATCH_FILE}" ]]; then
    echo "Required sample or Slurm file is missing." >&2
    exit 1
fi
jq -e '.status == "passed" and .selected_count == 100 and .strict_numeric_reference_count == 100 and .all_four_options_strict_numeric_count == 100' "${NUMERIC_VALIDATION}" >/dev/null
jq -e '.status == "passed" and .failures == 0 and .observed_records == 2' "${SMOKE_VALIDATION}" >/dev/null
bash -n "${SBATCH_FILE}"

mkdir -p "${SWEEP_ROOT}/slurm_logs"
if [[ -s "${MANIFEST}" ]]; then
    echo "Refusing duplicate submission; manifest already exists: ${MANIFEST}" >&2
    exit 1
fi

printf '# samples_sha256=%s\n' "${SAMPLE_SHA256}" >"${TMP_MANIFEST}"
printf 'ratio_tag\tratio\tjob_id\ttime_limit\taccount\tpartition\tworkflow\n' >>"${TMP_MANIFEST}"

for spec in r010:0.1 r020:0.2 r030:0.3 r040:0.4 r100:1.0; do
    ratio_tag="${spec%%:*}"
    ratio="${spec##*:}"
    submit_result=$(sbatch --parsable \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --time="${TIME_LIMIT}" \
        --job-name="mn_${ratio_tag}_p64" \
        --export="ALL,SWEEP_ROOT=${SWEEP_ROOT},RATIO=${ratio},RATIO_TAG=${ratio_tag}" \
        "${SBATCH_FILE}")
    job_id="${submit_result%%;*}"
    submitted_jobs+=("${job_id}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${ratio_tag}" "${ratio}" "${job_id}" "${TIME_LIMIT}" "${ACCOUNT}" "${PARTITION}" \
        'greedy_then_sample64_then_qwen36_safe_judge' >>"${TMP_MANIFEST}"
done

mv "${TMP_MANIFEST}" "${MANIFEST}"
committed=1
echo "Submitted exactly five numeric-clean100 pass@64 jobs. Manifest: ${MANIFEST}"
tail -n +2 "${MANIFEST}" | column -t -s $'\t'
