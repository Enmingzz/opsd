#!/usr/bin/env bash
# Submit five independent random-fixed-mask pass@64 conditions.

set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd
SWEEP_ROOT="${ROOT}/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_random_fixedmask_qwen25vl7b_t07_p095_20260722"
SBATCH_FILE="${ROOT}/hypothesis_validate/pass_at_k/slurm/run_mmstar_numeric_random_fixedmask_pass64_1gpu.sbatch"
SAMPLE_FILE="${ROOT}/hypothesis_validate/manual_review/mmstar_numeric_open_ended_clean100_seed42/samples.jsonl"
NUMERIC_VALIDATION="${ROOT}/hypothesis_validate/manual_review/mmstar_numeric_open_ended_clean100_seed42/numeric_validation_report.json"
SMOKE_ROOT="${ROOT}/hypothesis_validate/pass_at_k/_smoke/mmstar_numeric_random_fixedmask_r020"
SMOKE_R100="${ROOT}/hypothesis_validate/pass_at_k/_smoke/mmstar_numeric_random_fixedmask_r100/greedy/raw_outputs.jsonl"
REFERENCE_R100="${ROOT}/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_base_qwen25vl7b_t07_p095_20260722/mmstar_open_ended/r100/greedy/raw_outputs.jsonl"
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

jq -e \
    '.status == "passed" and .selected_count == 100 and
     .strict_numeric_reference_count == 100 and
     .all_four_options_strict_numeric_count == 100 and
     .consistent_unit_signature_count == 100 and
     .unique_image_count == 100 and (.failures | length) == 0' \
    "${NUMERIC_VALIDATION}" >/dev/null
bash -n "${SBATCH_FILE}"
python -m py_compile "${ROOT}/hypothesis_validate/scripts/run_random_fixed_mask_rollout_experiment.py"
python -m unittest hypothesis_validate.tests.test_random_fixed_mask_rollout >/dev/null

for mode in greedy sample64; do
    jq -e \
        '.status == "passed" and .failures == 0 and .pruning == "random" and
         .fixed_mask_invariant_all_samples == true and
         .fixed_prefix_invariant_all_samples == true and
         .sample_mask_seed_invariant_all_samples == true and
         .target_token_count_matches_all_records == true and
         (.mask_document_errors | length) == 0' \
        "${SMOKE_ROOT}/${mode}/raw_validation.json" >/dev/null
done
jq -e \
    '.cross_mode_mask_check.counterpart_available == true and
     .cross_mode_mask_check.matched_samples == 1 and
     (.cross_mode_mask_check.mismatched_samples | length) == 0' \
    "${SMOKE_ROOT}/sample64/raw_validation.json" >/dev/null
python - "${SMOKE_R100}" "${REFERENCE_R100}" <<'PY'
import json
import sys
from pathlib import Path

smoke = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[0])
reference = next(
    json.loads(line)
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if json.loads(line)["sample_id"] == smoke["sample_id"]
)
assert smoke["num_full_visual_tokens"] == smoke["num_kept_visual_tokens"]
assert reference["generated_token_ids"][: len(smoke["generated_token_ids"])] == smoke["generated_token_ids"]
PY

mkdir -p "${SWEEP_ROOT}/slurm_logs"
if [[ -s "${MANIFEST}" ]]; then
    echo "Refusing duplicate submission; manifest already exists: ${MANIFEST}" >&2
    exit 1
fi

printf '# samples_sha256=%s\n' "${SAMPLE_SHA256}" >"${TMP_MANIFEST}"
printf '# mask_policy=sample_specific_seed_fixed_across_greedy_and_64_rollouts\n' >>"${TMP_MANIFEST}"
printf 'ratio_tag\tratio\tjob_id\ttime_limit\taccount\tpartition\tworkflow\n' >>"${TMP_MANIFEST}"

for spec in r010:0.1 r020:0.2 r030:0.3 r040:0.4 r100:1.0; do
    ratio_tag="${spec%%:*}"
    ratio="${spec##*:}"
    submit_result=$(sbatch --parsable \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --time="${TIME_LIMIT}" \
        --job-name="mr_${ratio_tag}_p64" \
        --export="ALL,SWEEP_ROOT=${SWEEP_ROOT},RATIO=${ratio},RATIO_TAG=${ratio_tag},MASK_SEED=42,RUN_SUMMARY=0" \
        "${SBATCH_FILE}")
    job_id="${submit_result%%;*}"
    submitted_jobs+=("${job_id}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${ratio_tag}" "${ratio}" "${job_id}" "${TIME_LIMIT}" "${ACCOUNT}" "${PARTITION}" \
        'random_fixedmask_greedy_then_sample64_then_qwen36_safe_judge' >>"${TMP_MANIFEST}"
done

mv "${TMP_MANIFEST}" "${MANIFEST}"
committed=1
echo "Submitted exactly five random-fixed-mask pass@64 jobs. Manifest: ${MANIFEST}"
tail -n +3 "${MANIFEST}" | column -t -s $'\t'
