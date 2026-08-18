#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd/experiments/llm_only/lcot20k_retrain_dropout0_20260803/eval_progressive20k_qwen5_20260805
OUT_ROOT=/scratch/enmingzz/outputs/llm_only/eval/progressive_opsd20k_dropout0_step20000_merged_qwen5_20260805
EVAL_SCRIPT="${ROOT}/eval_one_case.sbatch"
MERGE_SCRIPT="${ROOT}/merge_after_training.sbatch"
EVAL_MANIFEST="${ROOT}/submission.tsv"
PIPELINE_MANIFEST="${ROOT}/pipeline_submission.tsv"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-4599261}"
ACCOUNT="${ACCOUNT:-aip-btaati}"
MERGE_ACCOUNT="${MERGE_ACCOUNT:-${ACCOUNT}}"
PARTITION="${PARTITION:-gpubase_l40s_b3}"

if [[ -s "${EVAL_MANIFEST}" && "${FORCE_RESUBMIT:-0}" != "1" ]]; then
  echo "Refusing to duplicate an existing campaign: ${EVAL_MANIFEST}" >&2
  echo "Set FORCE_RESUBMIT=1 only after auditing the prior jobs." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs"

merge_raw="$(sbatch \
  --parsable \
  --account="${MERGE_ACCOUNT}" \
  --partition="${PARTITION}" \
  --dependency="afterok:${TRAIN_JOB_ID}" \
  "${MERGE_SCRIPT}")"
merge_job_id="${merge_raw%%;*}"

printf 'submitted_at\tjob_id\taccount\tstage\tdependency\ttime_limit\n' > "${PIPELINE_MANIFEST}"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date --iso-8601=seconds)" "${merge_job_id}" "${MERGE_ACCOUNT}" merge \
  "afterok:${TRAIN_JOB_ID}" 01:00:00 >> "${PIPELINE_MANIFEST}"

printf 'submitted_at\tjob_id\taccount\ttask\tratio\ttime_limit\tdependency\n' > "${EVAL_MANIFEST}"
for task in mme mmstar mathvista mathverse mmmupro; do
  case "${task}" in
    mme) walltime=02:00:00; short=mme ;;
    mmstar) walltime=02:00:00; short=mms ;;
    mathvista) walltime=02:00:00; short=mvi ;;
    mathverse) walltime=02:30:00; short=mve ;;
    mmmupro) walltime=03:00:00; short=mmm ;;
  esac
  for ratio in noprune r010 r020 r030; do
    ratio_short="${ratio#r}"
    [[ "${ratio}" == noprune ]] && ratio_short=np
    job_raw="$(sbatch \
      --parsable \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --dependency="afterok:${merge_job_id}" \
      --time="${walltime}" \
      --job-name="pg20-${short}-${ratio_short}" \
      "${EVAL_SCRIPT}" "${task}" "${ratio}")"
    job_id="${job_raw%%;*}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${job_id}" "${ACCOUNT}" "${task}" "${ratio}" \
      "${walltime}" "afterok:${merge_job_id}" >> "${EVAL_MANIFEST}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${job_id}" "${ACCOUNT}" "eval_${task}_${ratio}" \
      "afterok:${merge_job_id}" "${walltime}" >> "${PIPELINE_MANIFEST}"
    echo "submitted job=${job_id} task=${task} ratio=${ratio} time=${walltime} dependency=afterok:${merge_job_id}"
  done
done

python - "${TRAIN_JOB_ID}" "${merge_job_id}" "${ACCOUNT}" "${MERGE_ACCOUNT}" \
  "${EVAL_MANIFEST}" "${ROOT}/submission_manifest.json" <<'PY'
import csv
import datetime as dt
import json
import sys
from pathlib import Path

train_job, merge_job, account, merge_account, tsv_path, output_path = sys.argv[1:]
rows = list(csv.DictReader(Path(tsv_path).open(encoding="utf-8"), delimiter="\t"))
payload = {
    "status": "submitted",
    "submitted_at": dt.datetime.now().astimezone().isoformat(),
    "training_job_id": train_job,
    "merge_job_id": merge_job,
    "merge_account": merge_account,
    "eval_account": account,
    "checkpoint_step": 20000,
    "checkpoint_load_form": "merged_full_checkpoint",
    "method": "progressive_opsd20k_dropout0",
    "benchmarks": ["MME", "MMStar", "MathVista_MINI", "MathVerse_MINI_Vision_Only", "MMMU_Pro_4c"],
    "ratios": ["noprune", "r010", "r020", "r030"],
    "eval_jobs": rows,
}
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"merge_job_id": merge_job, "eval_jobs": len(rows)}, indent=2))
PY
