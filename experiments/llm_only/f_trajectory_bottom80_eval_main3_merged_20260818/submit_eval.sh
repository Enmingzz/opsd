#!/usr/bin/env bash

set -euo pipefail
ROOT=/project/6101803/enmingzz/opsd/experiments/llm_only/f_trajectory_bottom80_eval_main3_merged_20260818
source /project/6101803/enmingzz/ckpt_eval_trainenv/env_train_runtime.sh
source "${ROOT}/common.sh"
set_method ft80
ACCOUNT="${ACCOUNT:-aip-btaati}"
PARTITION="${PARTITION:-gpubase_l40s_b3}"
TSV="${ROOT}/submission.tsv"
JSON="${ROOT}/submission.json"
[[ ! -e "${TSV}" && ! -e "${JSON}" ]] || { echo "Campaign already submitted" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs"
bash -n "${ROOT}/common.sh" "${ROOT}/eval_one_case.sbatch" "${ROOT}/postprocess_mathvista_strict.sbatch"
python -m py_compile "${ROOT}/validate_artifact.py" "${ROOT}/collect_case.py"
python "${ROOT}/validate_artifact.py" merged --adapter "${ADAPTER_PATH}" \
  --config "${TRAIN_CONFIG}" --run-root "${RUN_ROOT}" --method "${METHOD_LONG}" \
  --objective "${TRAINING_OBJECTIVE}" --training-job-id "${TRAIN_JOB_ID}" \
  --merged "${MERGED_MODEL}" >/dev/null

printf 'job_id\tstage\tmethod\ttask\tratio\taccount\tpartition\ttime_limit\tdependency\tsubmitted_at\n' > "${TSV}"
all_ids=()
post_ids=()
for task in mme mmstar mathvista; do
  for ratio in noprune r010 r020 r030; do
    short_task="${task/mmstar/mms}"; short_task="${short_task/mathvista/mv}"
    short_ratio="${ratio#r}"; [[ "${ratio}" == "noprune" ]] && short_ratio=np
    eval_id="$(sbatch --parsable --account="${ACCOUNT}" --partition="${PARTITION}" \
      --time=04:00:00 --job-name="ft80-${short_task}-${short_ratio}" \
      "${ROOT}/eval_one_case.sbatch" ft80 "${task}" "${ratio}")"
    eval_id="${eval_id%%;*}"
    all_ids+=("${eval_id}")
    printf '%s\tinference\tft80\t%s\t%s\t%s\t%s\t04:00:00\t-\t%s\n' \
      "${eval_id}" "${task}" "${ratio}" "${ACCOUNT}" "${PARTITION}" \
      "$(date --iso-8601=seconds)" >> "${TSV}"
    if [[ "${task}" == "mathvista" ]]; then
      post_id="$(sbatch --parsable --account="${ACCOUNT}" --partition="${PARTITION}" \
        --time=00:30:00 --dependency="afterok:${eval_id}" \
        --job-name="ft80-mvgt-${short_ratio}" \
        "${ROOT}/postprocess_mathvista_strict.sbatch" ft80 "${ratio}" "${eval_id}")"
      post_id="${post_id%%;*}"
      all_ids+=("${post_id}")
      post_ids+=("${post_id}")
      printf '%s\tmathvista_strict_gt\tft80\tmathvista\t%s\t%s\t%s\t00:30:00\tafterok:%s\t%s\n' \
        "${post_id}" "${ratio}" "${ACCOUNT}" "${PARTITION}" "${eval_id}" \
        "$(date --iso-8601=seconds)" >> "${TSV}"
    fi
  done
done

python - "${JSON}" "${ACCOUNT}" "${PARTITION}" "${all_ids[*]}" "${post_ids[*]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
account, partition, all_ids, post_ids = sys.argv[2:]
payload = {
    "status": "submitted",
    "submitted_at": datetime.now().astimezone().isoformat(),
    "account": account,
    "partition": partition,
    "method": "trajectory_bottom80_delta002",
    "datasets": ["MME", "MMStar", "MathVista_MINI"],
    "ratios": ["noprune", "r010", "r020", "r030"],
    "checkpoint_load_form": "merged_full_checkpoint",
    "inference_time_limit": "04:00:00",
    "raw_predictions_saved": True,
    "mathvista_protocol": "mathvista-strict-gt-v1.0",
    "all_job_ids": all_ids.split(),
    "mathvista_postprocess_job_ids": post_ids.split(),
}
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
