#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd/experiments/llm_only/jsd_weighted_opsd_five_eval_main4_20260807
source "${ROOT}/common.sh"
BTAATI_ACCOUNT="${BTAATI_ACCOUNT:-aip-btaati}"
GIGOR_ACCOUNT="${GIGOR_ACCOUNT:-aip-gigor}"
MERGE_ACCOUNT="${MERGE_ACCOUNT:-${BTAATI_ACCOUNT}}"
COLLECT_ACCOUNT="${COLLECT_ACCOUNT:-${BTAATI_ACCOUNT}}"
GPU_PARTITION_SHORT="${GPU_PARTITION_SHORT:-gpubase_l40s_b1}"
GPU_PARTITION_LONG="${GPU_PARTITION_LONG:-gpubase_l40s_b3}"
CPU_PARTITION="${CPU_PARTITION:-cpubase_bynode_b1}"
MANIFEST="${ROOT}/submission.tsv"
CHAIN="${ROOT}/dependency_chain.json"

mkdir -p "${OUT_ROOT}/logs"
[[ ! -e "${CHAIN}" ]] || { echo "Campaign already fully submitted" >&2; exit 1; }
for path in common.sh merge_and_preflight.sbatch eval_one_case.sbatch collect_case.py collect_results.py collect_results.sbatch; do
  [[ -f "${ROOT}/${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
done
bash -n "${ROOT}/common.sh" "${ROOT}/merge_and_preflight.sbatch" "${ROOT}/eval_one_case.sbatch" "${ROOT}/collect_results.sbatch"
python -m py_compile "${ROOT}/collect_case.py" "${ROOT}/collect_results.py"

manifest_header=$'job_id\tmethod\tstage\ttask\tratio\tdependency\taccount\tpartition\tgpus\ttime_limit\tsubmitted_at'
if [[ -e "${MANIFEST}" ]]; then
  [[ "$(head -n 1 "${MANIFEST}")" == "${manifest_header}" ]] || {
    echo "Unexpected existing manifest header" >&2
    exit 1
  }
  echo "Resuming partial submission recorded in ${MANIFEST}"
else
  printf '%s\n' "${manifest_header}" > "${MANIFEST}"
fi

existing_job_id() {
  local method="$1" stage="$2" task="$3" ratio="$4"
  awk -F '\t' -v method="${method}" -v stage="${stage}" -v task="${task}" -v ratio="${ratio}" '
    NR > 1 && $2 == method && $3 == stage && $4 == task && $5 == ratio {
      count += 1
      id = $1
    }
    END {
      if (count > 1) exit 3
      if (count == 1) print id
    }
  ' "${MANIFEST}"
}

merge_ids=()
eval_ids=()
methods=(direct_inv_d25 softmax_t005_d25 softmax_t010_d25 softmax_t005_d40 softmax_t010_d40)
for method in "${methods[@]}"; do
  configure_method "${method}"
  merge_id="$(existing_job_id "${method}" merge - -)"
  if [[ -n "${merge_id}" ]]; then
    scontrol show job "${merge_id}" >/dev/null
    echo "Reusing merge job ${merge_id} for ${method}"
  else
    merge_id="$(sbatch --parsable --account="${MERGE_ACCOUNT}" --partition="${GPU_PARTITION_SHORT}" \
      --dependency="afterok:${TRAIN_JOB_ID}" --job-name="j5-mg-${method:0:6}" \
      "${ROOT}/merge_and_preflight.sbatch" "${method}")"
    merge_id="${merge_id%%;*}"
    printf '%s\t%s\tmerge\t-\t-\tafterok:%s\t%s\t%s\t1xL40S\t01:30:00\t%s\n' \
      "${merge_id}" "${method}" "${TRAIN_JOB_ID}" "${MERGE_ACCOUNT}" "${GPU_PARTITION_SHORT}" "$(date --iso-8601=seconds)" >> "${MANIFEST}"
  fi
  merge_ids+=("${merge_id}")

  for task in mme mmstar mathvista mmmupro; do
    case "${task}" in
      mme) short=mme; walltime=03:00:00; eval_account="${BTAATI_ACCOUNT}"; eval_partition="${GPU_PARTITION_SHORT}" ;;
      mmstar) short=mms; walltime=03:00:00; eval_account="${BTAATI_ACCOUNT}"; eval_partition="${GPU_PARTITION_SHORT}" ;;
      mathvista) short=mvi; walltime=03:00:00; eval_account="${GIGOR_ACCOUNT}"; eval_partition="${GPU_PARTITION_SHORT}" ;;
      mmmupro) short=mmm; walltime=04:00:00; eval_account="${BTAATI_ACCOUNT}"; eval_partition="${GPU_PARTITION_LONG}" ;;
    esac
    for ratio in noprune r010 r020 r030; do
      ratio_short="${ratio#r}"; [[ "${ratio}" == noprune ]] && ratio_short=np
      job_id="$(existing_job_id "${method}" eval "${task}" "${ratio}")"
      if [[ -n "${job_id}" ]]; then
        scontrol show job "${job_id}" >/dev/null
        echo "Reusing eval job ${job_id} for ${method}/${task}/${ratio}"
      else
        job_id="$(sbatch --parsable --account="${eval_account}" --partition="${eval_partition}" \
          --dependency="afterok:${merge_id}" --time="${walltime}" \
          --job-name="j5-${method:0:3}-${short}-${ratio_short}" \
          "${ROOT}/eval_one_case.sbatch" "${method}" "${task}" "${ratio}")"
        job_id="${job_id%%;*}"
        printf '%s\t%s\teval\t%s\t%s\tafterok:%s\t%s\t%s\t4xL40S\t%s\t%s\n' \
          "${job_id}" "${method}" "${task}" "${ratio}" "${merge_id}" \
          "${eval_account}" "${eval_partition}" "${walltime}" "$(date --iso-8601=seconds)" >> "${MANIFEST}"
      fi
      eval_ids+=("${job_id}")
    done
  done
done
[[ "${#merge_ids[@]}" -eq 5 && "${#eval_ids[@]}" -eq 80 ]]

python - "${MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open(), delimiter="\t"))
keys = [(row["method"], row["stage"], row["task"], row["ratio"]) for row in rows]
assert len(rows) == 85, len(rows)
assert len(keys) == len(set(keys)), "duplicate campaign task"
assert sum(row["stage"] == "merge" for row in rows) == 5
assert sum(row["stage"] == "eval" for row in rows) == 80
assert sum(row["stage"] == "eval" and row["account"] == "aip-btaati" for row in rows) == 60
assert sum(row["stage"] == "eval" and row["account"] == "aip-gigor" for row in rows) == 20
assert all(row["account"] == "aip-gigor" for row in rows if row["task"] == "mathvista")
assert all(row["partition"] == "gpubase_l40s_b3" for row in rows if row["task"] == "mmmupro")
PY

dependency="afterany:$(IFS=:; echo "${eval_ids[*]}")"
collect_id="$(sbatch --parsable --account="${COLLECT_ACCOUNT}" --partition="${CPU_PARTITION}" \
  --dependency="${dependency}" "${ROOT}/collect_results.sbatch")"
collect_id="${collect_id%%;*}"

python - "${CHAIN}" "${MERGE_ACCOUNT}" "${COLLECT_ACCOUNT}" "${BTAATI_ACCOUNT}" "${GIGOR_ACCOUNT}" "${GPU_PARTITION_SHORT}" "${GPU_PARTITION_LONG}" "${CPU_PARTITION}" \
  "${collect_id}" "${merge_ids[*]}" "${eval_ids[*]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path
path = Path(sys.argv[1])
(
    merge_account,
    collect_account,
    btaati_account,
    gigor_account,
    gpu_partition_short,
    gpu_partition_long,
    cpu_partition,
    collect_id,
    merge_text,
    eval_text,
) = sys.argv[2:]
payload = {
    "status": "submitted",
    "submitted_at": datetime.now().astimezone().isoformat(),
    "merge_account": merge_account,
    "collector_account": collect_account,
    "benchmark_accounts": {
        "MME": btaati_account,
        "MMStar": btaati_account,
        "MathVista_MINI": gigor_account,
        "MMMU_Pro_4c": btaati_account,
    },
    "gpu_partitions": {
        "short": gpu_partition_short,
        "long": gpu_partition_long,
    },
    "benchmark_partitions": {
        "MME": gpu_partition_short,
        "MMStar": gpu_partition_short,
        "MathVista_MINI": gpu_partition_short,
        "MMMU_Pro_4c": gpu_partition_long,
    },
    "cpu_partition": cpu_partition,
    "methods": [
        "direct_inv_d25", "softmax_t005_d25", "softmax_t010_d25",
        "softmax_t005_d40", "softmax_t010_d40",
    ],
    "benchmarks": ["MME", "MMStar", "MathVista_MINI", "MMMU_Pro_4c"],
    "ratios": ["noprune", "r010", "r020", "r030"],
    "checkpoint_step": 10240,
    "checkpoint_load_form": "merged_full_checkpoint",
    "merge_job_ids": merge_text.split(),
    "evaluation_job_ids": eval_text.split(),
    "evaluation_count": len(eval_text.split()),
    "collector_job_id": collect_id,
}
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
