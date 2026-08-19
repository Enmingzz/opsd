#!/usr/bin/env bash
set -euo pipefail

OPSD_ROOT=/project/6101803/enmingzz/opsd
SBATCH_FILE=${OPSD_ROOT}/hypothesis_validate/opsd_checkpoint_kl_trajectory/slurm/run_checkpoint.sbatch
CONFIG=${OPSD_ROOT}/hypothesis_validate/opsd_checkpoint_kl_trajectory/config.json
MANIFEST=${OPSD_ROOT}/hypothesis_validate/opsd_checkpoint_kl_trajectory/outputs/slurm_jobs_20260729.tsv

mkdir -p "$(dirname "${MANIFEST}")"
mkdir -p /scratch/enmingzz/outputs/llm_only/logs/opsd_checkpoint_kl_20260729
printf 'checkpoint_label\tcheckpoint_step\tadapter_path\tjob_id\tsubmitted_at\n' > "${MANIFEST}"

while IFS=$'\t' read -r label step adapter; do
  submission=$(sbatch --parsable --job-name="kl-${label}" "${SBATCH_FILE}" "${label}" "${adapter}")
  job_id=${submission%%;*}
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${step}" "${adapter}" "${job_id}" "$(date --iso-8601=seconds)" | tee -a "${MANIFEST}"
done < <(
  python - "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(cfg["checkpoint_root"])
for item in cfg["checkpoint_steps"]:
    raw = item["adapter_path"]
    adapter = raw if raw == "__BASE__" else str((root / raw).resolve())
    print(f"{item['label']}\t{item['step']}\t{adapter}")
PY
)

echo "Submission manifest: ${MANIFEST}"
