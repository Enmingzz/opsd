#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd/hypothesis_validate/opsd_checkpoint_kl_trajectory
CONFIG_ROOT=${ROOT}/configs/ratio_pairs
OUTPUT_ROOT=${ROOT}/outputs/ratio_pairs_mmstar_clean100_max1024
PROJECT_ROOT=/project/6101803/enmingzz

source "${PROJECT_ROOT}/env/vsi-official.sh"
export PYTHONPATH=${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}

python - <<'PY'
from pathlib import Path
root = Path("/project/6101803/enmingzz/opsd/hypothesis_validate/opsd_checkpoint_kl_trajectory/outputs/ratio_pairs_mmstar_clean100_max1024")
sample_count = sum(1 for _ in root.glob("*/*/step_*/samples/*.json"))
error_count = sum(1 for _ in root.glob("*/*/step_*/errors/*.json"))
if sample_count != 8800 or error_count != 0:
    raise SystemExit(f"Refusing incomplete finalization: samples={sample_count}/8800 errors={error_count}")
print(f"input_check=PASS samples={sample_count} errors={error_count}")
PY

for pair in r015_r0175 r015_r020 r0175_r020 r020_r0225; do
  for method in opsd sft; do
    python "${ROOT}/scripts/aggregate.py" \
      --config "${CONFIG_ROOT}/config_${method}_${pair}_max1024.json"
  done
done

compare_pair() {
  local pair="$1"
  local low="$2"
  local high="$3"
  python "${ROOT}/scripts/compare_sft_opsd.py" \
    --opsd-summary "${OUTPUT_ROOT}/${pair}/opsd/analysis/checkpoint_summary.csv" \
    --sft-summary "${OUTPUT_ROOT}/${pair}/sft/analysis/checkpoint_summary.csv" \
    --output-dir "${OUTPUT_ROOT}/${pair}/sft_vs_opsd" \
    --low-ratio "${low}" \
    --high-ratio "${high}"
}

compare_pair r015_r0175 0.15 0.175
compare_pair r015_r020 0.15 0.20
compare_pair r0175_r020 0.175 0.20
compare_pair r020_r0225 0.20 0.225

python "${ROOT}/scripts/summarize_ratio_pair_sweep.py"

echo "final_output=${OUTPUT_ROOT}/analysis"
