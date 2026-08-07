#!/usr/bin/env bash
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=mn_p64
#SBATCH --output=/project/6101803/enmingzz/opsd/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_base_qwen25vl7b_t07_p095_20260722/slurm_logs/%x-%j.out
#SBATCH --error=/project/6101803/enmingzz/opsd/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_base_qwen25vl7b_t07_p095_20260722/slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd
BASE_RUNNER="${ROOT}/hypothesis_validate/pass_at_k/slurm/run_mmstar_clean100_raw_and_postjudge_1gpu.sbatch"
export SWEEP_ROOT="${SWEEP_ROOT:-${ROOT}/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_base_qwen25vl7b_t07_p095_20260722}"
export SAMPLE_FILE="${ROOT}/hypothesis_validate/manual_review/mmstar_numeric_open_ended_clean100_seed42/samples.jsonl"
export RUN_SUMMARY=0
export LIMIT=100
export START_INDEX=0

: "${RATIO:?Set RATIO to 0.1, 0.2, 0.3, 0.4, or 1.0}"
: "${RATIO_TAG:?Set RATIO_TAG to r010, r020, r030, r040, or r100}"

mkdir -p "${SWEEP_ROOT}/slurm_logs"
cd "${ROOT}"

echo "[$(date --iso-8601=seconds)] numeric clean100 start ratio=${RATIO_TAG} job=${SLURM_JOB_ID}"
DECODE_MODE=greedy bash "${BASE_RUNNER}"
DECODE_MODE=sample64 bash "${BASE_RUNNER}"
echo "[$(date --iso-8601=seconds)] numeric clean100 complete ratio=${RATIO_TAG} job=${SLURM_JOB_ID}"
