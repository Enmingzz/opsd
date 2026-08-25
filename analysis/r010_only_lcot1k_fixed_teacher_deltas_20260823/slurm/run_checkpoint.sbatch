#!/usr/bin/env bash
#SBATCH --job-name=r10lcot-div
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/enmingzz/logs/r010_lcot1k_fixed_teacher_deltas/%x-%j.out
#SBATCH --error=/scratch/enmingzz/logs/r010_lcot1k_fixed_teacher_deltas/%x-%j.err

set -euo pipefail
: "${STEP:?Submit with --export=ALL,STEP=<checkpoint step>}"
LIMIT=${LIMIT:-1000}
OFFSET=${OFFSET:-0}

ROOT=/project/6101803/enmingzz/opsd
OUT_ROOT=${OUT_ROOT:-${ROOT}/analysis/r010_only_lcot1k_fixed_teacher_deltas_20260823/outputs}
PADDED=$(printf '%06d' "${STEP}")
if [[ "${LIMIT}" == "1000" && "${OFFSET}" == "0" ]]; then
  OUTPUT_DIR="${OUT_ROOT}/step_${PADDED}"
else
  OUTPUT_DIR="${OUT_ROOT}/step_${PADDED}/shards/offset_$(printf '%06d' "${OFFSET}")_limit_$(printf '%06d' "${LIMIT}")"
fi
mkdir -p /scratch/enmingzz/logs/r010_lcot1k_fixed_teacher_deltas "${OUTPUT_DIR}"
cd "${ROOT}"

echo "job=${SLURM_JOB_ID} account=${SLURM_JOB_ACCOUNT} host=$(hostname) date=$(date --iso-8601=seconds)"
echo "git=$(git rev-parse HEAD) step=${STEP} limit=${LIMIT} offset=${OFFSET} output=${OUTPUT_DIR}"

exec bash analysis/r010_only_lcot1k_fixed_teacher_deltas_20260823/run_checkpoint.sh \
  "${STEP}" "${OUTPUT_DIR}" "${LIMIT}" "${OFFSET}"
