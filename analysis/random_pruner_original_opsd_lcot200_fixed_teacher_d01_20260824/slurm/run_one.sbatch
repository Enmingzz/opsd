#!/usr/bin/env bash
#SBATCH --job-name=rpF1-lc200
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/project/6101803/enmingzz/opsd/analysis/random_pruner_original_opsd_lcot200_fixed_teacher_d01_20260824/logs/%x-%j.out
#SBATCH --error=/project/6101803/enmingzz/opsd/analysis/random_pruner_original_opsd_lcot200_fixed_teacher_d01_20260824/logs/%x-%j.err

set -euo pipefail
: "${STEP:?Submit with --export=ALL,STEP=<0|1024|2048|3072>}"

ROOT=/project/6101803/enmingzz/opsd/analysis/random_pruner_original_opsd_lcot200_fixed_teacher_d01_20260824
echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID} account=${SLURM_JOB_ACCOUNT}"
echo "git=$(git -C /project/6101803/enmingzz/opsd rev-parse HEAD) step=${STEP}"
nvidia-smi
exec bash "${ROOT}/run_one.sh" "${STEP}"
