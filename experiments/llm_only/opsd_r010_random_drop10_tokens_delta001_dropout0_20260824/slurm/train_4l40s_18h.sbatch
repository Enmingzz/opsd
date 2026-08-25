#!/usr/bin/env bash
#SBATCH --job-name=r10-randdrop10
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824/logs/slurm-%x-%j.err

set -euo pipefail
OPSD_ROOT=/project/6101803/enmingzz/opsd
EXP_ROOT=${OPSD_ROOT}/experiments/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824
mkdir -p /scratch/enmingzz/outputs/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824/logs
cd "${OPSD_ROOT}"
exec bash "${EXP_ROOT}/run_4gpu.sh"
