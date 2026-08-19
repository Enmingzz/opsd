#!/usr/bin/env bash
#SBATCH --job-name=r010-20k-s1
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/logs/slurm-%x-%j.err

set -euo pipefail
OPSD_ROOT=/project/6101803/enmingzz/opsd
EXP_ROOT="${OPSD_ROOT}/experiments/llm_only/opsd_random_r010_only_dropout0_20260815"
OUTPUT_ROOT=/scratch/enmingzz/outputs/llm_only/opsd_r010_only_dropout0_lcot20k_20260818
CONFIG_PATH="${EXP_ROOT}/configs/train_20000_stage1.yaml"
OUT_DIR="${OUTPUT_ROOT}/run"
mkdir -p "${OUTPUT_ROOT}/logs"
cd "${OPSD_ROOT}"
exec bash "${EXP_ROOT}/run_4gpu.sh" "${CONFIG_PATH}" "${OUT_DIR}"
