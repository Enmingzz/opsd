#!/usr/bin/env bash
#SBATCH --job-name=lcot20k_smoke
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --exclude=kn136
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/logs/slurm-%x-%j.err

set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd/experiments/llm_only/lcot20k_retrain_dropout0_20260803
OUT_ROOT=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/smoke4gpu

"${ROOT}/slurm/run_train.sh" \
  "${ROOT}/configs/progressive_opsd_smoke32.yaml" \
  "${OUT_ROOT}/progressive_opsd" progressive 32 false

"${ROOT}/slurm/run_train.sh" \
  "${ROOT}/configs/sft20k_smoke32.yaml" \
  "${OUT_ROOT}/sft" sft 32 false
