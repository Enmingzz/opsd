#!/usr/bin/env bash
#SBATCH --job-name=lcot20k_prog2
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --exclude=kn136
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/logs/slurm-%x-%j.err

set -euo pipefail
ROOT=/project/6101803/enmingzz/opsd/experiments/llm_only/lcot20k_retrain_dropout0_20260803
OUT=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/progressive_opsd
if [[ ! -f "${OUT}/resume_checkpoints/step_009984/COMPLETE" ]]; then
  echo "stage-1 checkpoint is incomplete" >&2
  exit 1
fi
exec "${ROOT}/slurm/run_train.sh" "${ROOT}/configs/progressive_opsd_stage2.yaml" "${OUT}" progressive 20000 true
