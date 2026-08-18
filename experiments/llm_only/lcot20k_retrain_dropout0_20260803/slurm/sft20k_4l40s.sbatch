#!/usr/bin/env bash
#SBATCH --job-name=lcot20k_sft
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --exclude=kn136
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/logs/slurm-%x-%j.err

set -euo pipefail
ROOT=/project/6101803/enmingzz/opsd/experiments/llm_only/lcot20k_retrain_dropout0_20260803
OUT=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/sft
exec "${ROOT}/slurm/run_train.sh" "${ROOT}/configs/sft20k.yaml" "${OUT}" sft 20000 true
