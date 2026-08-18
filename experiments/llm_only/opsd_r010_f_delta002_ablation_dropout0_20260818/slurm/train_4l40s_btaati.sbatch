#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=kn136
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818/logs/%x-%j.err

set -euo pipefail
: "${VARIANT:?Submit with --export=ALL,VARIANT=<variant>}"
EXP=/project/6101803/enmingzz/opsd/experiments/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818
mkdir -p /scratch/enmingzz/outputs/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818/logs
date --iso-8601=seconds
hostname
nvidia-smi
exec bash "${EXP}/run_variant.sh" "${VARIANT}" train
