#!/usr/bin/env bash
#SBATCH --job-name=opsd_d0_res_smoke
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --exclude=kn136
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/original_opsd_dropout0_exact_resume_to20k_20260803/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/original_opsd_dropout0_exact_resume_to20k_20260803/logs/slurm-%x-%j.err

set -euo pipefail
export CONFIG_PATH=/project/6101803/enmingzz/opsd/experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/configs/smoke_exact_resume_32samples.yaml
export OUT_DIR=/scratch/enmingzz/outputs/llm_only/original_opsd_dropout0_exact_resume_to20k_20260803/smoke32
exec /project/6101803/enmingzz/opsd/experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/slurm/run_exact_resume_4l40s.sh
