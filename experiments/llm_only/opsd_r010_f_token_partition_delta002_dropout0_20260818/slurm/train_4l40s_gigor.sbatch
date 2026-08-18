#!/usr/bin/env bash
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=kn136
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/opsd_r010_f_token_partition_delta002_dropout0_20260818/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/opsd_r010_f_token_partition_delta002_dropout0_20260818/logs/%x-%j.err

set -euo pipefail
: "${VARIANT:?Submit with --export=ALL,VARIANT=token_top20|token_bottom80|token_random_drop20}"
mkdir -p /scratch/enmingzz/outputs/llm_only/opsd_r010_f_token_partition_delta002_dropout0_20260818/logs
exec bash /project/6101803/enmingzz/opsd/experiments/llm_only/opsd_r010_f_token_partition_delta002_dropout0_20260818/run_variant.sh "${VARIANT}"
