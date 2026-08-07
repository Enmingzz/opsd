#!/usr/bin/env bash
#SBATCH --job-name=jsd_traj_opsd
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/jsd_trajectory_reweight_dropout0_20260806/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/jsd_trajectory_reweight_dropout0_20260806/logs/slurm-%x-%j.err

set -euo pipefail
: "${CONFIG_PATH:?Set CONFIG_PATH}"
: "${OUT_DIR:?Set OUT_DIR}"

mkdir -p /scratch/enmingzz/outputs/llm_only/jsd_trajectory_reweight_dropout0_20260806/logs
cd /project/6101803/enmingzz/opsd
exec bash experiments/llm_only/jsd_trajectory_reweight_dropout0_20260806/run_4gpu.sh \
  "${CONFIG_PATH}" "${OUT_DIR}"
