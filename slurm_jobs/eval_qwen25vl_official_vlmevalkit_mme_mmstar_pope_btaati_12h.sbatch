#!/bin/bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen25vl_eval
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_qwen25vl_official_7b_qwen25bootstrap_vlmevalkit_mme_mmstar_pope/slurm-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_qwen25vl_official_7b_qwen25bootstrap_vlmevalkit_mme_mmstar_pope/slurm-%j.err

set -euo pipefail

mkdir -p /scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_qwen25vl_official_7b_qwen25bootstrap_vlmevalkit_mme_mmstar_pope

/bin/bash /project/6101803/enmingzz/opsd/scripts/eval_qwen25vl_official_vlmevalkit_mme_mmstar_pope.sh
