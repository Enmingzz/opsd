#!/bin/bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --job-name=vzip_official
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030/slurm-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030/slurm-%j.err

set -euo pipefail

mkdir -p /scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/eval_visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030

/bin/bash /project/6101803/enmingzz/opsd/scripts/eval_visionzip_base_official_split_vlmevalkit_mme_mmstar_pope.sh
