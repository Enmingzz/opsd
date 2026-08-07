#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=cpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=j5-collect
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/eval/jsd_weighted_opsd_five_step10240_merged_20260807/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/eval/jsd_weighted_opsd_five_step10240_merged_20260807/logs/%x-%j.err

set -euo pipefail
python /project/6101803/enmingzz/opsd/experiments/llm_only/jsd_weighted_opsd_five_eval_main4_20260807/collect_results.py
