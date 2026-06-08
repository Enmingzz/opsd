#!/bin/bash
#SBATCH --job-name=opsd_r025_analyze
#SBATCH --account=aip-btaati
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/enmingzz/temp/opsd_logs/%x_%j.out

set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export TOKENIZERS_PARALLELISM=false

OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/real_single_r025
TRAIN_DIR="${OUT_ROOT}/${TRAIN_DIR_NAME:-divprune_kl_single_r025}"
EVAL_DIR="${OUT_ROOT}/${EVAL_DIR_NAME:-evals}"
ANALYSIS_DIR="${OUT_ROOT}/${ANALYSIS_DIR_NAME:-analysis_limit200}"
mkdir -p "${ANALYSIS_DIR}" /scratch/enmingzz/temp/opsd_logs

python opsd/scripts/analyze_teacher_agreement.py \
  --baseline_summary_csv "${OUT_ROOT}/baseline_sweep_200/summary.csv" \
  --training_log_jsonl "${TRAIN_DIR}/training_log.jsonl" \
  --output_dir "${ANALYSIS_DIR}" \
  --title "OPSD real single-ratio divprune_lite r=0.25 limit=200" \
  --eval_jsonl "${EVAL_DIR}/eval_step250_r025_limit200.jsonl" \
  --checkpoint_name step_250 \
  --eval_jsonl "${EVAL_DIR}/eval_step500_r025_limit200.jsonl" \
  --checkpoint_name step_500 \
  --eval_jsonl "${EVAL_DIR}/eval_step750_r025_limit200.jsonl" \
  --checkpoint_name step_750 \
  --eval_jsonl "${EVAL_DIR}/eval_step1000_r025_limit200.jsonl" \
  --checkpoint_name step_1000

cp "${ANALYSIS_DIR}/report.md" "${OUT_ROOT}/report.md"
