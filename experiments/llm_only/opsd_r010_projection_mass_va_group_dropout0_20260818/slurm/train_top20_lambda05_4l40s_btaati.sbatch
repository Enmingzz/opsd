#!/usr/bin/env bash
#SBATCH --job-name=opsd-r10-pm20
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/opsd_r010_projection_mass_top20_lambda05_dropout0_20260818/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/opsd_r010_projection_mass_top20_lambda05_dropout0_20260818/logs/slurm-%x-%j.err

set -euo pipefail
OPSD_ROOT=/project/6101803/enmingzz/opsd
EXP_ROOT="${OPSD_ROOT}/experiments/llm_only/opsd_r010_projection_mass_va_group_dropout0_20260818"
OUTPUT_ROOT=/scratch/enmingzz/outputs/llm_only/opsd_r010_projection_mass_top20_lambda05_dropout0_20260818
CONFIG_PATH="${EXP_ROOT}/configs/train_top20_lambda05_10240.yaml"
OUT_DIR="${OUTPUT_ROOT}/run"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUT_DIR}"
cd "${OPSD_ROOT}"
source /project/6101803/enmingzz/env/vsi-official.sh
export PYTHONPATH=/project/6101803/enmingzz
python "${EXP_ROOT}/validate_config.py" --config "${CONFIG_PATH}" \
  --expected-steps 10240 --expected-top-fraction 0.20 \
  --expected-high-group-lambda 0.50 \
  --output "${OUT_DIR}/variant_preflight.json" --overwrite
exec bash "${EXP_ROOT}/run_4gpu.sh" "${CONFIG_PATH}" "${OUT_DIR}"
