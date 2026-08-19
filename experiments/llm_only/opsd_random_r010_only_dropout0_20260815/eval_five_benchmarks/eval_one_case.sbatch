#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=r010-eval
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/logs/%x-%j.err

set -euo pipefail
[[ "$#" -eq 2 ]] || { echo "Usage: $0 TASK RATIO" >&2; exit 2; }
TASK="${1,,}"
RATIO_TAG="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"

case "${TASK}" in
  mme) EVAL_TASK=mme; DATASET_NAME=MME; EXPECTED_ROWS=2374 ;;
  mmstar) EVAL_TASK=mmstar; DATASET_NAME=MMStar; EXPECTED_ROWS=1500 ;;
  mathvista) EVAL_TASK=mathvista; DATASET_NAME=MathVista_MINI; EXPECTED_ROWS=1000 ;;
  mathverse) EVAL_TASK=mathverse; DATASET_NAME=MathVerse_MINI_Vision_Only; EXPECTED_ROWS=788 ;;
  mmmupro) EVAL_TASK=mmmupro; DATASET_NAME=MMMU_Pro_4c; EXPECTED_ROWS=1730 ;;
  *) echo "Unsupported task: ${TASK}" >&2; exit 2 ;;
esac
case "${RATIO_TAG}" in noprune|r010|r020|r030) ;; *) echo "Unsupported ratio: ${RATIO_TAG}" >&2; exit 2 ;; esac

[[ -f "${MERGED_MODEL}/merge_metadata.json" ]] || {
  echo "Merged checkpoint is missing; run ${HERE}/merge_model.sh first" >&2
  exit 1
}
mkdir -p "${OUT_ROOT}/logs"

export CKPT_EVAL_ROOT=/project/6101803/enmingzz/ckpt_eval_trainenv
export CKPT_EVAL_OUT_ROOT="${OUT_ROOT}"
export EXPECTED_BASE_MODEL="${BASE_MODEL}"
export EXPECTED_ADAPTER_PATH="${ADAPTER_PATH}"
export MERGED_MODEL METHOD_TAG EVAL_TASK RATIO_TAG CAMPAIGN_GROUP OPSD_ROOT
export LOG_ROOT="${OUT_ROOT}/logs"
export EVAL_NPROC_PER_NODE=4
export VLMEVAL_SAVE_RAW_PREDICTION=1 VLMEVAL_STRICT_ERRORS=1

ENGINE="${OPSD_ROOT}/slurm_jobs/eval_llava_cot_sft1080init_opsd_cached_merged_5bench_20260713.sbatch"
echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-manual} task=${TASK} ratio=${RATIO_TAG}"
bash "${ENGINE}"

WORK_DIR="${OUT_ROOT}/eval_vlmevalkit_trainenv/${CAMPAIGN_GROUP}/${METHOD_TAG}_${RATIO_TAG}_${DATASET_NAME}_${SLURM_JOB_ID:-manual}"
result_file="$(find "${WORK_DIR}/Qwen" -type f -name "Qwen_${DATASET_NAME}.xlsx" | sort | tail -n 1)"
[[ -n "${result_file}" && -f "${result_file}" ]]
source "${CKPT_EVAL_ROOT}/env_train_runtime.sh"
python "${OPSD_ROOT}/scripts/eval/validate_raw_prediction_output.py" \
  --result-file "${result_file}" \
  --expected-rows "${EXPECTED_ROWS}" \
  --report-out "${WORK_DIR}/raw_prediction_validation.json"
echo "raw_result=${result_file}"
