#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=f3-eval
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/eval/f_trajectory_bottom80_step10240_merged_main3_20260818/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/eval/f_trajectory_bottom80_step10240_merged_main3_20260818/logs/%x-%j.err

set -euo pipefail
[[ "$#" -eq 3 ]] || { echo "Usage: $0 METHOD TASK RATIO" >&2; exit 2; }
METHOD_ARG="$1"
TASK="${2,,}"
RATIO_TAG="$3"
source /project/6101803/enmingzz/ckpt_eval_trainenv/env_train_runtime.sh
source /project/6101803/enmingzz/opsd/experiments/llm_only/f_trajectory_bottom80_eval_main3_merged_20260818/common.sh
set_method "${METHOD_ARG}"

case "${TASK}" in
  mme) EVAL_TASK=mme; DATASET_NAME=MME; EXPECTED_ROWS=2374 ;;
  mmstar) EVAL_TASK=mmstar; DATASET_NAME=MMStar; EXPECTED_ROWS=1500 ;;
  mathvista) EVAL_TASK=mathvista; DATASET_NAME=MathVista_MINI; EXPECTED_ROWS=1000 ;;
  *) echo "Unsupported task: ${TASK}" >&2; exit 2 ;;
esac
case "${RATIO_TAG}" in noprune|r010|r020|r030) ;; *) echo "Unsupported ratio: ${RATIO_TAG}" >&2; exit 2 ;; esac

python "${CAMPAIGN_ROOT}/validate_artifact.py" merged --adapter "${ADAPTER_PATH}" \
  --config "${TRAIN_CONFIG}" --run-root "${RUN_ROOT}" --method "${METHOD_LONG}" \
  --objective "${TRAINING_OBJECTIVE}" --training-job-id "${TRAIN_JOB_ID}" \
  --merged "${MERGED_MODEL}" >/dev/null

export CKPT_EVAL_ROOT=/project/6101803/enmingzz/ckpt_eval_trainenv
export CKPT_EVAL_OUT_ROOT="${OUT_ROOT}"
export OPSD_ROOT EXPECTED_BASE_MODEL="${BASE_MODEL}"
export EXPECTED_ADAPTER_PATH="${ADAPTER_PATH}"
export MERGED_MODEL METHOD_TAG EVAL_TASK RATIO_TAG CAMPAIGN_GROUP
export LOG_ROOT="${OUT_ROOT}/logs"
export EVAL_NPROC_PER_NODE=1

ENGINE="${OPSD_ROOT}/slurm_jobs/eval_llava_cot_sft1080init_opsd_cached_merged_5bench_20260713.sbatch"
VALIDATOR="${OPSD_ROOT}/scripts/eval/validate_raw_prediction_output.py"
echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID} method=${METHOD} task=${TASK} ratio=${RATIO_TAG}"
nvidia-smi
bash "${ENGINE}"

if [[ "${EVAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight-only completed; skipping result collection"
  exit 0
fi

WORK_DIR="${OUT_ROOT}/eval_vlmevalkit_trainenv/${CAMPAIGN_GROUP}/${METHOD_TAG}_${RATIO_TAG}_${DATASET_NAME}_${SLURM_JOB_ID}"
result_file="$(find "${WORK_DIR}/Qwen" -type f -name "Qwen_${DATASET_NAME}.xlsx" | sort | tail -n 1)"
[[ -n "${result_file}" && -f "${result_file}" ]]
source "${CKPT_EVAL_ROOT}/env_train_runtime.sh"
python "${VALIDATOR}" --result-file "${result_file}" --expected-rows "${EXPECTED_ROWS}" \
  --report-out "${WORK_DIR}/raw_prediction_validation.json"
python "${CAMPAIGN_ROOT}/collect_case.py" --work-dir "${WORK_DIR}" \
  --dataset "${DATASET_NAME}" --ratio "${RATIO_TAG}" --method "${METHOD}" \
  --method-long "${METHOD_LONG}" --job-id "${SLURM_JOB_ID}"
