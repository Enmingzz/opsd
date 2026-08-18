#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=f3-merge
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/eval/f_trajectory_bottom80_step10240_merged_main3_20260818/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/eval/f_trajectory_bottom80_step10240_merged_main3_20260818/logs/%x-%j.err

set -euo pipefail
[[ "$#" -eq 1 ]] || { echo "Usage: $0 METHOD" >&2; exit 2; }
source /project/6101803/enmingzz/ckpt_eval_trainenv/env_train_runtime.sh
source /project/6101803/enmingzz/opsd/experiments/llm_only/f_trajectory_bottom80_eval_main3_merged_20260818/common.sh
set_method "$1"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/merge" "${MERGED_ROOT}"
cd "${OPSD_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/scratch/enmingzz/cache/huggingface}"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1

validator=(python "${CAMPAIGN_ROOT}/validate_artifact.py" --adapter "${ADAPTER_PATH}" \
  --config "${TRAIN_CONFIG}" --run-root "${RUN_ROOT}" --method "${METHOD_LONG}" \
  --objective "${TRAINING_OBJECTIVE}" --training-job-id "${TRAIN_JOB_ID}")

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID} method=${METHOD}"
nvidia-smi
"${validator[@]}" source > "${OUT_ROOT}/merge/${METHOD}_source_validation.json"

if [[ -d "${MERGED_MODEL}" ]]; then
  "${validator[@]}" merged --merged "${MERGED_MODEL}" > "${OUT_ROOT}/merge/${METHOD}_merged_validation.json"
  echo "Matching merged checkpoint already exists: ${MERGED_MODEL}"
  exit 0
fi

temporary="${MERGED_MODEL}.tmp-${SLURM_JOB_ID}"
[[ ! -e "${temporary}" ]] || { echo "Temporary path already exists: ${temporary}" >&2; exit 1; }
python scripts/merge_qwen25vl_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --output_dir "${temporary}" \
  --attn_implementation flash_attention_2 \
  --device_map auto \
  --bf16 \
  --note "${METHOD_LONG}, step 10240, r010-only VisionZip OPSD, LLM-only LoRA dropout=0."

"${validator[@]}" finalize --merged "${temporary}" --final-path "${MERGED_MODEL}"
mv "${temporary}" "${MERGED_MODEL}"
"${validator[@]}" merged --merged "${MERGED_MODEL}" > "${OUT_ROOT}/merge/${METHOD}_merged_validation.json"
echo "completed=$(date --iso-8601=seconds) merged=${MERGED_MODEL}"
