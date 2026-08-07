#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=j5-eval
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/eval/jsd_weighted_opsd_five_step10240_merged_20260807/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/eval/jsd_weighted_opsd_five_step10240_merged_20260807/logs/%x-%j.err

set -euo pipefail
[[ "$#" -eq 3 ]] || { echo "Usage: $0 METHOD TASK RATIO" >&2; exit 2; }
METHOD_ARG="$1"
TASK="${2,,}"
RATIO_TAG="$3"

source /project/6101803/enmingzz/opsd/experiments/llm_only/jsd_weighted_opsd_five_eval_main4_20260807/common.sh
configure_method "${METHOD_ARG}"
ENGINE="${OPSD_ROOT}/slurm_jobs/eval_llava_cot_sft1080init_opsd_cached_merged_5bench_20260713.sbatch"
VALIDATOR="${OPSD_ROOT}/scripts/eval/validate_raw_prediction_output.py"
MMMU_SCORER="${OPSD_ROOT}/experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/eval_step20000_qwen5_20260804/score_mmmupro_with_qwen7.py"

case "${TASK}" in
  mme) EVAL_TASK=mme; DATASET_NAME=MME; EXPECTED_ROWS=2374 ;;
  mmstar) EVAL_TASK=mmstar; DATASET_NAME=MMStar; EXPECTED_ROWS=1500 ;;
  mathvista) EVAL_TASK=mathvista; DATASET_NAME=MathVista_MINI; EXPECTED_ROWS=1000 ;;
  mmmupro) EVAL_TASK=mmmupro; DATASET_NAME=MMMU_Pro_4c; EXPECTED_ROWS=1730 ;;
  *) echo "Unsupported task: ${TASK}" >&2; exit 2 ;;
esac
case "${RATIO_TAG}" in noprune|r010|r020|r030) ;; *) echo "Unsupported ratio: ${RATIO_TAG}" >&2; exit 2 ;; esac

python - "${ADAPTER_PATH}" "${MERGED_MODEL}" "${TRAIN_JOB_ID}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

adapter, merged = map(lambda x: Path(x).resolve(), sys.argv[1:3])
job_id = sys.argv[3]
assert adapter.joinpath("COMPLETE").is_file()
metadata = json.loads(merged.joinpath("merge_metadata.json").read_text())
assert Path(metadata["adapter_path"]).resolve() == adapter
assert Path(metadata["output_dir"]).resolve() == merged
assert metadata["adapter_model_sha256"] == sha256(adapter / "adapter_model.safetensors")
assert metadata["source_training_job_id"] == job_id
assert int(metadata["source_final_global_step"]) == 10240
assert metadata["parameter_scope"] == "language_decoder_only"
assert float(metadata["lora_dropout"]) == 0.0
assert merged.joinpath("model.safetensors.index.json").is_file()
assert len(list(merged.glob("model-*.safetensors"))) == 4
PY

export CKPT_EVAL_ROOT=/project/6101803/enmingzz/ckpt_eval_trainenv
export CKPT_EVAL_OUT_ROOT="${OUT_ROOT}"
export OPSD_ROOT EXPECTED_BASE_MODEL="${BASE_MODEL}"
export EXPECTED_ADAPTER_PATH="${ADAPTER_PATH}"
export MERGED_MODEL METHOD_TAG EVAL_TASK RATIO_TAG CAMPAIGN_GROUP
export LOG_ROOT="${OUT_ROOT}/logs"
export EVAL_NPROC_PER_NODE=4
export MMMU_PRO_TOTAL_MIN_IMAGE_TOKENS=1280
export MMMU_PRO_TOTAL_MAX_IMAGE_TOKENS=4096
export MMSTAR_QWEN_JUDGE_SCOPE=misses
export VLMEVAL_SAVE_RAW_PREDICTION=1
export VLMEVAL_STRICT_ERRORS=1
export TOKENIZERS_PARALLELISM=false

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-manual} method=${METHOD} task=${TASK} ratio=${RATIO_TAG}"
if [[ "${CASE_PREFLIGHT_ONLY:-0}" == "1" ]]; then export EVAL_PREFLIGHT_ONLY=1; fi
bash "${ENGINE}"
if [[ "${CASE_PREFLIGHT_ONLY:-0}" == "1" ]]; then exit 0; fi

WORK_DIR="${OUT_ROOT}/eval_vlmevalkit_trainenv/${CAMPAIGN_GROUP}/${METHOD_TAG}_${RATIO_TAG}_${DATASET_NAME}_${SLURM_JOB_ID}"
result_file="$(find "${WORK_DIR}/Qwen" -type f -name "Qwen_${DATASET_NAME}.xlsx" | sort | tail -n 1)"
[[ -n "${result_file}" && -f "${result_file}" ]]
source "${CKPT_EVAL_ROOT}/env_train_runtime.sh"

if [[ "${TASK}" == "mmmupro" ]]; then
  export MMMU_QWEN_JUDGE_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
  export MMMU_QWEN_JUDGE_DEVICE=cuda:0
  export MMMU_QWEN_JUDGE_DTYPE=bfloat16
  export MMMU_QWEN_JUDGE_MAX_NEW_TOKENS=8
  python "${MMMU_SCORER}" --result-file "${result_file}"
fi

python "${VALIDATOR}" \
  --result-file "${result_file}" \
  --expected-rows "${EXPECTED_ROWS}" \
  --report-out "${WORK_DIR}/raw_prediction_validation.json"
python "${CAMPAIGN_ROOT}/collect_case.py" \
  --work-dir "${WORK_DIR}" --method "${METHOD}" --dataset "${DATASET_NAME}" \
  --ratio "${RATIO_TAG}" --job-id "${SLURM_JOB_ID}"

echo "completed=$(date --iso-8601=seconds) method=${METHOD} task=${TASK} ratio=${RATIO_TAG}"
