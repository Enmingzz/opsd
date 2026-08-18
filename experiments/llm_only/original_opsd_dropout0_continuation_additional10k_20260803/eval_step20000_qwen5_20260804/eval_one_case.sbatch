#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=op20-q5
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/eval/original_opsd_dropout0_step20000_merged_qwen5_20260804/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/eval/original_opsd_dropout0_step20000_merged_qwen5_20260804/logs/%x-%j.err

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 TASK RATIO_TAG" >&2
  exit 2
fi

TASK="${1,,}"
RATIO_TAG="$2"
OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
CKPT_EVAL_ROOT=/project/6101803/enmingzz/ckpt_eval_trainenv
CAMPAIGN_ROOT="${OPSD_ROOT}/experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/eval_step20000_qwen5_20260804"
OUT_ROOT=/scratch/enmingzz/outputs/llm_only/eval/original_opsd_dropout0_step20000_merged_qwen5_20260804
RUN_ROOT=/scratch/enmingzz/outputs/llm_only/original_opsd_dropout0_exact_resume_to20k_20260803/run
ADAPTER_PATH="${RUN_ROOT}/resume_checkpoints/step_020000"
MERGED_MODEL=/scratch/enmingzz/outputs/llm_only/merged_models/original_opsd_dropout0_20k_20260804/original_opsd_dropout0_step20000_qwen25vl7b_merged_bf16
ENGINE="${OPSD_ROOT}/slurm_jobs/eval_llava_cot_sft1080init_opsd_cached_merged_5bench_20260713.sbatch"
VALIDATOR="${OPSD_ROOT}/scripts/eval/validate_raw_prediction_output.py"
METHOD_TAG=original_opsd_dropout0_step20000_merged
CAMPAIGN_GROUP=original_opsd_dropout0_step20000_merged_reasoning_qwen5_cleanarmen_20260804

case "${TASK}" in
  mme) EVAL_TASK=mme; DATASET_NAME=MME; EXPECTED_ROWS=2374 ;;
  mmstar) EVAL_TASK=mmstar; DATASET_NAME=MMStar; EXPECTED_ROWS=1500 ;;
  mathvista) EVAL_TASK=mathvista; DATASET_NAME=MathVista_MINI; EXPECTED_ROWS=1000 ;;
  mathverse) EVAL_TASK=mathverse; DATASET_NAME=MathVerse_MINI_Vision_Only; EXPECTED_ROWS=788 ;;
  mmmupro) EVAL_TASK=mmmupro; DATASET_NAME=MMMU_Pro_4c; EXPECTED_ROWS=1730 ;;
  *) echo "Unsupported task: ${TASK}" >&2; exit 2 ;;
esac
case "${RATIO_TAG}" in
  noprune) enable_visionzip=False; visionzip_ratio=0.0; retention_ratio=1.0 ;;
  r010) enable_visionzip=True; visionzip_ratio=0.95; retention_ratio=0.1 ;;
  r020) enable_visionzip=True; visionzip_ratio=0.85; retention_ratio=0.2 ;;
  r030) enable_visionzip=True; visionzip_ratio=0.75; retention_ratio=0.3 ;;
  *) echo "Unsupported ratio: ${RATIO_TAG}" >&2; exit 2 ;;
esac

mkdir -p "${OUT_ROOT}/logs"

python - "${RUN_ROOT}" "${ADAPTER_PATH}" "${MERGED_MODEL}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_root, adapter, merged = map(lambda value: Path(value).resolve(), sys.argv[1:])
adapter_weights = adapter / "adapter_model.safetensors"
adapter_sha256 = hashlib.sha256(adapter_weights.read_bytes()).hexdigest()
adapter_config = json.loads(adapter.joinpath("adapter_config.json").read_text())
audit = json.loads(run_root.joinpath("post_training_audit.json").read_text())
scope = json.loads(run_root.joinpath("final_scope_verification.json").read_text())
metadata = json.loads(merged.joinpath("merge_metadata.json").read_text())

assert audit["status"] == "passed" and audit["final_global_step"] == 20000
assert audit["all_new_losses_finite"] is True and audit["lora_dropout"] == 0.0
assert audit["teacher_ground_truth_access"] is False
assert scope["scope_verified"] is True and scope["expected_scope"] == "language_decoder_only"
assert scope["visual_name_count"] == 0 and scope["adapter_tensor_count"] == 392
assert Path(scope["adapter_path"]).resolve() == adapter
assert float(adapter_config["lora_dropout"]) == 0.0
assert metadata["adapter_model_sha256"] == adapter_sha256
assert Path(metadata["adapter_path"]).resolve() == adapter
assert Path(metadata["output_dir"]).resolve() == merged
assert metadata["source_final_global_step"] == 20000
assert metadata["parameter_scope"] == "language_decoder_only"
assert metadata["lora_dropout"] == 0.0
assert merged.joinpath("model.safetensors.index.json").is_file()
assert len(list(merged.glob("model-*.safetensors"))) == 4
print(json.dumps({
    "status": "artifact_preflight_passed",
    "adapter": str(adapter),
    "adapter_sha256": adapter_sha256,
    "merged": str(merged),
}, indent=2))
PY

if [[ "${CASE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight_complete task=${TASK} ratio=${RATIO_TAG}"
  exit 0
fi

export CKPT_EVAL_ROOT OUT_ROOT OPSD_ROOT
export CKPT_EVAL_OUT_ROOT="${OUT_ROOT}"
export EXPECTED_BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export EXPECTED_ADAPTER_PATH="${ADAPTER_PATH}"
export MERGED_MODEL METHOD_TAG EVAL_TASK RATIO_TAG CAMPAIGN_GROUP
export LOG_ROOT="${OUT_ROOT}/logs"
export EVAL_NPROC_PER_NODE=4
export MMMU_PRO_TOTAL_MIN_IMAGE_TOKENS=1280
export MMMU_PRO_TOTAL_MAX_IMAGE_TOKENS=4096
export max_new_tokens=2048
export min_pixels=$((1280 * 28 * 28))
export max_pixels=$((4096 * 28 * 28))
export MMSTAR_QWEN_JUDGE_SCOPE=misses
export VLMEVAL_SAVE_RAW_PREDICTION=1
export VLMEVAL_STRICT_ERRORS=1
export TOKENIZERS_PARALLELISM=false

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-manual} task=${TASK} ratio=${RATIO_TAG}"
echo "git=$(git -C "${OPSD_ROOT}" rev-parse HEAD)"
nvidia-smi

WORK_DIR="${OUT_ROOT}/eval_vlmevalkit_trainenv/${CAMPAIGN_GROUP}/${METHOD_TAG}_${RATIO_TAG}_${DATASET_NAME}_${SLURM_JOB_ID:-manual}"
if [[ "${TASK}" != "mathverse" ]]; then
  bash "${ENGINE}"
else
  export MODEL_NAME=Qwen
  export model_path="${MERGED_MODEL}"
  export adapter_path=""
  export ADAPTER_TAG="${METHOD_TAG}"
  export enable_thinking=True
  export enable_visionzip visionzip_ratio retention_ratio
  export temperature=0.0
  export num_return_sequences=1
  export use_kv_cache=True
  export VISIONZIP_ATTN_CHUNK=128
  export MATH_QWEN_JUDGE_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
  export MATH_QWEN_JUDGE_DEVICE=cuda:0
  export MATH_QWEN_JUDGE_DTYPE=bfloat16
  export MATH_QWEN_JUDGE_MAX_NEW_TOKENS=128
  export WORK_DIR
  export MASTER_PORT="$((35500 + (${SLURM_JOB_ID:-0} % 2000)))"
  mkdir -p "${WORK_DIR}"
  source "${CKPT_EVAL_ROOT}/env_train_runtime.sh"
  cd "${VLM_ROOT}"
  python - <<'PY'
from vlmeval.dataset import DATASET_TYPE, build_dataset

dataset = build_dataset("MathVerse_MINI_Vision_Only")
assert type(dataset).__name__ == "MathVerse"
assert DATASET_TYPE("MathVerse_MINI_Vision_Only") == "VQA"
assert len(dataset) == 788
print("MathVerse preflight passed", len(dataset))
PY
  python -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${EVAL_NPROC_PER_NODE}" \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    run.py \
      --data MathVerse_MINI_Vision_Only \
      --model Qwen \
      --mode infer \
      --work-dir "${WORK_DIR}"
  result_file="$(find "${WORK_DIR}/Qwen" -type f -name 'Qwen_MathVerse_MINI_Vision_Only.xlsx' | sort | tail -n 1)"
  [[ -n "${result_file}" && -f "${result_file}" ]]
  python "${OPSD_ROOT}/scripts/score_math_vqa_with_qwen7.py" \
    --dataset MathVerse_MINI_Vision_Only \
    --result-file "${result_file}"
  python - "${WORK_DIR}" "${MERGED_MODEL}" "${RATIO_TAG}" "${retention_ratio}" <<'PY'
import json
import os
import sys
from pathlib import Path

work, model, ratio, retention = sys.argv[1:]
payload = {
    "commit": "51682a6baab948d3dbb4b867a3eab178504ac3f5",
    "dataset": "MathVerse_MINI_Vision_Only",
    "dataset_rows": 788,
    "method": "original_opsd_dropout0_step20000_merged",
    "model_path": model,
    "checkpoint_load_form": "merged_full_checkpoint",
    "ratio_tag": ratio,
    "retention_ratio": float(retention),
    "enable_thinking": True,
    "min_pixels": 1280 * 28 * 28,
    "max_pixels": 4096 * 28 * 28,
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "use_kv_cache": True,
    "postprocess": "mathverse_qwen7",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
}
Path(work).joinpath("run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
  cd "${OPSD_ROOT}"
fi

result_file="$(find "${WORK_DIR}/Qwen" -type f -name "Qwen_${DATASET_NAME}.xlsx" | sort | tail -n 1)"
[[ -n "${result_file}" && -f "${result_file}" ]]
source "${CKPT_EVAL_ROOT}/env_train_runtime.sh"

if [[ "${TASK}" == "mmmupro" ]]; then
  export MMMU_QWEN_JUDGE_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
  export MMMU_QWEN_JUDGE_DEVICE=cuda:0
  export MMMU_QWEN_JUDGE_DTYPE=bfloat16
  export MMMU_QWEN_JUDGE_MAX_NEW_TOKENS=8
  python "${CAMPAIGN_ROOT}/score_mmmupro_with_qwen7.py" --result-file "${result_file}"
fi

python "${VALIDATOR}" \
  --result-file "${result_file}" \
  --expected-rows "${EXPECTED_ROWS}" \
  --report-out "${WORK_DIR}/raw_prediction_validation.json"
python "${CAMPAIGN_ROOT}/collect_case.py" \
  --work-dir "${WORK_DIR}" \
  --dataset "${DATASET_NAME}" \
  --ratio "${RATIO_TAG}" \
  --job-id "${SLURM_JOB_ID:-manual}"

echo "completed=$(date --iso-8601=seconds) task=${TASK} ratio=${RATIO_TAG}"
