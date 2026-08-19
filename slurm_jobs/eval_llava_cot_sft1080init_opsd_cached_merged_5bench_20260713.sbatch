#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=e_s1080opkv
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_llava_cot_reasoning/logs/eval/lcot_sft1080init_opsd_cached_merged_5bench_20260713/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_llava_cot_reasoning/logs/eval/lcot_sft1080init_opsd_cached_merged_5bench_20260713/%x-%j.err

set -euo pipefail

ROOT="${CKPT_EVAL_ROOT:-/project/6101803/enmingzz/ckpt_eval_trainenv}"
OPSD_ROOT="${OPSD_ROOT:-/project/6101803/enmingzz/opsd}"
OUT_ROOT="${CKPT_EVAL_OUT_ROOT:-/scratch/enmingzz/outputs/visionzip_llava_cot_reasoning}"
EXPECTED_BASE_MODEL="${EXPECTED_BASE_MODEL:-${OUT_ROOT}/merged_models/qwen25vl7b_lcot_sft1080_final_merged_bf16_20260707}"
EXPECTED_ADAPTER_PATH="${EXPECTED_ADAPTER_PATH:-${OUT_ROOT}/checkpoints/llava_cot_sft1080merged_opsd_cachedrollout_qwentok512_maxpix1080_reasoning_ebs32_cleanarmen_20260712/llava_cot_sft1080merged_opsd_nogt_ema_maxpix1080_progressive_40to10_cachedrollout_ebs32/final}"
MERGED_MODEL="${MERGED_MODEL:-${OUT_ROOT}/merged_models/qwen25vl7b_lcot_sft1080init_opsd_cached_final_merged_bf16_20260713}"
METHOD_TAG="${METHOD_TAG:-lcot_sft1080init_opsd_cached_merged}"
EVAL_TASK="${EVAL_TASK:?Set EVAL_TASK to mme, mmstar, mathvista, mathverse, mathvision, or mmmupro}"
RATIO_TAG="${RATIO_TAG:?Set RATIO_TAG to noprune, r010, r020, or r030}"

case "${RATIO_TAG}" in
  noprune)
    enable_visionzip=False
    visionzip_ratio=0.0
    retention_ratio=1.0
    ;;
  r010)
    enable_visionzip=True
    visionzip_ratio=0.95
    retention_ratio=0.1
    ;;
  r020)
    enable_visionzip=True
    visionzip_ratio=0.85
    retention_ratio=0.2
    ;;
  r030)
    enable_visionzip=True
    visionzip_ratio=0.75
    retention_ratio=0.3
    ;;
  *)
    echo "Unknown RATIO_TAG=${RATIO_TAG}" >&2
    exit 1
    ;;
esac

case "${EVAL_TASK,,}" in
  mme)
    DATASET_NAME=MME
    EXPECTED_ROWS=2374
    POSTPROCESS=mme_qwen7yn
    ;;
  mmstar)
    DATASET_NAME=MMStar
    EXPECTED_ROWS=1500
    POSTPROCESS=integrated_mmstar_qwen7
    ;;
  mathvista|mathvista_mini)
    DATASET_NAME=MathVista_MINI
    EXPECTED_ROWS=1000
    POSTPROCESS=mathvista_qwen7
    ;;
  mathverse|mathverse_mini_vision_only)
    DATASET_NAME=MathVerse_MINI_Vision_Only
    EXPECTED_ROWS=788
    POSTPROCESS=mathverse_qwen7
    ;;
  mathvision)
    DATASET_NAME=MathVision
    EXPECTED_ROWS=3040
    POSTPROCESS=mathvision_qwen7
    ;;
  mmmupro|mmmu_pro|mmmu_pro_4c)
    DATASET_NAME=MMMU_Pro_4c
    EXPECTED_ROWS=1730
    POSTPROCESS=official_mcq
    ;;
  *)
    echo "Unknown EVAL_TASK=${EVAL_TASK}" >&2
    exit 1
    ;;
esac

LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs/eval/lcot_sft1080init_opsd_cached_merged_5bench_20260713}"
CAMPAIGN_GROUP="${CAMPAIGN_GROUP:-lcot_sft1080init_opsd_cached_merged_reasoning_5bench_cleanarmen_20260713}"
CASE_TAG="${METHOD_TAG}_${RATIO_TAG}_${DATASET_NAME}_${SLURM_JOB_ID:-manual}"
RUN_GROUP="${CAMPAIGN_GROUP}/${CASE_TAG}"
WORK_DIR="${OUT_ROOT}/eval_vlmevalkit_trainenv/${RUN_GROUP}"

mkdir -p "${LOG_ROOT}" "${WORK_DIR}"

export CKPT_EVAL_ROOT="${ROOT}"
export CKPT_EVAL_OUT_ROOT="${OUT_ROOT}"
export OPSD_ROOT
export MODEL_NAME=Qwen
export model_path="${MERGED_MODEL}"
export adapter_path=""
export ADAPTER_TAG="${METHOD_TAG}"
export RATIO_TAG
export retention_ratio
export enable_thinking=True
export enable_visionzip
export visionzip_ratio
export temperature=0.0
export num_return_sequences=1
export use_kv_cache=True
export VLMEVAL_SAVE_RAW_PREDICTION=1
export VLMEVAL_STRICT_ERRORS=1
export VISIONZIP_ATTN_CHUNK="${VISIONZIP_ATTN_CHUNK:-128}"
export EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
export EVAL_SPLIT_DATASETS=1
export EVAL_DATASETS="${DATASET_NAME}"
export RUN_GROUP
export EVAL_NAME="${RUN_GROUP}"
export WORK_DIR
export LOG_DIR="${LOG_ROOT}"
export MASTER_PORT="$((30000 + (${SLURM_JOB_ID:-0} % 10000)))"

export MMSTAR_QWEN_JUDGE=1
export MMSTAR_QWEN_JUDGE_MODEL_PATH="${MMSTAR_QWEN_JUDGE_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export MMSTAR_QWEN_JUDGE_DEVICE="${MMSTAR_QWEN_JUDGE_DEVICE:-cuda:0}"
export MMSTAR_QWEN_JUDGE_DTYPE="${MMSTAR_QWEN_JUDGE_DTYPE:-bfloat16}"
export MMSTAR_QWEN_JUDGE_SCOPE="${MMSTAR_QWEN_JUDGE_SCOPE:-misses}"
export MMSTAR_QWEN_JUDGE_ALLOW_OPEN_ANSWER=1
export MMSTAR_QWEN_JUDGE_STRICT=1
export MMSTAR_QWEN_JUDGE_MAX_NEW_TOKENS=16

export QWEN_YN_JUDGE_MODEL_PATH="${QWEN_YN_JUDGE_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export QWEN_YN_JUDGE_DEVICE="${QWEN_YN_JUDGE_DEVICE:-cuda:0}"
export QWEN_YN_JUDGE_DTYPE="${QWEN_YN_JUDGE_DTYPE:-bfloat16}"
export QWEN_YN_JUDGE_MAX_NEW_TOKENS=16

export MATH_QWEN_JUDGE_MODEL_PATH="${MATH_QWEN_JUDGE_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export MATH_QWEN_JUDGE_DEVICE="${MATH_QWEN_JUDGE_DEVICE:-cuda:0}"
export MATH_QWEN_JUDGE_DTYPE="${MATH_QWEN_JUDGE_DTYPE:-bfloat16}"
export MATH_QWEN_JUDGE_MAX_NEW_TOKENS=128

export MMMU_PRO_TOTAL_MIN_IMAGE_TOKENS="${MMMU_PRO_TOTAL_MIN_IMAGE_TOKENS:-1280}"
export MMMU_PRO_TOTAL_MAX_IMAGE_TOKENS="${MMMU_PRO_TOTAL_MAX_IMAGE_TOKENS:-4096}"
export EXPECTED_BASE_MODEL EXPECTED_ADAPTER_PATH MERGED_MODEL DATASET_NAME EXPECTED_ROWS POSTPROCESS

python "${OPSD_ROOT}/scripts/eval/validate_eval_load_mode.py" \
  --model-path "${MERGED_MODEL}" \
  --adapter-path "" \
  --load-mode merged_full_checkpoint \
  --method "${METHOD_TAG}" \
  --dataset "${DATASET_NAME}" \
  --ratio "${RATIO_TAG}" \
  --manifest-out "${WORK_DIR}/checkpoint_load_validation.json"

source "${ROOT}/env_train_runtime.sh"
cd "${VLM_ROOT}"

python - <<'PY'
from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path

import PIL
import pyarrow
import tokenizers
import transformers
from vlmeval.config import supported_VLM
from vlmeval.dataset import DATASET_TYPE, build_dataset
from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat


def as_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y")


model_path = Path(os.environ["MERGED_MODEL"])
assert model_path.joinpath("config.json").is_file(), model_path
assert model_path.joinpath("model.safetensors.index.json").is_file(), model_path
assert model_path.joinpath("merge_metadata.json").is_file(), model_path

if os.environ.get("SKIP_MERGE_PROVENANCE_CHECK", "0") != "1":
    metadata = json.loads(model_path.joinpath("merge_metadata.json").read_text())
    assert Path(metadata["base_model"]).resolve() == Path(os.environ["EXPECTED_BASE_MODEL"]).resolve(), metadata
    assert Path(metadata["adapter_path"]).resolve() == Path(os.environ["EXPECTED_ADAPTER_PATH"]).resolve(), metadata

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert commit == "51682a6baab948d3dbb4b867a3eab178504ac3f5", commit

kwargs = supported_VLM["Qwen"].keywords
assert kwargs["model_path"] == str(model_path), kwargs["model_path"]
assert kwargs["adapter_path"] == "", kwargs["adapter_path"]
assert kwargs["min_pixels"] == 1280 * 28 * 28
assert kwargs["max_pixels"] == 4096 * 28 * 28
assert kwargs["use_custom_prompt"] is False
assert kwargs["enable_thinking"] is True
assert kwargs["enable_visionzip"] == as_bool(os.environ["enable_visionzip"])
assert abs(float(kwargs["visionzip_ratio"]) - float(os.environ["visionzip_ratio"])) < 1e-12
assert float(kwargs["temperature"]) == 0.0
assert kwargs["use_kv_cache"] is True
assert inspect.signature(Qwen2VLChat.__init__).parameters["max_new_tokens"].default == 2048
assert str(Path(transformers.__file__).resolve()).startswith(str(Path(os.environ["VLM_ROOT"]).resolve() / "transformers" / "src"))
assert tokenizers.__version__ == "0.22.2", tokenizers.__version__
assert PIL.__version__ == "9.5.0.post2", PIL.__version__
assert pyarrow.__version__ == "23.0.1", pyarrow.__version__

dataset_name = os.environ["DATASET_NAME"]
dset = build_dataset(dataset_name)
assert len(dset) == int(os.environ["EXPECTED_ROWS"]), (dataset_name, len(dset))
if dataset_name == "MMMU_Pro_4c":
    assert type(dset).__name__ == "MMMUProDataset", type(dset).__name__
    assert DATASET_TYPE(dataset_name) == "MCQ_MMMU_Pro"
    source = inspect.getsource(Qwen2VLChat._use_total_image_budget)
    budget_source = inspect.getsource(Qwen2VLChat._per_image_pixel_budget)
    assert "MMMU_PRO" in source
    assert "MMMU_PRO_TOTAL_MAX_IMAGE_TOKENS" in budget_source
    assert Path("/scratch/enmingzz/vlmevalkit_data/MMMU_Pro_4c.tsv").is_file()
    assert Path("/scratch/enmingzz/vlmevalkit_data/images/MMMU_Pro_4c").is_dir()

manifest = {
    "commit": commit,
    "dataset": dataset_name,
    "dataset_rows": len(dset),
    "dataset_type": DATASET_TYPE(dataset_name),
    "method": os.environ["ADAPTER_TAG"],
    "model_path": str(model_path),
    "adapter_path": "",
    "checkpoint_load_form": "merged_full_checkpoint",
    "checkpoint_load_form_evidence": "empty adapter_path plus full weights and merge_metadata.json",
    "ratio_tag": os.environ["RATIO_TAG"],
    "retention_ratio": float(os.environ["retention_ratio"]),
    "enable_thinking": True,
    "enable_visionzip": kwargs["enable_visionzip"],
    "visionzip_ratio_backend": kwargs["visionzip_ratio"],
    "min_pixels": kwargs["min_pixels"],
    "max_pixels": kwargs["max_pixels"],
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "use_kv_cache": True,
    "postprocess": os.environ["POSTPROCESS"],
    "save_raw_prediction": os.environ["VLMEVAL_SAVE_RAW_PREDICTION"],
    "work_dir": os.environ["WORK_DIR"],
}
Path(os.environ["WORK_DIR"]).joinpath("run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

if [[ "${EVAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[$(date --iso-8601=seconds)] evaluation preflight-only checks passed"
  exit 0
fi

run_distributed_infer() {
  python -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${EVAL_NPROC_PER_NODE}" \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    run.py \
      --data "${DATASET_NAME}" \
      --model "${MODEL_NAME}" \
      --mode infer \
      --work-dir "${WORK_DIR}"
}

find_result_file() {
  find "${WORK_DIR}/Qwen" -type f -name "Qwen_${DATASET_NAME}.xlsx" | sort | tail -n 1
}

case "${POSTPROCESS}" in
  mme_qwen7yn)
    bash "${ROOT}/eval_one.sh"
    result_file="$(find_result_file)"
    [[ -n "${result_file}" && -f "${result_file}" ]]
    python "${OPSD_ROOT}/scripts/posthoc_qwen7yn_judge_mme_pope.py" \
      --force \
      --out-dir "${WORK_DIR}/posthoc_qwen7yn" \
      --case "${METHOD_TAG}:${RATIO_TAG}:MME:${result_file}"
    ;;
  integrated_mmstar_qwen7|official_mcq)
    bash "${ROOT}/eval_one.sh"
    ;;
  mathvista_qwen7)
    run_distributed_infer
    result_file="$(find_result_file)"
    [[ -n "${result_file}" && -f "${result_file}" ]]
    python "${OPSD_ROOT}/scripts/score_mathvista_with_qwen7.py" --result-file "${result_file}"
    ;;
  mathverse_qwen7|mathvision_qwen7)
    run_distributed_infer
    result_file="$(find_result_file)"
    [[ -n "${result_file}" && -f "${result_file}" ]]
    python "${OPSD_ROOT}/scripts/score_math_vqa_with_qwen7.py" \
      --dataset "${DATASET_NAME}" \
      --result-file "${result_file}"
    ;;
  *)
    echo "Unsupported POSTPROCESS=${POSTPROCESS}" >&2
    exit 1
    ;;
esac

echo "[$(date --iso-8601=seconds)] finished method=${METHOD_TAG} ratio=${RATIO_TAG} dataset=${DATASET_NAME}"
