#!/usr/bin/env bash
#SBATCH --job-name=epic_debug32
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/debug_epic_actual_train_20260621_0027/slurm-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/debug_epic_actual_train_20260621_0027/slurm-%j.err

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/project/6101803/enmingzz/opsd}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/6101803/enmingzz}"
OUT_ROOT="${OUT_ROOT:-/scratch/enmingzz/temp/outputs/vsi_bench}"
LOG_ROOT="${LOG_ROOT:-/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/debug_epic_actual_train_20260621_0027}"
RUN_GROUP="${RUN_GROUP:-debug_epic_actual_train_20260621_0027}"
STAGE_NAME="${STAGE_NAME:-epic_debug32}"
CONFIG_PATH="${CONFIG_PATH:-${OPSD_ROOT}/configs/visionzip_aokvqa/aokvqa_epic_tcd_reasoning_ebs32_4l40s.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MAX_STEPS="${MAX_STEPS:-32}"
LIMIT="${LIMIT:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 10000))}"

cd "${OPSD_ROOT}"

source "${PROJECT_ROOT}/env/vsi-official.sh"
export HF_HOME="${HF_HOME:-/home/enmingzz/scratch/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB034_ROOT="${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}"
export TOKENIZERS_QWEN25_ROOT="${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}"
export VSI_OFFICIAL_SITE_PACKAGES="${VSI_OFFICIAL_SITE_PACKAGES:-/scratch/enmingzz/temp/venvs/vsi-official/lib/python3.11/site-packages}"
export ARMEN_TRANSFORMERS_SRC="${ARMEN_TRANSFORMERS_SRC:-${OPSD_ROOT}/third_party/VLMEvalKit_armen51682/transformers/src}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

SANITIZED_PYTHONPATH=""
if [[ -n "${PYTHONPATH:-}" ]]; then
  IFS=':' read -r -a PYTHONPATH_PARTS <<< "${PYTHONPATH}"
  for path in "${PYTHONPATH_PARTS[@]}"; do
    if [[ -z "${path}" || "${path}" == /scratch/enmingzz/temp/qwen25_bootstrap* ]]; then
      continue
    fi
    SANITIZED_PYTHONPATH="${SANITIZED_PYTHONPATH:+${SANITIZED_PYTHONPATH}:}${path}"
  done
fi
export PYTHONPATH="${PROJECT_ROOT}:${ARMEN_TRANSFORMERS_SRC}:${HF_HUB034_ROOT}:${TOKENIZERS_QWEN25_ROOT}:${VSI_OFFICIAL_SITE_PACKAGES}${SANITIZED_PYTHONPATH:+:${SANITIZED_PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export VISIONZIP_QWEN25VL_ROOT="${VISIONZIP_QWEN25VL_ROOT:-${OPSD_ROOT}/third_party/VisionZip/Qwen2_5_VL}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OPSD_DDP_TIMEOUT_MINUTES="${OPSD_DDP_TIMEOUT_MINUTES:-120}"
export OPSD_DDP_STAGGER_LOAD_SECONDS="${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}"

OUT_DIR="${OUT_ROOT}/checkpoints/${RUN_GROUP}/${STAGE_NAME}"
mkdir -p "${OUT_DIR}" "${LOG_ROOT}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} stage=${STAGE_NAME}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LIMIT=${LIMIT}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"

python - "${CONFIG_PATH}" "${NPROC_PER_NODE}" "${MAX_STEPS}" "${MAX_NEW_TOKENS}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
world = int(sys.argv[2])
max_steps = int(sys.argv[3])
max_new_tokens = int(sys.argv[4])
micro = int(cfg["training"]["micro_batch_size"])
accum = int(cfg["training"]["gradient_accumulation_steps"])
eff = world * micro * accum
min_pixels = int(cfg["dataset"]["min_pixels"])
max_pixels = int(cfg["dataset"]["max_pixels"])
print(
    {
        "method": cfg["training"]["method"],
        "prompt_enable_thinking": cfg.get("prompt", {}).get("enable_thinking"),
        "attn": cfg["training"]["attn_implementation"],
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "config_max_new_tokens": cfg["generation"]["max_new_tokens"],
        "run_max_new_tokens": max_new_tokens,
        "max_unparseable_new_tokens": cfg["generation"].get("max_unparseable_new_tokens"),
        "generation_temperature": cfg["generation"]["temperature"],
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": eff,
        "max_steps": max_steps,
        "epic": cfg.get("epic", {}),
    }
)
assert cfg["training"]["method"] == "epic"
assert eff == 32, eff
assert max_steps % world == 0, (max_steps, world)
assert min_pixels == 1280 * 28 * 28, min_pixels
assert max_pixels == 16384 * 28 * 28, max_pixels
assert int(max_new_tokens) == 1024
assert float(cfg["generation"]["temperature"]) == 0.0
assert cfg.get("prompt", {}).get("enable_thinking") is True
assert cfg.get("epic", {}).get("teacher_retention_policy") == "full"
PY

python - <<'PY'
from opsd.visionzip_aokvqa.qwen_wrapper import import_qwen25_modules

import_qwen25_modules()

import datasets
import flash_attn
import huggingface_hub
import PIL
import pyarrow
import tokenizers
import transformers

print(
    {
        "flash_attn": getattr(flash_attn, "__version__", "ok"),
        "huggingface_hub": huggingface_hub.__version__,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
        "PIL": PIL.__version__,
        "pyarrow": pyarrow.__version__,
        "datasets": datasets.__version__,
    }
)
PY

python -m torch.distributed.run \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-port="${MASTER_PORT}" \
  visionzip_aokvqa/train.py \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --limit "${LIMIT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}"

echo "[$(date --iso-8601=seconds)] done stage=${STAGE_NAME}"
