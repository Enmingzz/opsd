#!/usr/bin/env bash
#SBATCH --job-name=pure_opsd_no_unparse
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/train/aokvqa_hires_reasoning_ebs32_4l40s_cleanenv_pureopsd_max512_20260620_015909/slurm-pure_opsd_max512-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/train/aokvqa_hires_reasoning_ebs32_4l40s_cleanenv_pureopsd_max512_20260620_015909/slurm-pure_opsd_max512-%j.err

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/project/6101803/enmingzz/opsd}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/6101803/enmingzz}"
OUT_ROOT="${OUT_ROOT:-/scratch/enmingzz/temp/outputs/vsi_bench}"
LOG_ROOT="${LOG_ROOT:-/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/train}"
RUN_GROUP="${RUN_GROUP:-aokvqa_hires_reasoning_ebs32_4l40s_cleanenv_pureopsd_max512_20260620_015909}"
STAGE_NAME="${STAGE_NAME:-01_opsd_ema_teacher_nogt_max512_ebs32}"
CONFIG_PATH="${CONFIG_PATH:-${OPSD_ROOT}/configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_reasoning_ebs32_4l40s_no_unparseable.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 10000))}"
START_STEP="${START_STEP:-0}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

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
export OPSD_DDP_TIMEOUT_MINUTES="${OPSD_DDP_TIMEOUT_MINUTES:-720}"
export OPSD_DDP_STAGGER_LOAD_SECONDS="${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}"

OUT_DIR="${OUT_ROOT}/checkpoints/${RUN_GROUP}/${STAGE_NAME}"
LOG_DIR="${LOG_ROOT}/${RUN_GROUP}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} stage=${STAGE_NAME}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "START_STEP=${START_STEP}"
echo "ADAPTER_PATH=${ADAPTER_PATH:-<none>}"

python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
world = 4
micro = int(cfg["training"]["micro_batch_size"])
accum = int(cfg["training"]["gradient_accumulation_steps"])
eff = world * micro * accum
min_pixels = int(cfg["dataset"]["min_pixels"])
max_pixels = int(cfg["dataset"]["max_pixels"])
max_unparseable = int(cfg["generation"].get("max_unparseable_new_tokens", -1))
raw_stop_on_parse = cfg["generation"].get("stop_on_parse", True)
stop_on_parse = (
    raw_stop_on_parse.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(raw_stop_on_parse, str)
    else bool(raw_stop_on_parse)
)
print(
    {
        "method": cfg["training"]["method"],
        "prompt_enable_thinking": cfg.get("prompt", {}).get("enable_thinking"),
        "attn": cfg["training"]["attn_implementation"],
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_new_tokens": cfg["generation"]["max_new_tokens"],
        "max_unparseable_new_tokens": max_unparseable,
        "stop_on_parse": stop_on_parse,
        "generation_temperature": cfg["generation"]["temperature"],
        "manual_pruned_generate": cfg["generation"].get("manual_pruned_generate"),
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accum,
        "effective_batch_size_assuming_4_gpus": eff,
        "train_retention_ratios": cfg["pruning"]["train_retention_ratios"],
        "opsd": cfg.get("opsd", {}),
    }
)
assert eff == 32, eff
assert min_pixels == 1280 * 28 * 28, min_pixels
assert max_pixels == 16384 * 28 * 28, max_pixels
assert int(cfg["generation"]["max_new_tokens"]) == 512
assert max_unparseable == 0, max_unparseable
assert stop_on_parse is False, stop_on_parse
assert float(cfg["generation"]["temperature"]) == 0.0
assert cfg.get("prompt", {}).get("enable_thinking") is True
assert cfg["training"]["method"] == "opsd_nogt"
assert cfg.get("opsd", {}).get("teacher_strategy") == "ema"
assert cfg.get("opsd", {}).get("use_ema_teacher") is True
assert float(cfg.get("opsd", {}).get("ema_decay")) == 0.9999
PY

python - <<'PY'
from opsd.visionzip_aokvqa.qwen_wrapper import import_qwen25_modules

import_qwen25_modules()

import sys

import flash_attn
import huggingface_hub
import PIL
import tokenizers
import transformers

disallowed = "/scratch/enmingzz/temp/qwen25_bootstrap"
loaded = {
    "huggingface_hub": huggingface_hub.__file__,
    "tokenizers": tokenizers.__file__,
    "transformers": transformers.__file__,
    "PIL": PIL.__file__,
}
assert not any(path.startswith(disallowed) for path in sys.path if path), sys.path
for package, path in loaded.items():
    assert not str(path).startswith(disallowed), (package, path)
print(
    {
        "flash_attn": getattr(flash_attn, "__version__", "ok"),
        "huggingface_hub": huggingface_hub.__version__,
        "huggingface_hub_file": huggingface_hub.__file__,
        "tokenizers": tokenizers.__version__,
        "tokenizers_file": tokenizers.__file__,
        "transformers": transformers.__version__,
        "transformers_file": transformers.__file__,
        "PIL": PIL.__version__,
        "PIL_file": PIL.__file__,
        "bootstrap_in_sys_path": any(path.startswith(disallowed) for path in sys.path if path),
    }
)
PY

TRAIN_ARGS=(
  --config "${CONFIG_PATH}"
  --output_dir "${OUT_DIR}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)
if [[ "${START_STEP}" != "0" ]]; then
  TRAIN_ARGS+=(--start_step "${START_STEP}")
fi
if [[ -n "${ADAPTER_PATH}" ]]; then
  TRAIN_ARGS+=(--adapter_path "${ADAPTER_PATH}")
fi

python -m torch.distributed.run \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-port="${MASTER_PORT}" \
  visionzip_aokvqa/train.py \
  "${TRAIN_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] done stage=${STAGE_NAME}"
