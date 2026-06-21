#!/usr/bin/env bash
#SBATCH --job-name=resume_01_opsd_ema
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/slurm-01_opsd_ema_teacher_nogt_ebs32-resume-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/slurm-01_opsd_ema_teacher_nogt_ebs32-resume-%j.err

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/project/6101803/enmingzz/opsd}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/6101803/enmingzz}"
VENV_ROOT="${VENV_ROOT:-/scratch/enmingzz/temp/venvs/vsi-official}"
OUT_ROOT="${OUT_ROOT:-/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning}"
RUN_GROUP="${RUN_GROUP:-aokvqa_hires_reasoning_ebs32_4l40s_armenvzfix_20260614_162619}"
STAGE_NAME="${STAGE_NAME:-01_opsd_ema_teacher_nogt_ebs32}"
CONFIG_PATH="${CONFIG_PATH:-${OPSD_ROOT}/configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_reasoning_ebs32_4l40s.yaml}"
RESUME_ADAPTER="${RESUME_ADAPTER:-${OUT_ROOT}/checkpoints/${RUN_GROUP}/${STAGE_NAME}/step_12288}"
START_STEP="${START_STEP:-12288}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 10000))}"

cd "${OPSD_ROOT}"

module load StdEnv/2023 python/3.11.5 scipy-stack/2026a cuda/12.2 arrow/21.0.0 >/dev/null 2>&1 || true

export PATH="${VENV_ROOT}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/home/enmingzz/scratch/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB034_ROOT="${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export VISIONZIP_QWEN25VL_ROOT="${VISIONZIP_QWEN25VL_ROOT:-${OPSD_ROOT}/third_party/VisionZip/Qwen2_5_VL}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OPSD_DDP_TIMEOUT_MINUTES="${OPSD_DDP_TIMEOUT_MINUTES:-120}"
export OPSD_DDP_STAGGER_LOAD_SECONDS="${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}"

OUT_DIR="${OUT_ROOT}/checkpoints/${RUN_GROUP}/${STAGE_NAME}"
mkdir -p "${OUT_DIR}" "${OUT_ROOT}/logs"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} stage=${STAGE_NAME}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "RESUME_ADAPTER=${RESUME_ADAPTER}"
echo "START_STEP=${START_STEP}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"

test -f "${RESUME_ADAPTER}/adapter_model.safetensors"
test -f "${RESUME_ADAPTER}/ema_shadow.pt"

python - "${CONFIG_PATH}" "${START_STEP}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
start_step = int(sys.argv[2])
world = 4
micro = int(cfg["training"]["micro_batch_size"])
accum = int(cfg["training"]["gradient_accumulation_steps"])
max_steps = int(cfg["training"]["max_steps"])
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
        "max_new_tokens": cfg["generation"]["max_new_tokens"],
        "max_unparseable_new_tokens": cfg["generation"].get("max_unparseable_new_tokens"),
        "generation_temperature": cfg["generation"]["temperature"],
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accum,
        "effective_batch_size_assuming_4_gpus": eff,
        "max_steps": max_steps,
        "start_step": start_step,
        "remaining_global_steps": max_steps - start_step,
        "opsd": cfg.get("opsd", {}),
    }
)
assert eff == 32, eff
assert min_pixels == 1280 * 28 * 28, min_pixels
assert max_pixels == 16384 * 28 * 28, max_pixels
assert int(cfg["generation"]["max_new_tokens"]) == 1024
assert int(cfg["generation"].get("max_unparseable_new_tokens", 0)) == 256
assert float(cfg["generation"]["temperature"]) == 0.0
assert cfg.get("prompt", {}).get("enable_thinking") is True
assert cfg["training"]["method"] == "opsd_nogt"
assert cfg.get("opsd", {}).get("teacher_strategy") == "ema"
assert cfg.get("opsd", {}).get("use_ema_teacher") is True
assert float(cfg.get("opsd", {}).get("ema_decay")) == 0.9999
assert start_step % world == 0
assert (max_steps - start_step) % world == 0
PY

python - <<'PY'
from opsd.visionzip_aokvqa.qwen_wrapper import import_qwen25_modules

import_qwen25_modules()

import flash_attn
import huggingface_hub
import tokenizers
import transformers

print(
    {
        "flash_attn": getattr(flash_attn, "__version__", "ok"),
        "huggingface_hub": huggingface_hub.__version__,
        "huggingface_hub_file": huggingface_hub.__file__,
        "tokenizers": tokenizers.__version__,
        "tokenizers_file": tokenizers.__file__,
        "transformers": transformers.__version__,
        "transformers_file": transformers.__file__,
    }
)
PY

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-port="${MASTER_PORT}" \
  visionzip_aokvqa/train.py \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUT_DIR}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --adapter_path "${RESUME_ADAPTER}" \
  --start_step "${START_STEP}"

echo "[$(date --iso-8601=seconds)] done stage=${STAGE_NAME}"
