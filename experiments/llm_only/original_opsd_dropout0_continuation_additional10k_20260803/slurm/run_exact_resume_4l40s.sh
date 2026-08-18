#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG_PATH:?CONFIG_PATH must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"

OPSD_ROOT="${OPSD_ROOT:-/project/6101803/enmingzz/opsd}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/6101803/enmingzz}"
EXPERIMENT_ROOT="${OPSD_ROOT}/experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803"
JOB_ID_VALUE="${SLURM_JOB_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((29500 + JOB_ID_VALUE % 10000))}"

if [[ ! -f "${OUT_DIR}/resume_checkpoints/step_009984/COMPLETE" ]]; then
  echo "Prepared exact-resume bridge is missing: ${OUT_DIR}" >&2
  exit 1
fi
if [[ -e "${OUT_DIR}/final" ]]; then
  echo "Refusing to overwrite a completed output: ${OUT_DIR}/final" >&2
  exit 1
fi
cd "${OPSD_ROOT}"

source "${PROJECT_ROOT}/env/vsi-official.sh"

export HF_HOME="${HF_HOME:-/home/enmingzz/scratch/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB034_ROOT="${HF_HUB034_ROOT:-/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2}"
export TOKENIZERS_QWEN25_ROOT="${TOKENIZERS_QWEN25_ROOT:-/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only}"
export ARMEN_TRANSFORMERS_SRC="${ARMEN_TRANSFORMERS_SRC:-${OPSD_ROOT}/third_party/VLMEvalKit_armen51682/transformers/src}"
export VISIONZIP_QWEN25VL_ROOT="${VISIONZIP_QWEN25VL_ROOT:-${PROJECT_ROOT}/VisionZip/Qwen2_5_VL}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OPSD_DDP_TIMEOUT_MINUTES="${OPSD_DDP_TIMEOUT_MINUTES:-120}"
export OPSD_DDP_STAGGER_LOAD_SECONDS="${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

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
export PYTHONPATH="${PROJECT_ROOT}${SANITIZED_PYTHONPATH:+:${SANITIZED_PYTHONPATH}}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "git_commit=$(git rev-parse HEAD)"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python "${EXPERIMENT_ROOT}/prepare_exact_resume.py" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUT_DIR}" \
  --report "${OUT_DIR}/resume_fork_validation_at_launch.json" \
  --check-only

python - <<'PY'
from opsd.visionzip_aokvqa.qwen_wrapper import import_qwen25_modules

import_qwen25_modules()
import flash_attn
import PIL
import tokenizers
import transformers

assert "opsd/third_party/VLMEvalKit_armen51682/transformers/src" in str(transformers.__file__)
print({
    "flash_attn": getattr(flash_attn, "__version__", "ok"),
    "tokenizers": tokenizers.__version__,
    "transformers": transformers.__version__,
    "transformers_file": transformers.__file__,
    "PIL": PIL.__version__,
})
PY

python - "${CONFIG_PATH}" "${OUT_DIR}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
diff = subprocess.run(["git", "diff", "--binary"], check=True, capture_output=True).stdout
manifest = {
    "status": "exact_resume_prelaunch_validated",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "dirty_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "config_path": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "resume_checkpoint": config["checkpointing"]["resume_from"],
    "parent_checkpoint": config["checkpointing"]["resume_fork"]["parent_checkpoint"],
    "resume_global_step": 9984,
    "target_global_step": int(config["training"]["max_steps"]),
    "optimizer_state": "inherited",
    "rank_rng_state": "inherited",
    "ema_state": "inherited",
    "lora_dropout": float(config["training"]["lora_dropout"]),
    "learning_rate": float(config["training"]["learning_rate"]),
    "learning_rate_schedule": "none (constant AdamW learning rate)",
    "effective_batch_size": 32,
}
(output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

torchrun \
  --nproc-per-node=4 \
  --master-port="${MASTER_PORT}" \
  visionzip_aokvqa/train.py \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUT_DIR}"

python "${OPSD_ROOT}/experiments/scripts/verify_lora_scope.py" \
  --config "${CONFIG_PATH}" \
  --expected-scope language_decoder_only \
  --adapter-path "${OUT_DIR}/final" \
  --output "${OUT_DIR}/final_scope_verification.json" \
  --overwrite

python "${EXPERIMENT_ROOT}/audit_completed_run.py" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUT_DIR}"

echo "[$(date --iso-8601=seconds)] exact resume completed and audited"
