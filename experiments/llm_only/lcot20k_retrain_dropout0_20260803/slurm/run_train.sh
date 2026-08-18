#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CONFIG OUT_DIR {progressive|sft} EXPECTED_STOP {true|false}" >&2
  exit 2
fi

CONFIG_PATH="$(realpath "$1")"
OUT_DIR="$2"
RUN_KIND="$3"
EXPECTED_STOP="$4"
EXPECT_FINAL="$5"

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
EXPERIMENT_ROOT="${OPSD_ROOT}/experiments/llm_only/lcot20k_retrain_dropout0_20260803"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
JOB_ID_VALUE="${SLURM_JOB_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((29500 + JOB_ID_VALUE % 10000))}"

if [[ "${RUN_KIND}" != progressive && "${RUN_KIND}" != sft ]]; then
  echo "invalid run kind: ${RUN_KIND}" >&2
  exit 2
fi
if [[ "${EXPECT_FINAL}" != true && "${EXPECT_FINAL}" != false ]]; then
  echo "EXPECT_FINAL must be true or false" >&2
  exit 2
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
echo "RUN_KIND=${RUN_KIND} EXPECTED_STOP=${EXPECTED_STOP} EXPECT_FINAL=${EXPECT_FINAL}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python "${EXPERIMENT_ROOT}/validate_configs.py"

python - "${CONFIG_PATH}" "${OUT_DIR}" "${EXPECTED_STOP}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
output = Path(sys.argv[2]).resolve()
expected_stop = int(sys.argv[3])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if int(config["checkpointing"]["stop_at_step"]) != expected_stop:
    raise RuntimeError("launcher expected-stop does not match config")
resume = str(config["checkpointing"].get("resume_from", "") or "").strip()
if resume:
    checkpoint = Path(resume).resolve()
    if not (checkpoint / "COMPLETE").is_file():
        raise RuntimeError(f"resume checkpoint is incomplete: {checkpoint}")
    if checkpoint.parent.parent.resolve() != output:
        raise RuntimeError("resume output directory does not match checkpoint owner")
else:
    stale = [
        output / "training_log.jsonl",
        output / "resume_checkpoints",
        output / "eval_snapshots",
        output / "final",
    ]
    if any(path.exists() for path in stale):
        raise RuntimeError(f"refusing to overwrite nonempty training output: {output}")
output.mkdir(parents=True, exist_ok=True)
print(json.dumps({"status": "launch_preflight_passed", "resume_from": resume, "output": str(output)}))
PY

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

python - "${CONFIG_PATH}" "${OUT_DIR}" "${RUN_KIND}" "${EXPECTED_STOP}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
kind = sys.argv[3]
stop = int(sys.argv[4])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
diff = subprocess.run(["git", "diff", "--binary"], check=True, capture_output=True).stdout
manifest = {
    "status": "prelaunch_validated",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "dirty_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "config_path": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "kind": kind,
    "stop_at_step": stop,
    "training_horizon": int(config["training"]["max_steps"]),
    "resume_from": str(config["checkpointing"].get("resume_from", "")),
    "dataset": config["dataset"]["name"],
    "dataset_shuffle": config["dataset"]["shuffle"],
    "lora_dropout": float(config["training"]["lora_dropout"]),
    "learning_rate": float(config["training"]["learning_rate"]),
    "learning_rate_schedule": "none (constant AdamW learning rate)",
    "parameter_scope": config["experiment"]["parameter_scope"],
    "world_size": int(os.environ.get("NPROC_PER_NODE", "4")),
}
path = output / f"launch_manifest_step_{stop:06d}.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-port="${MASTER_PORT}" \
  visionzip_aokvqa/train.py \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUT_DIR}"

printf -v STOP_PAD '%06d' "${EXPECTED_STOP}"
CHECKPOINT="${OUT_DIR}/resume_checkpoints/step_${STOP_PAD}"
python "${OPSD_ROOT}/experiments/scripts/verify_lora_scope.py" \
  --config "${CONFIG_PATH}" \
  --expected-scope language_decoder_only \
  --adapter-path "${CHECKPOINT}" \
  --output "${OUT_DIR}/scope_verification_step_${STOP_PAD}.json" \
  --overwrite

AUDIT_ARGS=(
  --config "${CONFIG_PATH}"
  --output-dir "${OUT_DIR}"
  --kind "${RUN_KIND}"
  --expected-stop "${EXPECTED_STOP}"
)
if [[ "${EXPECT_FINAL}" == true ]]; then
  AUDIT_ARGS+=(--expect-final)
fi
python "${EXPERIMENT_ROOT}/audit_completed_run.py" "${AUDIT_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] ${RUN_KIND} step ${EXPECTED_STOP} completed and audited"
