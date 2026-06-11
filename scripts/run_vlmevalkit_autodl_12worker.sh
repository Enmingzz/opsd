#!/usr/bin/env bash
set -euo pipefail

# AutoDL VLMEvalKit launcher: 12 inference workers over 4 GPUs by default.
# VLMEvalKit/run.py maps LOCAL_RANK % 4, so each GPU hosts 3 workers.

BASE_ROOT="${BASE_ROOT:-/root/autodl-tmp/opsd_eval}"
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-${BASE_ROOT}/VLMEvalKit}"
OPSD_ROOT="${OPSD_ROOT:-${BASE_ROOT}/opsd}"
OUT_ROOT="${OUT_ROOT:-${BASE_ROOT}/outputs/visionzip_aokvqa_reasoning}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin}"

MODELS="${MODELS:-opsd_qwen25vl_official_7b_flashattn2_mnt32}"
DATASETS="${DATASETS:-POPE}"
RUN_NAME="${RUN_NAME:-vlmevalkit_12worker_$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="${WORK_DIR:-${OUT_ROOT}/eval_vlmevalkit/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs/full/eval_${RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-${OUT_ROOT}/reports/${RUN_NAME}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-12}"
MASTER_PORT="${MASTER_PORT:-29500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
JUDGE="${JUDGE:-exact_matching}"
REUSE_FLAG="${REUSE_FLAG:---reuse}"

if [[ -d "${CONDA_BIN}" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi
export HF_HOME="${HF_HOME:-${BASE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export LMUData="${LMUData:-${BASE_ROOT}/vlmeval_data}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${VLMEVALKIT_ROOT}:${BASE_ROOT}:${PYTHONPATH:-}"
export VISIONZIP_QWEN25VL_ROOT="${VISIONZIP_QWEN25VL_ROOT:-${BASE_ROOT}/VisionZip/Qwen2_5_VL}"
export OPSD_BASE_ROOT="${OPSD_BASE_ROOT:-${BASE_ROOT}}"
export CUDA_VISIBLE_DEVICES
unset MMEVAL_ROOT || true

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${REPORT_DIR}" "${LMUData}"

cd "${VLMEVALKIT_ROOT}"

read -r -a MODEL_ARGS <<< "${MODELS}"
read -r -a DATA_ARGS <<< "${DATASETS}"

{
  echo "[$(date --iso-8601=seconds)] Starting VLMEvalKit 12-worker eval"
  echo "Models: ${MODELS}"
  echo "Datasets: ${DATASETS}"
  echo "Work dir: ${WORK_DIR}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
  echo "MASTER_PORT: ${MASTER_PORT}"
  echo "Judge: ${JUDGE}"
} | tee "${LOG_DIR}/launch.info"

set -x
torchrun --nproc-per-node="${NPROC_PER_NODE}" --master-port="${MASTER_PORT}" run.py \
  --data "${DATA_ARGS[@]}" \
  --model "${MODEL_ARGS[@]}" \
  --work-dir "${WORK_DIR}" \
  --judge "${JUDGE}" \
  ${REUSE_FLAG} \
  2>&1 | tee "${LOG_DIR}/vlmevalkit.log"
set +x

find "${WORK_DIR}" -type f | sort > "${REPORT_DIR}/result_files.txt"
find "${WORK_DIR}" -name status.json -type f | sort > "${REPORT_DIR}/status_files.txt"

python - <<PY
from pathlib import Path
work_dir = Path(${WORK_DIR@Q})
report_dir = Path(${REPORT_DIR@Q})
models = ${MODELS@Q}
datasets = ${DATASETS@Q}
nproc = ${NPROC_PER_NODE@Q}
cuda_visible = ${CUDA_VISIBLE_DEVICES@Q}
lines = [
    '# VLMEvalKit 12-Worker Evaluation',
    '',
    f'- Work dir: `{work_dir}`',
    f'- Models: `{models}`',
    f'- Datasets: `{datasets}`',
    f'- NPROC_PER_NODE: `{nproc}`',
    f'- CUDA_VISIBLE_DEVICES: `{cuda_visible}`',
    '',
    '| Score file |',
    '|---|',
]
for path in sorted(work_dir.rglob('*_score.csv')):
    lines.append(f'| `{path}` |')
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
PY

echo "[$(date --iso-8601=seconds)] Finished VLMEvalKit 12-worker eval"
