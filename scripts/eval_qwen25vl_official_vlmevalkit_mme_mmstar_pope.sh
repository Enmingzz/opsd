#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/project/6101803/enmingzz"
REPO_ROOT="${PROJECT_ROOT}/opsd"
OUT_ROOT="/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning"
VLMEVALKIT_ROOT="${PROJECT_ROOT}/vlm/official_thinking_in_space/third_party/VLMEvalKit"
QWEN25_BOOTSTRAP="/scratch/enmingzz/temp/qwen25_bootstrap"
OPENCV_ROOT="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/gcc12/cuda12.2/opencv/4.11.0"

MODEL_NAME="Qwen2.5-VL-7B-Instruct"
EVAL_NAME="qwen25vl_official_7b_qwen25bootstrap_vlmevalkit_mme_mmstar_pope"
WORK_DIR="${OUT_ROOT}/eval_vlmevalkit/${EVAL_NAME}"
LOG_DIR="${OUT_ROOT}/logs/full/eval_${EVAL_NAME}"
REPORT_DIR="${OUT_ROOT}/reports/${EVAL_NAME}"
PROJECT_REPORT="${REPO_ROOT}/report/${EVAL_NAME}.md"

source "${PROJECT_ROOT}/env/vsi-official.sh"

export HF_HOME="/scratch/enmingzz/hf_cache"
export TRANSFORMERS_CACHE="/scratch/enmingzz/hf_cache"
export LMUData="/scratch/enmingzz/vlmevalkit_data"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONNOUSERSITE=1
unset MMEVAL_ROOT

export PYTHONPATH="${QWEN25_BOOTSTRAP}:${VLMEVALKIT_ROOT}:${PROJECT_ROOT}:${PROJECT_ROOT}/vlm/official_thinking_in_space:${OPENCV_ROOT}/lib/python3.11/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${OPENCV_ROOT}/lib64:${OPENCV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${REPORT_DIR}" "${LMUData}" "${REPO_ROOT}/report"

cd "${VLMEVALKIT_ROOT}"

python - <<'PY'
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vlmeval.config import supported_VLM

assert "Qwen2.5-VL-7B-Instruct" in supported_VLM
print(
    "Qwen2.5-VL base import preflight passed:",
    Qwen2_5_VLForConditionalGeneration.__name__,
    AutoProcessor.__name__,
)
PY

echo "[$(date --iso-8601=seconds)] Starting official VLMEvalKit ${MODEL_NAME}"
echo "Work dir: ${WORK_DIR}"
python run.py \
  --data MME MMStar POPE \
  --model "${MODEL_NAME}" \
  --work-dir "${WORK_DIR}" \
  --reuse \
  2>&1 | tee "${LOG_DIR}/${MODEL_NAME}.log"

find "${WORK_DIR}" -type f | sort > "${REPORT_DIR}/result_files.txt"
find "${WORK_DIR}" -name status.json -type f | sort > "${REPORT_DIR}/status_files.txt"

python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

work_dir = Path("/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/eval_vlmevalkit/qwen25vl_official_7b_qwen25bootstrap_vlmevalkit_mme_mmstar_pope")
report_dir = Path("/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/reports/qwen25vl_official_7b_qwen25bootstrap_vlmevalkit_mme_mmstar_pope")
project_report = Path("/project/6101803/enmingzz/opsd/report/qwen25vl_official_7b_qwen25bootstrap_vlmevalkit_mme_mmstar_pope.md")


status_files = sorted(work_dir.glob("Qwen2.5-VL-7B-Instruct/T*/status.json"), key=lambda p: p.stat().st_mtime)
if not status_files:
    raise SystemExit(f"No status.json found under {work_dir}")
status_file = status_files[-1]
status = json.loads(status_file.read_text())
datasets = status.get("datasets", {})

def dataset_metrics(name: str) -> tuple[str, dict, str | None]:
    row = datasets.get(name, {})
    return row.get("status", "missing"), row.get("metrics", {}), row.get("error_message")

mme_status, mme, mme_error = dataset_metrics("MME")
mmstar_status, mmstar, mmstar_error = dataset_metrics("MMStar")
pope_status, pope, pope_error = dataset_metrics("POPE")

mme_total = None
if "perception" in mme and "reasoning" in mme:
    mme_total = float(mme["perception"]) + float(mme["reasoning"])
mmstar_overall = mmstar.get("split=none|Overall")
pope_score = pope.get("split=Overall|Overall")

lines = [
    "# Qwen2.5-VL-7B-Instruct Official VLMEvalKit Evaluation",
    "",
    "- Model: `Qwen2.5-VL-7B-Instruct`",
    "- Wrapper: official VLMEvalKit `Qwen2VLChat` registration",
    "- Datasets: `MME`, `MMStar`, `POPE`",
    "- Mode: direct, no VisionZip pruning, no LoRA adapter",
    "- Transformers: Qwen2.5 bootstrap path `/scratch/enmingzz/temp/qwen25_bootstrap`",
    f"- Work dir: `{work_dir}`",
    f"- Status file: `{status_file}`",
    "",
    "## Scores",
    "",
    "| Benchmark | Status | Main score | Details |",
    "|---|---|---:|---|",
]

lines.append(
    "| MME | {status} | {score} | perception={perception}, reasoning={reasoning} |".format(
        status=mme_status,
        score="" if mme_total is None else f"{mme_total:.2f}",
        perception="" if "perception" not in mme else f"{float(mme['perception']):.2f}",
        reasoning="" if "reasoning" not in mme else f"{float(mme['reasoning']):.2f}",
    )
)
lines.append(
    "| MMStar | {status} | {score} | Overall Acc (%) |".format(
        status=mmstar_status,
        score="" if mmstar_overall is None else f"{float(mmstar_overall) * 100:.2f}",
    )
)
lines.append(
    "| POPE | {status} | {score} | split=Overall|Overall |".format(
        status=pope_status,
        score="" if pope_score is None else f"{float(pope_score):.2f}",
    )
)

lines.extend([
    "",
    "## Raw Result Files",
    "",
])
for path in sorted(work_dir.rglob("*")):
    if path.is_file() and path.name.endswith((".csv", ".xlsx", ".json", ".pkl")):
        lines.append(f"- `{path}`")

text = "\n".join(lines) + "\n"
report_dir.mkdir(parents=True, exist_ok=True)
project_report.parent.mkdir(parents=True, exist_ok=True)
(report_dir / "summary.md").write_text(text)
project_report.write_text(text)
print(f"Wrote {project_report}")

failures = []
for name, state, error in [
    ("MME", mme_status, mme_error),
    ("MMStar", mmstar_status, mmstar_error),
    ("POPE", pope_status, pope_error),
]:
    if state != "done" or error:
        failures.append(f"{name}: status={state}, error={error}")
if mme_total is None:
    failures.append("MME: missing perception/reasoning metrics")
if mmstar_overall is None:
    failures.append("MMStar: missing Overall metric")
if pope_score is None:
    failures.append("POPE: missing Overall metric")
if failures:
    raise SystemExit("Invalid base Qwen2.5-VL evaluation:\n" + "\n".join(failures))
PY

echo "[$(date --iso-8601=seconds)] Finished official VLMEvalKit ${MODEL_NAME}"
