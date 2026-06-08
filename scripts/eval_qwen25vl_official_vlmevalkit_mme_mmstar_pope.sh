#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/project/6101803/enmingzz"
REPO_ROOT="${PROJECT_ROOT}/opsd"
OUT_ROOT="${PROJECT_ROOT}/outputs/visionzip_aokvqa_reasoning"
VLMEVALKIT_ROOT="${PROJECT_ROOT}/vlm/official_thinking_in_space/third_party/VLMEvalKit"
OPENCV_ROOT="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/gcc12/cuda12.2/opencv/4.11.0"

MODEL_NAME="Qwen2.5-VL-7B-Instruct"
EVAL_NAME="qwen25vl_official_7b_vlmevalkit_mme_mmstar_pope"
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

export PYTHONPATH="${VLMEVALKIT_ROOT}:${PROJECT_ROOT}:${PROJECT_ROOT}/vlm/official_thinking_in_space:${OPENCV_ROOT}/lib/python3.11/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${OPENCV_ROOT}/lib64:${OPENCV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${REPORT_DIR}" "${LMUData}" "${REPO_ROOT}/report"

cd "${VLMEVALKIT_ROOT}"

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

import csv
from pathlib import Path

work_dir = Path("/project/6101803/enmingzz/outputs/visionzip_aokvqa_reasoning/eval_vlmevalkit/qwen25vl_official_7b_vlmevalkit_mme_mmstar_pope")
report_dir = Path("/project/6101803/enmingzz/outputs/visionzip_aokvqa_reasoning/reports/qwen25vl_official_7b_vlmevalkit_mme_mmstar_pope")
project_report = Path("/project/6101803/enmingzz/opsd/report/qwen25vl_official_7b_vlmevalkit_mme_mmstar_pope.md")


def latest(pattern: str) -> Path | None:
    files = sorted(work_dir.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def read_single_row_csv(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


mme = read_single_row_csv(latest("*_MME_score.csv"))
mmstar = read_single_row_csv(latest("*_MMStar_acc.csv"))
pope = read_single_row_csv(latest("*_POPE_score.csv"))

lines = [
    "# Qwen2.5-VL-7B-Instruct Official VLMEvalKit Evaluation",
    "",
    "- Model: `Qwen2.5-VL-7B-Instruct`",
    "- Wrapper: official VLMEvalKit `Qwen2VLChat` registration",
    "- Datasets: `MME`, `MMStar`, `POPE`",
    "- Mode: direct, no VisionZip pruning, no LoRA adapter",
    f"- Work dir: `{work_dir}`",
    "",
    "## Scores",
    "",
    "| Benchmark | Main score | Source file |",
    "|---|---:|---|",
]

mme_total = mme.get("total") or mme.get("Overall") or mme.get("score") or ""
mmstar_overall = mmstar.get("Overall") or ""
pope_score = pope.get("Overall") or pope.get("overall") or pope.get("accuracy") or pope.get("acc") or ""

files = {
    "MME": latest("*_MME_score.csv"),
    "MMStar": latest("*_MMStar_acc.csv"),
    "POPE": latest("*_POPE_score.csv"),
}
values = {"MME": mme_total, "MMStar": mmstar_overall, "POPE": pope_score}
for name in ["MME", "MMStar", "POPE"]:
    value = values[name] or "pending"
    source = str(files[name]) if files[name] else "missing"
    lines.append(f"| {name} | {value} | `{source}` |")

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
PY

echo "[$(date --iso-8601=seconds)] Finished official VLMEvalKit ${MODEL_NAME}"
