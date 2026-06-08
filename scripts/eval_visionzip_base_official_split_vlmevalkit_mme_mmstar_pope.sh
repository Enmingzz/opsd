#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/project/6101803/enmingzz"
REPO_ROOT="${PROJECT_ROOT}/opsd"
OUT_ROOT="/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning"
VLMEVALKIT_ROOT="${PROJECT_ROOT}/vlm/official_thinking_in_space/third_party/VLMEvalKit"
OPENCV_ROOT="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/gcc12/cuda12.2/opencv/4.11.0"

EVAL_NAME="visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030"
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

MODELS=(
  "opsd_qwen25vl_visionzip_base_r005"
  "opsd_qwen25vl_visionzip_base_r010"
  "opsd_qwen25vl_visionzip_base_r020"
  "opsd_qwen25vl_visionzip_base_r030"
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_IDS=(0 1 2 3)
fi

echo "[$(date --iso-8601=seconds)] Starting official-split VisionZip VLMEvalKit evaluation"
echo "Work dir: ${WORK_DIR}"

pids=()
for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPU_IDS[$idx]:-$idx}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "[$(date --iso-8601=seconds)] ${model} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    python run.py \
      --data MME MMStar POPE \
      --model "${model}" \
      --work-dir "${WORK_DIR}" \
      --reuse
  ) > "${LOG_DIR}/${model}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

find "${WORK_DIR}" -type f | sort > "${REPORT_DIR}/result_files.txt"
find "${WORK_DIR}" -name status.json -type f | sort > "${REPORT_DIR}/status_files.txt"

python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

work_dir = Path("/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/eval_vlmevalkit/visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030")
report_dir = Path("/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/reports/visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030")
project_report = Path("/project/6101803/enmingzz/opsd/report/visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030.md")

rows = []
for status_file in sorted(work_dir.glob("opsd_qwen25vl_visionzip_base_r*/T*/status.json")):
    model = status_file.parent.parent.name
    ratio = model.rsplit("_r", 1)[-1]
    status = json.loads(status_file.read_text())
    datasets = status.get("datasets", {})
    mme = datasets.get("MME", {}).get("metrics", {})
    mmstar = datasets.get("MMStar", {}).get("metrics", {})
    pope = datasets.get("POPE", {}).get("metrics", {})
    mme_total = None
    if "perception" in mme and "reasoning" in mme:
        mme_total = float(mme["perception"]) + float(mme["reasoning"])
    rows.append({
        "ratio": ratio,
        "mme_total": mme_total,
        "mme_perception": mme.get("perception"),
        "mme_reasoning": mme.get("reasoning"),
        "mmstar": mmstar.get("split=none|Overall"),
        "pope": pope.get("split=Overall|Overall"),
        "status": {name: datasets.get(name, {}).get("status", "missing") for name in ["MME", "MMStar", "POPE"]},
        "status_file": status_file,
    })

lines = [
    "# VisionZip Base Official-Split VLMEvalKit Evaluation",
    "",
    "- Model: `Qwen2.5-VL-7B-Instruct`, no LoRA adapter",
    "- VisionZip split: contextual = `int(0.05 * original_visual_tokens)`, dominant = retention budget minus contextual",
    "- Datasets: `MME`, `MMStar`, `POPE`",
    "- Mode: direct answer",
    "- Max generation: MME/POPE 16, MMStar 32",
    f"- Work dir: `{work_dir}`",
    "",
    "| Ratio | Status | MME total | MME perception | MME reasoning | MMStar (%) | POPE |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    status = ", ".join(f"{k}:{v}" for k, v in row["status"].items())
    mmstar = "" if row["mmstar"] is None else f"{float(row['mmstar']) * 100:.2f}"
    lines.append(
        "| {ratio} | {status} | {mme_total} | {mme_perception} | {mme_reasoning} | {mmstar} | {pope} |".format(
            ratio=row["ratio"],
            status=status,
            mme_total="" if row["mme_total"] is None else f"{row['mme_total']:.2f}",
            mme_perception="" if row["mme_perception"] is None else f"{float(row['mme_perception']):.2f}",
            mme_reasoning="" if row["mme_reasoning"] is None else f"{float(row['mme_reasoning']):.2f}",
            mmstar=mmstar,
            pope="" if row["pope"] is None else f"{float(row['pope']):.2f}",
        )
    )

lines.extend([
    "",
    "## Status Files",
    "",
])
for row in rows:
    lines.append(f"- `{row['status_file']}`")

text = "\n".join(lines) + "\n"
report_dir.mkdir(parents=True, exist_ok=True)
project_report.parent.mkdir(parents=True, exist_ok=True)
(report_dir / "summary.md").write_text(text)
project_report.write_text(text)
print(f"Wrote {project_report}")
PY

echo "[$(date --iso-8601=seconds)] Finished official-split VisionZip VLMEvalKit evaluation"
exit "${failed}"
