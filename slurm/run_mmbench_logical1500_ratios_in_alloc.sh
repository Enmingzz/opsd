#!/bin/bash
# Evaluate logical step_1500 OPSD on MMBench EN dev at 5/15/25/35/45% retention.
set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false

OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_1500/mmbench_en_dev
ADAPTER=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_1500/divprune_kl_multiratio_010_020_030_040_resume_from_step1000_to_step1500/step_500
EVAL_JSONL=/scratch/enmingzz/temp/opsd_data/mmbench_en_dev.jsonl
IMAGE_ROOT=/scratch/enmingzz/temp/opsd_data/mmbench_en_dev_images

if [[ ! -d "${ADAPTER}" ]]; then
  echo "Missing adapter: ${ADAPTER}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_JSONL}" ]]; then
  echo "Missing MMBench jsonl: ${EVAL_JSONL}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"

run_ratio() {
  local ratio="$1"
  local label="$2"
  local out_dir="${OUT_ROOT}/r${label}"
  mkdir -p "${out_dir}/shards"
  echo "Starting MMBench EN dev eval ratio=${ratio} label=r${label} at $(date)"

  run_shard() {
    local shard="$1"
    local gpu="$2"
    local out="${out_dir}/shards/eval_mmbench_r${label}_shard${shard}.jsonl"
    local log="${out_dir}/shards/eval_mmbench_r${label}_shard${shard}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" python opsd/scripts/eval_qwen25vl_pruned_student.py \
      --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
      --adapter_path "${ADAPTER}" \
      --eval_jsonl "${EVAL_JSONL}" \
      --image_root "${IMAGE_ROOT}" \
      --output_jsonl "${out}" \
      --keep_ratio "${ratio}" \
      --pruner divprune_lite \
      --student_input_mode drop_tokens \
      --max_new_tokens 8 \
      --num_shards 4 \
      --shard_index "${shard}" \
      --skip_full_token_base true \
      --attn_implementation flash_attention_2 \
      > "${log}" 2>&1
  }

  run_shard 0 0 &
  pid0=$!
  run_shard 1 1 &
  pid1=$!
  run_shard 2 2 &
  pid2=$!
  run_shard 3 3 &
  pid3=$!
  wait "${pid0}"
  wait "${pid1}"
  wait "${pid2}"
  wait "${pid3}"

  cat \
    "${out_dir}/shards/eval_mmbench_r${label}_shard0.jsonl" \
    "${out_dir}/shards/eval_mmbench_r${label}_shard1.jsonl" \
    "${out_dir}/shards/eval_mmbench_r${label}_shard2.jsonl" \
    "${out_dir}/shards/eval_mmbench_r${label}_shard3.jsonl" \
    > "${out_dir}/eval_mmbench_r${label}_full.jsonl"

  python opsd/scripts/score_mmbench_eval.py \
    --eval_jsonl "${out_dir}/eval_mmbench_r${label}_full.jsonl" \
    --output_dir "${out_dir}/score"

  echo "Finished MMBench EN dev eval ratio=${ratio} label=r${label} at $(date)"
}

run_ratio 0.05 005
run_ratio 0.15 015
run_ratio 0.25 025
run_ratio 0.35 035
run_ratio 0.45 045

python - <<'PY'
import csv
import json
from pathlib import Path

out_root = Path("/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_1500/mmbench_en_dev")
rows = []
for label, ratio in [("005", 0.05), ("015", 0.15), ("025", 0.25), ("035", 0.35), ("045", 0.45)]:
    summary_path = out_root / f"r{label}" / "score" / "mmbench_summary.json"
    data = json.loads(summary_path.read_text())
    for mode, vals in data["overall"].items():
        vals = dict(vals)
        vals["ratio"] = ratio
        vals["label"] = f"r{label}"
        vals["mode"] = mode
        rows.append(vals)

fields = [
    "label",
    "ratio",
    "mode",
    "num_rows",
    "accuracy",
    "invalid_predictions",
    "invalid_rate",
    "avg_full_visual_tokens",
    "avg_kept_visual_tokens",
    "avg_latency_seconds",
    "avg_generated_tokens",
]
summary_csv = out_root / "mmbench_logical_step1500_retention_sweep.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})

report = out_root / "mmbench_logical_step1500_retention_sweep.md"
lines = ["# MMBench EN Dev Logical Step 1500 Retention Sweep", ""]
lines.append("Simple option-letter accuracy, not GPT-based official MMBench matching.")
lines.append("")
lines.append("| Ratio | Mode | Rows | Acc | Invalid | Avg Kept Tokens |")
lines.append("|---:|---|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['ratio']:.2f} | {row['mode']} | {row['num_rows']} | {row['accuracy']:.4f} | "
        f"{row['invalid_predictions']} | {row['avg_kept_visual_tokens']:.2f} |"
    )
report.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"summary_csv": str(summary_csv), "report": str(report)}, indent=2))
PY

echo "MMBench EN dev retention sweep complete at $(date)"
