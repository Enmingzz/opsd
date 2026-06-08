#!/bin/bash
# Evaluate OPSD multi-ratio step_1000 on MME at 5/15/25/35/45% retention.
set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false

OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_1000
ADAPTER=/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_3000/divprune_kl_multiratio_010_020_030_040/step_1000
EVAL_JSONL=/scratch/enmingzz/temp/opsd_data/mme_test.jsonl
IMAGE_ROOT=/scratch/enmingzz/temp/opsd_data/mme_images
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/scratch/enmingzz/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}

if [[ ! -d "${ADAPTER}" ]]; then
  echo "Missing adapter checkpoint: ${ADAPTER}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"

run_ratio() {
  local ratio="$1"
  local label="$2"
  local out_dir="${OUT_ROOT}/mme_step1000_${label}"
  mkdir -p "${out_dir}/shards"
  echo "Starting MME eval step1000 ratio=${ratio} label=${label} at $(date)"

  run_shard() {
    local shard="$1"
    local gpu="$2"
    local out="${out_dir}/shards/eval_step1000_${label}_mme_shard${shard}.jsonl"
    local log="${out_dir}/shards/eval_step1000_${label}_mme_shard${shard}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" python opsd/scripts/eval_qwen25vl_pruned_student.py \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --adapter_path "${ADAPTER}" \
      --eval_jsonl "${EVAL_JSONL}" \
      --image_root "${IMAGE_ROOT}" \
      --output_jsonl "${out}" \
      --keep_ratio "${ratio}" \
      --pruner divprune_lite \
      --student_input_mode drop_tokens \
      --max_new_tokens 16 \
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
    "${out_dir}/shards/eval_step1000_${label}_mme_shard0.jsonl" \
    "${out_dir}/shards/eval_step1000_${label}_mme_shard1.jsonl" \
    "${out_dir}/shards/eval_step1000_${label}_mme_shard2.jsonl" \
    "${out_dir}/shards/eval_step1000_${label}_mme_shard3.jsonl" \
    > "${out_dir}/eval_step1000_${label}_mme_full.jsonl"

  python opsd/scripts/score_mme_eval.py \
    --eval_jsonl "${out_dir}/eval_step1000_${label}_mme_full.jsonl" \
    --output_dir "${out_dir}/score_step1000_${label}_full"

  echo "Finished MME eval step1000 ratio=${ratio} label=${label} at $(date)"
}

run_ratio 0.05 r005
run_ratio 0.15 r015
run_ratio 0.25 r025
run_ratio 0.35 r035
run_ratio 0.45 r045

python - <<'PY'
import csv
import json
from pathlib import Path

out_root = Path("/scratch/enmingzz/temp/opsd_runs/multiratio_010_020_030_040_1000")
rows = []
for label, ratio in [("r005", 0.05), ("r015", 0.15), ("r025", 0.25), ("r035", 0.35), ("r045", 0.45)]:
    summary_path = out_root / f"mme_step1000_{label}" / f"score_step1000_{label}_full" / "mme_summary.json"
    data = json.loads(summary_path.read_text())
    for mode, vals in data.items():
        vals = dict(vals)
        vals["ratio"] = ratio
        vals["label"] = label
        vals["mode"] = mode
        rows.append(vals)
fields = [
    "label",
    "ratio",
    "mode",
    "num_rows",
    "yes_no_accuracy",
    "mme_total_score",
    "mme_perception_score",
    "mme_cognition_score",
    "avg_full_visual_tokens",
    "avg_kept_visual_tokens",
    "avg_latency_seconds",
    "avg_generated_tokens",
    "other_predictions",
]
summary_csv = out_root / "mme_step1000_retention_sweep.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})
report = out_root / "mme_step1000_retention_sweep.md"
lines = ["# OPSD Multi-Ratio Step 1000 MME Retention Sweep", ""]
lines.append("| Ratio | Mode | Acc | MME Total | Perception | Cognition | Avg Kept Tokens |")
lines.append("|---:|---|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['ratio']:.2f} | {row['mode']} | {row['yes_no_accuracy']:.4f} | "
        f"{row['mme_total_score']:.2f} | {row['mme_perception_score']:.2f} | "
        f"{row['mme_cognition_score']:.2f} | {row['avg_kept_visual_tokens']:.2f} |"
    )
report.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"summary_csv": str(summary_csv), "report": str(report)}, indent=2))
PY

echo "MME step1000 retention sweep complete at $(date)"
