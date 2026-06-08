#!/bin/bash
# Recover the failed MME step_1000 r045 shard, rescore MME, then run MMBench.
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

label=r045
ratio=0.45
out_dir="${OUT_ROOT}/mme_step1000_${label}"
mkdir -p "${out_dir}/shards"

recover_out="${out_dir}/shards/eval_step1000_${label}_mme_shard3.jsonl"
recover_log="${out_dir}/shards/eval_step1000_${label}_mme_shard3_recover.log"

echo "Recovering MME step1000 ${label} shard3 at $(date)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" python opsd/scripts/eval_qwen25vl_pruned_student.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --adapter_path "${ADAPTER}" \
  --eval_jsonl "${EVAL_JSONL}" \
  --image_root "${IMAGE_ROOT}" \
  --output_jsonl "${recover_out}" \
  --keep_ratio "${ratio}" \
  --pruner divprune_lite \
  --student_input_mode drop_tokens \
  --max_new_tokens 16 \
  --num_shards 4 \
  --shard_index 3 \
  --skip_full_token_base true \
  --attn_implementation flash_attention_2 \
  > "${recover_log}" 2>&1

for shard in 0 1 2 3; do
  file="${out_dir}/shards/eval_step1000_${label}_mme_shard${shard}.jsonl"
  if [[ ! -s "${file}" ]]; then
    echo "Missing recovered MME shard: ${file}" >&2
    exit 3
  fi
  wc -l "${file}"
done

cat \
  "${out_dir}/shards/eval_step1000_${label}_mme_shard0.jsonl" \
  "${out_dir}/shards/eval_step1000_${label}_mme_shard1.jsonl" \
  "${out_dir}/shards/eval_step1000_${label}_mme_shard2.jsonl" \
  "${out_dir}/shards/eval_step1000_${label}_mme_shard3.jsonl" \
  > "${out_dir}/eval_step1000_${label}_mme_full.jsonl"

python opsd/scripts/score_mme_eval.py \
  --eval_jsonl "${out_dir}/eval_step1000_${label}_mme_full.jsonl" \
  --output_dir "${out_dir}/score_step1000_${label}_full"

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

echo "Recovered MME step1000 r045 and wrote retention summary at $(date)"
echo "Starting MMBench step1000 retention sweep at $(date)"
bash opsd/slurm/run_mmbench_step1000_ratios_in_alloc.sh
echo "Recovered MME and completed MMBench step1000 retention sweep at $(date)"
