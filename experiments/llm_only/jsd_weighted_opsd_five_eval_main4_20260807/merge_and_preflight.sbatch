#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --exclude=kn136
#SBATCH --job-name=j5-merge
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/eval/jsd_weighted_opsd_five_step10240_merged_20260807/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/eval/jsd_weighted_opsd_five_step10240_merged_20260807/logs/%x-%j.err

set -euo pipefail
[[ "$#" -eq 1 ]] || { echo "Usage: $0 METHOD" >&2; exit 2; }

source /project/6101803/enmingzz/opsd/experiments/llm_only/jsd_weighted_opsd_five_eval_main4_20260807/common.sh
configure_method "$1"
TMP_MERGED_MODEL="${MERGED_MODEL}.tmp-${SLURM_JOB_ID:-manual}"

mkdir -p "${OUT_ROOT}/logs" "${MERGED_ROOT}"
cd "${OPSD_ROOT}"
source "${PROJECT_ROOT}/ckpt_eval_trainenv/env_train_runtime.sh"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/scratch/enmingzz/cache/huggingface}"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1

echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-manual} method=${METHOD}"
echo "git=$(git rev-parse HEAD) adapter=${ADAPTER_PATH} merged=${MERGED_MODEL}"
nvidia-smi

python - "${ADAPTER_PATH}" "${RUN_ROOT}" "${TRAIN_CONFIG}" "${EXPECTED_MODE}" "${EXPECTED_DELTA}" <<'PY'
import json
import math
import sys
from pathlib import Path

import yaml
from safetensors import safe_open

adapter, run_root, config_path = map(lambda x: Path(x).resolve(), sys.argv[1:4])
expected_mode = sys.argv[4]
expected_delta = float(sys.argv[5])
assert adapter.joinpath("COMPLETE").is_file(), adapter
assert adapter.joinpath("adapter_config.json").is_file(), adapter
assert adapter.joinpath("adapter_model.safetensors").is_file(), adapter
cfg = yaml.safe_load(config_path.read_text())
assert int(cfg["training"]["max_steps"]) == 10240
assert cfg["experiment"]["parameter_scope"] == "language_decoder_only"
assert float(cfg["training"]["lora_dropout"]) == 0.0
assert cfg["training"]["target_modules"] == [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]
assert cfg["opsd"]["trajectory_weighting"]["mode"] == expected_mode
assert cfg["opsd"]["trajectory_weighting"]["normalization_scope"] == "effective_batch"
assert float(cfg["opsd"]["native_budget_weighting"]["budget_delta_fraction"]) == expected_delta
adapter_cfg = json.loads(adapter.joinpath("adapter_config.json").read_text())
assert adapter_cfg["base_model_name_or_path"] == "Qwen/Qwen2.5-VL-7B-Instruct"
assert float(adapter_cfg["lora_dropout"]) == 0.0
with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
assert len(keys) == 392, len(keys)
assert not any(".visual." in key or "merger" in key for key in keys)

ratio_counts = {0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0}
for rank in range(4):
    path = run_root / f"rank{rank}_native_budget_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2560, (rank, len(rows))
    assert int(rows[-1]["step"]) == 10240, (rank, rows[-1]["step"])
    for row in rows:
        assert math.isfinite(float(row["loss"]))
        assert row["trajectory_weighting_mode"] == expected_mode
        assert row["trajectory_weight_detached"] is True
        assert abs(float(row["trajectory_probability_weight_sum"]) - 1.0) < 1e-5
        assert row["effective_batch_fixed_prefix_replay"] is True
        ratio = round(float(row["sampled_b"]), 1)
        assert ratio in ratio_counts
        assert abs(float(row["sampled_b_plus"]) - ratio * (1.0 + expected_delta)) < 1e-6
        ratio_counts[ratio] += 1
assert sum(ratio_counts.values()) == 10240
assert all(2300 <= count <= 2800 for count in ratio_counts.values()), ratio_counts
print(json.dumps({
    "status": "source_adapter_validated",
    "adapter_tensors": len(keys),
    "trajectory_weighting_mode": expected_mode,
    "budget_delta_fraction": expected_delta,
    "ratio_counts": {str(key): value for key, value in ratio_counts.items()},
}, indent=2))
PY

already_merged="$({ python - "${ADAPTER_PATH}" "${MERGED_MODEL}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

adapter, merged = map(lambda x: Path(x).resolve(), sys.argv[1:])
if not merged.exists():
    print("0")
    raise SystemExit
metadata = json.loads(merged.joinpath("merge_metadata.json").read_text())
assert Path(metadata["adapter_path"]).resolve() == adapter
assert Path(metadata["output_dir"]).resolve() == merged
assert metadata["adapter_model_sha256"] == sha256(adapter / "adapter_model.safetensors")
assert int(metadata["source_final_global_step"]) == 10240
assert merged.joinpath("model.safetensors.index.json").is_file()
assert len(list(merged.glob("model-*.safetensors"))) == 4
print("1")
PY
} )"

if [[ "${already_merged}" == "0" ]]; then
  [[ ! -e "${TMP_MERGED_MODEL}" ]] || { echo "Temporary merge path exists" >&2; exit 1; }
  python scripts/merge_qwen25vl_lora.py \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ADAPTER_PATH}" \
    --output_dir "${TMP_MERGED_MODEL}" \
    --attn_implementation flash_attention_2 \
    --device_map auto \
    --bf16 \
    --note "${METHOD}, step 10240, LLM-only dropout=0; merged BF16 for clean-Armen reasoning evaluation."

  python - "${ADAPTER_PATH}" "${TMP_MERGED_MODEL}" "${MERGED_MODEL}" "${TRAIN_JOB_ID}" "${TRAIN_CONFIG}" "${METHOD}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

adapter, temporary, final = map(lambda x: Path(x).resolve(), sys.argv[1:4])
job_id, config, method = sys.argv[4:]
metadata_path = temporary / "merge_metadata.json"
metadata = json.loads(metadata_path.read_text())
assert Path(metadata["adapter_path"]).resolve() == adapter
assert temporary.joinpath("model.safetensors.index.json").is_file()
assert len(list(temporary.glob("model-*.safetensors"))) == 4
metadata.update({
    "output_dir": str(final),
    "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
    "source_training_job_id": job_id,
    "source_training_config": str(Path(config).resolve()),
    "source_final_global_step": 10240,
    "training_method": method,
    "parameter_scope": "language_decoder_only",
    "lora_dropout": 0.0,
})
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
PY
  mv "${TMP_MERGED_MODEL}" "${MERGED_MODEL}"
elif [[ "${already_merged}" == "1" ]]; then
  echo "Matching merged model already exists."
else
  echo "Unexpected merged state: ${already_merged}" >&2
  exit 1
fi

for task in mme mmstar mathvista mmmupro; do
  for ratio in noprune r010 r020 r030; do
    CASE_PREFLIGHT_ONLY=1 bash "${CAMPAIGN_ROOT}/eval_one_case.sbatch" "${METHOD}" "${task}" "${ratio}"
  done
done

python - "${OUT_ROOT}" "${MERGED_MODEL}" "${METHOD}" "${SLURM_JOB_ID:-manual}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path
out_root, merged = map(lambda x: Path(x).resolve(), sys.argv[1:3])
method, job_id = sys.argv[3:]
path = out_root / f"merge_and_eval_preflight_{method}.json"
path.write_text(json.dumps({
    "status": "passed", "method": method, "merged_model": str(merged),
    "merge_job_id": job_id, "evaluation_preflights_passed": 16,
    "validated_at": datetime.now().astimezone().isoformat(),
}, indent=2) + "\n")
PY

echo "completed=$(date --iso-8601=seconds)"
