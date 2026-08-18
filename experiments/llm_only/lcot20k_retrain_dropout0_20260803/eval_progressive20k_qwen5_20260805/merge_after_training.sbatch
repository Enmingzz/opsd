#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=pg20-merge
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/eval/progressive_opsd20k_dropout0_step20000_merged_qwen5_20260805/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/eval/progressive_opsd20k_dropout0_step20000_merged_qwen5_20260805/logs/%x-%j.err

set -euo pipefail

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
RUN_ROOT=/scratch/enmingzz/outputs/llm_only/lcot20k_retrain_dropout0_20260803/progressive_opsd
ADAPTER_PATH="${RUN_ROOT}/resume_checkpoints/step_020000"
MERGED_MODEL=/scratch/enmingzz/outputs/llm_only/merged_models/lcot20k_retrain_dropout0_20260803/progressive_opsd_step20000_qwen25vl7b_merged_bf16
TMP_MODEL="${MERGED_MODEL}.tmp-${SLURM_JOB_ID:-manual}"

cd "${OPSD_ROOT}"
source "${PROJECT_ROOT}/ckpt_eval_trainenv/env_train_runtime.sh"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/scratch/enmingzz/cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

python - "${RUN_ROOT}" "${ADAPTER_PATH}" <<'PY'
import json
import sys
from pathlib import Path

run_root, adapter = map(lambda value: Path(value).resolve(), sys.argv[1:])
audit = json.loads(run_root.joinpath("post_training_audit_step_020000.json").read_text())
scope = json.loads(run_root.joinpath("scope_verification_step_020000.json").read_text())
adapter_config = json.loads(adapter.joinpath("adapter_config.json").read_text())
assert audit["status"] == "passed" and audit["expected_stop"] == 20000
assert audit["all_losses_finite"] is True and audit["optimizer_steps"] == 625
assert audit["lora_dropout"] == 0.0 and audit["ema_tensors"] == 392
assert audit["kind"] == "progressive" and audit["final"] is True
assert audit["effective_batch_size"] == 32
assert audit["ratio_counts"] == {"0.1": 5024, "0.2": 4992, "0.3": 4992, "0.4": 4992}
assert audit["sample_order_exact"] is True and audit["ratio_assignments_exact"] is True
assert scope["scope_verified"] is True and scope["expected_scope"] == "language_decoder_only"
assert scope["visual_name_count"] == 0 and scope["adapter_tensor_count"] == 392
assert Path(scope["adapter_path"]).resolve() == adapter
assert float(adapter_config["lora_dropout"]) == 0.0
assert adapter.joinpath("adapter_model.safetensors").is_file()
print(json.dumps({"status": "source_validated", "adapter": str(adapter)}, indent=2))
PY

if [[ -d "${MERGED_MODEL}" && -f "${MERGED_MODEL}/merge_metadata.json" ]]; then
  python - "${ADAPTER_PATH}" "${MERGED_MODEL}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

adapter, merged = map(lambda value: Path(value).resolve(), sys.argv[1:])
digest = hashlib.sha256(adapter.joinpath("adapter_model.safetensors").read_bytes()).hexdigest()
metadata = json.loads(merged.joinpath("merge_metadata.json").read_text())
assert Path(metadata["adapter_path"]).resolve() == adapter
assert metadata["adapter_model_sha256"] == digest
assert metadata["source_final_global_step"] == 20000
assert metadata["parameter_scope"] == "language_decoder_only"
assert metadata["lora_dropout"] == 0.0
assert metadata["training_method"] == "progressive_opsd_ema"
assert merged.joinpath("model.safetensors.index.json").is_file()
assert len(list(merged.glob("model-*.safetensors"))) == 4
print(json.dumps({"status": "matching_merge_exists", "merged": str(merged)}, indent=2))
PY
  exit 0
fi

if [[ -e "${TMP_MODEL}" ]]; then
  echo "Temporary output already exists: ${TMP_MODEL}" >&2
  exit 1
fi
mkdir -p "$(dirname "${MERGED_MODEL}")"
python scripts/merge_qwen25vl_lora.py \
  --base_model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter_path "${ADAPTER_PATH}" \
  --output_dir "${TMP_MODEL}" \
  --attn_implementation flash_attention_2 \
  --device_map auto \
  --bf16 \
  --note "LLM-only Progressive OPSD with EMA teacher, dropout=0, ordered decontaminated LLaVA-CoT 20K, step 20000."

python - "${ADAPTER_PATH}" "${TMP_MODEL}" "${MERGED_MODEL}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

adapter, temporary, final = map(lambda value: Path(value).resolve(), sys.argv[1:])
digest = hashlib.sha256(adapter.joinpath("adapter_model.safetensors").read_bytes()).hexdigest()
metadata_path = temporary / "merge_metadata.json"
metadata = json.loads(metadata_path.read_text())
assert Path(metadata["adapter_path"]).resolve() == adapter
assert temporary.joinpath("model.safetensors.index.json").is_file()
metadata.update({
    "output_dir": str(final),
    "adapter_model_sha256": digest,
    "source_training_job_id": "4599261",
    "source_final_global_step": 20000,
    "parameter_scope": "language_decoder_only",
    "lora_dropout": 0.0,
    "training_method": "progressive_opsd_ema",
})
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
PY
mv "${TMP_MODEL}" "${MERGED_MODEL}"
echo "merged_model=${MERGED_MODEL}"
