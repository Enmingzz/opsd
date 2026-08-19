#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
source "${PROJECT_ROOT}/ckpt_eval_trainenv/env_train_runtime.sh"

cd "${OPSD_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/scratch/enmingzz/cache/huggingface}"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1

python - "${TRAIN_CONFIG}" "${TRAIN_RUN_ROOT}" "${ADAPTER_PATH}" "${CHECKPOINT_STEP}" <<'PY'
import json
import sys
from pathlib import Path

import yaml
from safetensors import safe_open

config_path, run_root, adapter = map(lambda value: Path(value).resolve(), sys.argv[1:4])
step = int(sys.argv[4])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
audit = json.loads(run_root.joinpath("final_training_audit.json").read_text(encoding="utf-8"))
scope = json.loads(run_root.joinpath("final_scope_verification.json").read_text(encoding="utf-8"))
adapter_config = json.loads(adapter.joinpath("adapter_config.json").read_text(encoding="utf-8"))

assert config["training"]["method"] == "opsd_nogt"
assert config["experiment"]["parameter_scope"] == "language_decoder_only"
assert config["pruning"]["train_retention_ratios"] == [0.10]
assert float(config["training"]["lora_dropout"]) == 0.0
assert int(audit["final_step"]) == step and audit["status"] == "passed"
assert scope["scope_verified"] is True and scope["visual_name_count"] == 0
assert Path(scope["adapter_path"]).resolve() == adapter
assert float(adapter_config["lora_dropout"]) == 0.0
assert adapter.joinpath("COMPLETE").is_file()
with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
    names = list(handle.keys())
assert len(names) == 392
assert not any(".visual." in name or "merger" in name for name in names)
PY

if [[ -f "${MERGED_MODEL}/merge_metadata.json" ]]; then
  python - "${ADAPTER_PATH}" "${MERGED_MODEL}" "${CHECKPOINT_STEP}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

adapter, merged = map(lambda value: Path(value).resolve(), sys.argv[1:3])
step = int(sys.argv[3])
metadata = json.loads(merged.joinpath("merge_metadata.json").read_text(encoding="utf-8"))
digest = hashlib.sha256(adapter.joinpath("adapter_model.safetensors").read_bytes()).hexdigest()
assert Path(metadata["adapter_path"]).resolve() == adapter
assert metadata["adapter_model_sha256"] == digest
assert int(metadata["source_final_global_step"]) == step
assert metadata["train_retention_ratios"] == [0.1]
assert merged.joinpath("model.safetensors.index.json").is_file()
PY
  echo "Matching merged model already exists: ${MERGED_MODEL}"
  exit 0
fi

TMP_MODEL="${MERGED_MODEL}.tmp-${SLURM_JOB_ID:-manual}"
[[ ! -e "${TMP_MODEL}" ]] || { echo "Temporary merge path exists: ${TMP_MODEL}" >&2; exit 1; }
mkdir -p "$(dirname "${MERGED_MODEL}")"
python scripts/merge_qwen25vl_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --output_dir "${TMP_MODEL}" \
  --attn_implementation flash_attention_2 \
  --device_map auto \
  --bf16 \
  --note "r010-only OPSD step ${CHECKPOINT_STEP}; LLM-only LoRA dropout 0."

python - "${ADAPTER_PATH}" "${TMP_MODEL}" "${MERGED_MODEL}" "${CHECKPOINT_STEP}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

adapter, temporary, final = map(lambda value: Path(value).resolve(), sys.argv[1:4])
step = int(sys.argv[4])
metadata_path = temporary / "merge_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata.update({
    "output_dir": str(final),
    "adapter_model_sha256": hashlib.sha256(
        adapter.joinpath("adapter_model.safetensors").read_bytes()
    ).hexdigest(),
    "source_final_global_step": step,
    "training_method": "original_opsd_forward_kl_r010_only",
    "training_objective": "KL(q_full || p_r010)",
    "train_retention_ratios": [0.1],
    "parameter_scope": "language_decoder_only",
    "lora_dropout": 0.0,
})
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY
mv "${TMP_MODEL}" "${MERGED_MODEL}"
echo "merged_model=${MERGED_MODEL}"
