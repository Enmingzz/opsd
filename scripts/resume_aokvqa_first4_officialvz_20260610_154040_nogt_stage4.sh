#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/opsd_eval/opsd
export PATH=/root/miniconda3/bin:${PATH}
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/root/autodl-tmp/opsd_eval:/root/autodl-tmp/opsd_eval/VLMEvalKit:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OPSD_DDP_TIMEOUT_MINUTES=${OPSD_DDP_TIMEOUT_MINUTES:-120}
export OPSD_DDP_STAGGER_LOAD_SECONDS=${OPSD_DDP_STAGGER_LOAD_SECONDS:-15}
RUN_GROUP=aokvqa_first4_officialvz_20260610_154040
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29761}
BASE_ROOT=/root/autodl-tmp/opsd_eval/outputs/visionzip_aokvqa_reasoning
BASE_OUT=${BASE_ROOT}/checkpoints/${RUN_GROUP}
LOGDIR=${BASE_ROOT}/logs/train/${RUN_GROUP}
GENERATED_CONFIG_DIR=${LOGDIR}/generated_configs
mkdir -p "${LOGDIR}" "${GENERATED_CONFIG_DIR}"
printf '%s\n' "${RUN_GROUP}" > "${BASE_ROOT}/latest_aokvqa_first4_officialvz_train_run.txt"
STAGE3_OUT="${BASE_OUT}/03_ema_opsd_nogt_official_gbs8"
STAGE3_RESUME="${STAGE3_OUT}/step_17000"
echo "[$(date --iso-8601=seconds)] resume 03_ema_opsd_nogt_official_gbs8 from step_17000" | tee -a "${LOGDIR}/sequence.log"
torchrun \
  --nproc-per-node=4 \
  --master-port="${MASTER_PORT_BASE}" \
  visionzip_aokvqa/train.py \
  --config configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_gbs8.yaml \
  --output_dir "${STAGE3_OUT}" \
  --start_step 17000 \
  --adapter_path "${STAGE3_RESUME}" \
  2>&1 | tee -a "${LOGDIR}/03_ema_opsd_nogt_official_gbs8.log"
echo "[$(date --iso-8601=seconds)] done 03_ema_opsd_nogt_official_gbs8" | tee -a "${LOGDIR}/sequence.log"
FREEZE_CONFIG="${GENERATED_CONFIG_DIR}/aokvqa_opsd_sft_teacher_freeze_nogt_gbs8.yaml"
python - "configs/visionzip_aokvqa/aokvqa_opsd_sft_teacher_freeze_nogt_gbs8.yaml" "${FREEZE_CONFIG}" "${BASE_OUT}/01_sft_official_gbs8/final" <<'PY'
import sys
from pathlib import Path
import yaml
src = Path(sys.argv[1]); out = Path(sys.argv[2]); teacher_adapter = sys.argv[3]
cfg = yaml.safe_load(src.read_text(encoding='utf-8'))
cfg.setdefault('opsd', {})['teacher_adapter_path'] = teacher_adapter
assert cfg['training']['method'] == 'opsd_nogt'
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding='utf-8')
print(out)
PY
echo "[$(date --iso-8601=seconds)] start 04_freeze_sft_teacher_opsd_nogt_official_gbs8" | tee -a "${LOGDIR}/sequence.log"
torchrun \
  --nproc-per-node=4 \
  --master-port="$((MASTER_PORT_BASE + 1))" \
  visionzip_aokvqa/train.py \
  --config "${FREEZE_CONFIG}" \
  --output_dir "${BASE_OUT}/04_freeze_sft_teacher_opsd_nogt_official_gbs8" \
  2>&1 | tee "${LOGDIR}/04_freeze_sft_teacher_opsd_nogt_official_gbs8.log"
echo "[$(date --iso-8601=seconds)] done 04_freeze_sft_teacher_opsd_nogt_official_gbs8" | tee -a "${LOGDIR}/sequence.log"
