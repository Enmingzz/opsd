#!/usr/bin/env bash

set -euo pipefail

OPSD_ROOT="${OPSD_ROOT:-/project/6101803/enmingzz/opsd}"
PROJECT_ROOT="${PROJECT_ROOT:-/project/6101803/enmingzz}"
EXP_ROOT="${OPSD_ROOT}/experiments/llm_only/opsd_random_r010_only_dropout0_20260815"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-10240}"

if [[ "${CHECKPOINT_STEP}" == "10240" ]]; then
  DEFAULT_RUN_ROOT=/scratch/enmingzz/outputs/llm_only/opsd_random_r010_only_dropout0_20260815/run
  DEFAULT_CONFIG="${EXP_ROOT}/configs/train_10240.yaml"
elif [[ "${CHECKPOINT_STEP}" == "20000" ]]; then
  DEFAULT_RUN_ROOT=/scratch/enmingzz/outputs/llm_only/opsd_r010_only_dropout0_lcot20k_20260818/run
  DEFAULT_CONFIG="${EXP_ROOT}/configs/train_20000_stage2.yaml"
else
  echo "Unsupported CHECKPOINT_STEP=${CHECKPOINT_STEP}; expected 10240 or 20000" >&2
  return 2 2>/dev/null || exit 2
fi

TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-${DEFAULT_RUN_ROOT}}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${DEFAULT_CONFIG}}"
ADAPTER_PATH="${ADAPTER_PATH:-${TRAIN_RUN_ROOT}/final}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
METHOD_TAG="${METHOD_TAG:-opsd_r010_only_dropout0_step${CHECKPOINT_STEP}_merged}"
MERGED_MODEL="${MERGED_MODEL:-/scratch/enmingzz/outputs/llm_only/merged_models/${METHOD_TAG}_qwen25vl7b_bf16}"
OUT_ROOT="${OUT_ROOT:-/scratch/enmingzz/outputs/llm_only/eval/${METHOD_TAG}_five_benchmarks}"
CAMPAIGN_GROUP="${CAMPAIGN_GROUP:-${METHOD_TAG}_reasoning_cleanarmen}"

export OPSD_ROOT PROJECT_ROOT EXP_ROOT CHECKPOINT_STEP TRAIN_RUN_ROOT TRAIN_CONFIG
export ADAPTER_PATH BASE_MODEL METHOD_TAG MERGED_MODEL OUT_ROOT CAMPAIGN_GROUP
