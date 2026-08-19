#!/usr/bin/env bash

set -euo pipefail

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
CAMPAIGN_ROOT="${OPSD_ROOT}/experiments/llm_only/f_trajectory_bottom80_eval_main3_merged_20260818"
OUT_ROOT=/scratch/enmingzz/outputs/llm_only/eval/f_trajectory_bottom80_step10240_merged_main3_20260818
MERGED_ROOT=/scratch/enmingzz/outputs/llm_only/merged_models/f_trajectory_bottom80_20260818
BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

set_method() {
  local method="${1:?method is required}"
  [[ "${method}" == "ft80" ]] || { echo "Unknown method: ${method}" >&2; return 2; }
  METHOD=ft80
  METHOD_LONG=trajectory_bottom80_delta002
  METHOD_TAG=opsd_r010_trajectory_bottom80_d002_dropout0_step10240_merged
  TRAIN_JOB_ID=4845079
  TRAIN_CONFIG="${OPSD_ROOT}/experiments/llm_only/opsd_r010_f_trajectory_partition_delta002_dropout0_20260817/configs/train_trajectory_bottom80.yaml"
  RUN_ROOT=/scratch/enmingzz/outputs/llm_only/opsd_r010_f_trajectory_partition_delta002_dropout0_20260817/trajectory_bottom80/run
  ADAPTER_PATH="${RUN_ROOT}/resume_checkpoints/step_010240"
  TRAINING_OBJECTIVE=trajectory_bottom80_F_partition_opsd
  MERGED_MODEL="${MERGED_ROOT}/${METHOD_TAG}_qwen25vl7b_bf16"
  CAMPAIGN_GROUP="${METHOD_TAG}_reasoning_main3_cleanarmen_20260818"
  export METHOD METHOD_LONG METHOD_TAG TRAIN_JOB_ID TRAIN_CONFIG RUN_ROOT ADAPTER_PATH
  export TRAINING_OBJECTIVE MERGED_MODEL CAMPAIGN_GROUP
}

export OPSD_ROOT PROJECT_ROOT CAMPAIGN_ROOT OUT_ROOT MERGED_ROOT BASE_MODEL
