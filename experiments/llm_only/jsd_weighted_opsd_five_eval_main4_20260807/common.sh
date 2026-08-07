#!/usr/bin/env bash

set -euo pipefail

OPSD_ROOT=/project/6101803/enmingzz/opsd
PROJECT_ROOT=/project/6101803/enmingzz
CAMPAIGN_ROOT="${OPSD_ROOT}/experiments/llm_only/jsd_weighted_opsd_five_eval_main4_20260807"
OUT_ROOT=/scratch/enmingzz/outputs/llm_only/eval/jsd_weighted_opsd_five_step10240_merged_20260807
MERGED_ROOT=/scratch/enmingzz/outputs/llm_only/merged_models/jsd_weighted_opsd_five_20260807
BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

configure_method() {
  local method="${1:?method required}"
  case "${method}" in
    direct_inv_d25)
      METHOD_TAG=direct_inverse_d25_opsd_step10240_merged
      TRAIN_JOB_ID=4636577
      RUN_ROOT=/scratch/enmingzz/outputs/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/train_10240
      TRAIN_CONFIG="${OPSD_ROOT}/experiments/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/configs/train_10240.yaml"
      EXPECTED_MODE=jsd_over_current_kl_direct_inverse_batch
      EXPECTED_DELTA=0.25
      ;;
    softmax_t005_d25)
      METHOD_TAG=softmax_t005_d25_opsd_step10240_merged
      TRAIN_JOB_ID=4636680
      RUN_ROOT=/scratch/enmingzz/outputs/llm_only/jsd_current_kl_softmax_dropout0_20260806/train_T005_10240
      TRAIN_CONFIG="${OPSD_ROOT}/experiments/llm_only/jsd_current_kl_softmax_dropout0_20260806/configs/train_T005_10240.yaml"
      EXPECTED_MODE=jsd_over_current_kl_softmax_batch
      EXPECTED_DELTA=0.25
      ;;
    softmax_t010_d25)
      METHOD_TAG=softmax_t010_d25_opsd_step10240_merged
      TRAIN_JOB_ID=4636754
      RUN_ROOT=/scratch/enmingzz/outputs/llm_only/jsd_current_kl_softmax_dropout0_20260806/train_T010_10240
      TRAIN_CONFIG="${OPSD_ROOT}/experiments/llm_only/jsd_current_kl_softmax_dropout0_20260806/configs/train_T010_10240.yaml"
      EXPECTED_MODE=jsd_over_current_kl_softmax_batch
      EXPECTED_DELTA=0.25
      ;;
    softmax_t005_d40)
      METHOD_TAG=softmax_t005_d40_opsd_step10240_merged
      TRAIN_JOB_ID=4637029
      RUN_ROOT=/scratch/enmingzz/outputs/llm_only/jsd_current_kl_softmax_reldelta040_dropout0_20260806/train_T005_10240
      TRAIN_CONFIG="${OPSD_ROOT}/experiments/llm_only/jsd_current_kl_softmax_reldelta040_dropout0_20260806/configs/train_T005_10240.yaml"
      EXPECTED_MODE=jsd_over_current_kl_softmax_batch
      EXPECTED_DELTA=0.40
      ;;
    softmax_t010_d40)
      METHOD_TAG=softmax_t010_d40_opsd_step10240_merged
      TRAIN_JOB_ID=4637030
      RUN_ROOT=/scratch/enmingzz/outputs/llm_only/jsd_current_kl_softmax_reldelta040_dropout0_20260806/train_T010_10240
      TRAIN_CONFIG="${OPSD_ROOT}/experiments/llm_only/jsd_current_kl_softmax_reldelta040_dropout0_20260806/configs/train_T010_10240.yaml"
      EXPECTED_MODE=jsd_over_current_kl_softmax_batch
      EXPECTED_DELTA=0.40
      ;;
    *)
      echo "Unsupported method: ${method}" >&2
      return 2
      ;;
  esac
  METHOD="${method}"
  ADAPTER_PATH="${RUN_ROOT}/final"
  MERGED_MODEL="${MERGED_ROOT}/${METHOD}_step10240_qwen25vl7b_merged_bf16"
  CAMPAIGN_GROUP="${METHOD_TAG}_reasoning_main4_cleanarmen_20260807"
  export METHOD METHOD_TAG TRAIN_JOB_ID RUN_ROOT TRAIN_CONFIG EXPECTED_MODE
  export EXPECTED_DELTA ADAPTER_PATH MERGED_MODEL CAMPAIGN_GROUP
}
