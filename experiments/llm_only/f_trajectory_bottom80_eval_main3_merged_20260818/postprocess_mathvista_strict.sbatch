#!/usr/bin/env bash
#SBATCH --account=aip-btaati
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=f3-mvgt
#SBATCH --output=/scratch/enmingzz/outputs/llm_only/eval/f_trajectory_bottom80_step10240_merged_main3_20260818/logs/%x-%j.out
#SBATCH --error=/scratch/enmingzz/outputs/llm_only/eval/f_trajectory_bottom80_step10240_merged_main3_20260818/logs/%x-%j.err

set -euo pipefail
[[ "$#" -eq 3 ]] || { echo "Usage: $0 METHOD RATIO INFERENCE_JOB_ID" >&2; exit 2; }
METHOD_ARG="$1"
RATIO="$2"
INFERENCE_JOB_ID="$3"
source /project/6101803/enmingzz/opsd/experiments/llm_only/f_trajectory_bottom80_eval_main3_merged_20260818/common.sh
set_method "${METHOD_ARG}"
case "${RATIO}" in noprune|r010|r020|r030) ;; *) echo "Unsupported ratio: ${RATIO}" >&2; exit 2 ;; esac

POSTPROCESS="${OPSD_ROOT}/scripts/eval/postprocess_mathvista_strict_gt.py"
PYTHON=/scratch/enmingzz/temp/venvs/mmecot-qwen36-judge/bin/python
VLLM=/scratch/enmingzz/temp/venvs/mmecot-qwen36-judge/bin/vllm
MODEL_PATH=/scratch/enmingzz/models/Qwen3.6-27B-FP8
SERVED_MODEL=Qwen/Qwen3.6-27B-FP8
WORK_DIR="${OUT_ROOT}/eval_vlmevalkit_trainenv/${CAMPAIGN_GROUP}/${METHOD_TAG}_${RATIO}_MathVista_MINI_${INFERENCE_JOB_ID}"
RESULT_FILE="$(find "${WORK_DIR}/Qwen" -type f -name 'Qwen_MathVista_MINI.xlsx' | sort | tail -n 1)"
OUTPUT_DIR="${OUT_ROOT}/mathvista_strict_gt/${METHOD}/${RATIO}"
SERVER_LOG="${OUTPUT_DIR}/vllm_server_${SLURM_JOB_ID}.log"
PORT=$((18000 + SLURM_JOB_ID % 10000))

[[ -n "${RESULT_FILE}" && -f "${RESULT_FILE}" ]]
[[ -x "${PYTHON}" && -x "${VLLM}" && -f "${MODEL_PATH}/config.json" ]]
mkdir -p "${OUTPUT_DIR}" "${OUT_ROOT}/logs" "${SCRATCH}/opsd_mathvista_strict_gt/hf_cache" \
  "${SCRATCH}/opsd_mathvista_strict_gt/tmp"
cd "${OPSD_ROOT}"
export HF_HOME="${SCRATCH}/opsd_mathvista_strict_gt/hf_cache"
export TMPDIR="${SCRATCH}/opsd_mathvista_strict_gt/tmp"
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1

"${PYTHON}" "${POSTPROCESS}" --result-file "${RESULT_FILE}" --output-dir "${OUTPUT_DIR}" \
  --method "${METHOD_LONG}" --ratio "${RATIO}" --expected-rows 1000 --validate-only

"${VLLM}" serve "${MODEL_PATH}" --host 127.0.0.1 --port "${PORT}" \
  --served-model-name "${SERVED_MODEL}" --tensor-parallel-size 1 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --max-num-seqs 32 --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' --language-model-only \
  --generation-config vllm --enforce-eager --disable-uvicorn-access-log \
  --disable-log-stats >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill -TERM "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 160 "${SERVER_LOG}" >&2; exit 1; }
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
"${PYTHON}" "${POSTPROCESS}" --result-file "${RESULT_FILE}" --output-dir "${OUTPUT_DIR}" \
  --method "${METHOD_LONG}" --ratio "${RATIO}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --judge-model "${SERVED_MODEL}" --concurrency 16 --expected-rows 1000
