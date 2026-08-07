#!/usr/bin/env bash
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=kn136
#SBATCH --job-name=mr_p64
#SBATCH --output=/project/6101803/enmingzz/opsd/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_random_fixedmask_qwen25vl7b_t07_p095_20260722/slurm_logs/%x-%j.out
#SBATCH --error=/project/6101803/enmingzz/opsd/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_random_fixedmask_qwen25vl7b_t07_p095_20260722/slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/project/6101803/enmingzz/opsd
SWEEP_ROOT="${SWEEP_ROOT:-${ROOT}/hypothesis_validate/pass_at_k/mmstar_numeric_clean100_random_fixedmask_qwen25vl7b_t07_p095_20260722}"
SAMPLE_FILE="${ROOT}/hypothesis_validate/manual_review/mmstar_numeric_open_ended_clean100_seed42/samples.jsonl"
ENV_SCRIPT=/project/6101803/enmingzz/ckpt_eval_trainenv/env_train_runtime.sh
RUNNER="${ROOT}/hypothesis_validate/scripts/run_random_fixed_mask_rollout_experiment.py"
JUDGE="${ROOT}/hypothesis_validate/scripts/judge_raw_rollout_experiment_qwen36_27b.py"
REPAIR="${ROOT}/hypothesis_validate/scripts/repair_qwen36_special_token_transport.py"
VALIDATE="${ROOT}/hypothesis_validate/scripts/validate_qwen36_judge_condition.py"
SUMMARIZE="${ROOT}/hypothesis_validate/scripts/summarize_mmstar_clean100_sweep.py"
VLLM=/scratch/enmingzz/temp/venvs/mmecot-qwen36-judge/bin/vllm
MODEL_PATH=/scratch/enmingzz/models/Qwen3.6-27B-FP8
SERVED_MODEL=Qwen/Qwen3.6-27B-FP8
LIMIT="${LIMIT:-100}"
START_INDEX="${START_INDEX:-0}"
MASK_SEED="${MASK_SEED:-42}"

: "${RATIO:?Set RATIO to 0.1, 0.2, 0.3, 0.4, or 1.0}"
: "${RATIO_TAG:?Set RATIO_TAG to r010, r020, r030, r040, or r100}"

case "${RATIO_TAG}:${RATIO}" in
    r010:0.1|r020:0.2|r030:0.3|r040:0.4|r100:1.0) ;;
    *) echo "Inconsistent RATIO_TAG:RATIO=${RATIO_TAG}:${RATIO}" >&2; exit 2 ;;
esac

CONDITION_DIR="${SWEEP_ROOT}/mmstar_open_ended/${RATIO_TAG}"
MASK_DIR="${CONDITION_DIR}/fixed_masks"
SOURCE_JUDGE_DIR="${CONDITION_DIR}/qwen36_27b_judge_only_v1"
SAFE_JUDGE_DIR="${CONDITION_DIR}/qwen36_27b_judge_only_v2_safe_special_token_transport"
mkdir -p "${CONDITION_DIR}" "${MASK_DIR}" "${SWEEP_ROOT}/slurm_logs"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
    echo "This job requires exactly one visible GPU." >&2
    exit 1
fi

run_generation() {
    local decode_label="$1"
    local runner_mode="$2"
    local run_dir="${CONDITION_DIR}/${decode_label}"
    echo "[$(date --iso-8601=seconds)] random fixed-mask generation start ratio=${RATIO_TAG} mode=${decode_label} job=${SLURM_JOB_ID}"
    (
        # shellcheck source=/project/6101803/enmingzz/ckpt_eval_trainenv/env_train_runtime.sh
        # shellcheck disable=SC1091
        source "${ENV_SCRIPT}"
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS=8
        python "${RUNNER}" \
            --samples "${SAMPLE_FILE}" \
            --output-dir "${run_dir}" \
            --mask-dir "${MASK_DIR}" \
            --benchmark MMStar_OpenEnded \
            --decode-mode "${runner_mode}" \
            --retention-ratio "${RATIO}" \
            --num-rollouts 64 \
            --rollout-batch-size 32 \
            --temperature 0.7 \
            --top-p 0.95 \
            --top-k 50 \
            --max-new-tokens 1024 \
            --seed 42 \
            --mask-seed "${MASK_SEED}" \
            --start-index "${START_INDEX}" \
            --limit "${LIMIT}" \
            --no-continue-on-error
    )
    jq -e \
        '.status == "passed" and .failures == 0 and .pruning == "random" and
         .fixed_mask_invariant_all_samples == true and
         .fixed_prefix_invariant_all_samples == true and
         .sample_mask_seed_invariant_all_samples == true and
         .target_token_count_matches_all_records == true and
         (.mask_document_errors | length) == 0' \
        "${run_dir}/raw_validation.json" >/dev/null
    echo "[$(date --iso-8601=seconds)] random fixed-mask generation validated ratio=${RATIO_TAG} mode=${decode_label}"
}

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
run_generation greedy greedy
run_generation sample64 sample

jq -e \
    '.cross_mode_mask_check.counterpart_available == true and
     .cross_mode_mask_check.matched_samples == .sample_count and
     (.cross_mode_mask_check.mismatched_samples | length) == 0' \
    "${CONDITION_DIR}/sample64/raw_validation.json" >/dev/null

GREEDY_RAW="${CONDITION_DIR}/greedy/raw_outputs.jsonl"
SAMPLE_RAW="${CONDITION_DIR}/sample64/raw_outputs.jsonl"
if [[ ! -s "${GREEDY_RAW}" || ! -s "${SAMPLE_RAW}" ]]; then
    echo "Both greedy and sample64 raw outputs are required before post-judge." >&2
    exit 1
fi
if [[ ! -x "${VLLM}" || ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen3.6-27B judge environment or checkpoint is missing." >&2
    exit 1
fi

unset PYTHONPATH TRANSFORMERS_CACHE
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PORT="$((18000 + SLURM_JOB_ID % 10000))"
SERVER_LOG="${SWEEP_ROOT}/slurm_logs/qwen36_server_${RATIO_TAG}_${SLURM_JOB_ID}.log"

"${VLLM}" serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL}" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 128 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --language-model-only \
    --generation-config vllm \
    --enforce-eager \
    --disable-uvicorn-access-log \
    --disable-log-stats \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 150); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Judge server exited during startup." >&2
        tail -n 120 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
    echo "Judge server did not become healthy within 300 seconds." >&2
    tail -n 120 "${SERVER_LOG}" >&2 || true
    exit 1
fi

echo "[$(date --iso-8601=seconds)] Qwen3.6 judge healthy ratio=${RATIO_TAG} port=${PORT}"
python3 "${JUDGE}" \
    --benchmark MMStar_OpenEnded \
    --greedy-raw "${GREEDY_RAW}" \
    --sample64-raw "${SAMPLE_RAW}" \
    --output-dir "${SOURCE_JUDGE_DIR}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --expected-samples "${LIMIT}" \
    --concurrency 64

python3 "${REPAIR}" \
    --source-dir "${SOURCE_JUDGE_DIR}" \
    --output-dir "${SAFE_JUDGE_DIR}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --concurrency 64

python3 "${VALIDATE}" \
    --sweep-root "${SWEEP_ROOT}" \
    --ratio "${RATIO_TAG}" \
    --expected-samples "${LIMIT}"

if [[ "${RUN_SUMMARY:-0}" -eq 1 && "${LIMIT}" -eq 100 && "${START_INDEX}" -eq 0 ]]; then
    python3 "${SUMMARIZE}" --sweep-root "${SWEEP_ROOT}"
fi
echo "[$(date --iso-8601=seconds)] random fixed-mask generation and safe post-judge complete ratio=${RATIO_TAG}"
