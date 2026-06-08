#!/bin/bash
# Run inside an existing 4xL40S SLURM allocation, for example:
#   srun --jobid=<ALLOC_JOB_ID> --overlap --ntasks=1 --cpus-per-task=8 \
#     bash opsd/slurm/run_single_r025_ddp4_pipeline_in_alloc.sh
set -euo pipefail

source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz

export HF_HOME=/scratch/enmingzz/hf_cache
export TRANSFORMERS_CACHE=/scratch/enmingzz/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

OUT_ROOT=/scratch/enmingzz/temp/opsd_runs/real_single_r025
TRAIN_DIR="${OUT_ROOT}/divprune_kl_single_r025_ddp4"
EVAL_DIR="${OUT_ROOT}/evals_ddp4"
ANALYSIS_DIR="${OUT_ROOT}/analysis_limit200_ddp4"
LOG_DIR=/scratch/enmingzz/temp/opsd_logs

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

archive_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    local archived="${path}_restart_$(date +%Y%m%d_%H%M%S)"
    echo "Archiving existing ${path} to ${archived}"
    mv "${path}" "${archived}"
  fi
}

# The user requested a fresh restart for this parallel run.
archive_if_exists "${TRAIN_DIR}"
archive_if_exists "${EVAL_DIR}"
archive_if_exists "${ANALYSIS_DIR}"
mkdir -p "${EVAL_DIR}" "${ANALYSIS_DIR}"

echo "Starting 4-GPU DDP teacher-rollout KL training at $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  opsd/scripts/train_qwen25vl_prune_distill.py \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_single_r025.yaml \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /scratch/xxluo/mscoco/train2017 \
  --output_dir "${TRAIN_DIR}" \
  --max_steps 1000 \
  --save_every 250 \
  --eval_every 250 \
  --gradient_accumulation_steps 4 \
  --attn_implementation flash_attention_2

echo "Training complete at $(date)"

run_eval() {
  local step="$1"
  local gpu="$2"
  local ckpt="${TRAIN_DIR}/step_${step}"
  local out_jsonl="${EVAL_DIR}/eval_step${step}_r025_limit200.jsonl"
  local out_log="${EVAL_DIR}/eval_step${step}_r025_limit200.log"

  if [[ ! -d "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    return 2
  fi

  echo "Starting eval for step ${step} on GPU ${gpu} at $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" python opsd/scripts/eval_qwen25vl_pruned_student.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_path "${ckpt}" \
    --eval_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
    --image_root /scratch/xxluo/mscoco/train2017 \
    --output_jsonl "${out_jsonl}" \
    --keep_ratio 0.25 \
    --pruner divprune_lite \
    --student_input_mode drop_tokens \
    --max_new_tokens 32 \
    --limit 200 \
    --attn_implementation flash_attention_2 \
    > "${out_log}" 2>&1
  echo "Finished eval for step ${step} at $(date)"
}

run_eval 250 0 &
pid250=$!
run_eval 500 1 &
pid500=$!
run_eval 750 2 &
pid750=$!
run_eval 1000 3 &
pid1000=$!

wait "${pid250}"
wait "${pid500}"
wait "${pid750}"
wait "${pid1000}"

echo "All checkpoint evals complete at $(date)"

python opsd/scripts/analyze_teacher_agreement.py \
  --baseline_summary_csv "${OUT_ROOT}/baseline_sweep_200/summary.csv" \
  --training_log_jsonl "${TRAIN_DIR}/training_log.jsonl" \
  --output_dir "${ANALYSIS_DIR}" \
  --title "OPSD real single-ratio divprune_lite r=0.25 limit=200 DDP4" \
  --eval_jsonl "${EVAL_DIR}/eval_step250_r025_limit200.jsonl" \
  --checkpoint_name step_250 \
  --eval_jsonl "${EVAL_DIR}/eval_step500_r025_limit200.jsonl" \
  --checkpoint_name step_500 \
  --eval_jsonl "${EVAL_DIR}/eval_step750_r025_limit200.jsonl" \
  --checkpoint_name step_750 \
  --eval_jsonl "${EVAL_DIR}/eval_step1000_r025_limit200.jsonl" \
  --checkpoint_name step_1000

cp "${ANALYSIS_DIR}/report.md" "${OUT_ROOT}/report.md"
echo "Analysis complete at $(date)"
