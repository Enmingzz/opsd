# OPSD Qwen2.5-VL Fine-Tuning Workflow

OPSD means On-Policy Sparse-token Distillation for VLMs. The main method is pure KL distillation from a full-token frozen Qwen2.5-VL teacher into a LoRA student that sees pruned post-vision visual tokens.

## Environment

```bash
source /project/6101803/enmingzz/env/vsi-official.sh
cd /project/6101803/enmingzz
```

## 1. Compile And Basic Smoke

```bash
python -m py_compile $(find opsd -name '*.py' | sort)
python opsd/tests/test_qwen25vl_pruned_forward.py
```

## 2. Normal-Resolution Stress Test

```bash
python opsd/tests/stress_test_normal_resolution.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --attn_implementation flash_attention_2
```

## 3. Prepare LLaVA Subset

```bash
python opsd/scripts/prepare_llava_instruct_jsonl.py \
  --llava_json /path/to/llava_instruct_150k.json \
  --image_root /path/to/coco/train2017 \
  --output_train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --output_val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --limit 21000 \
  --val_ratio 0.05 \
  --seed 42
```

Answers are retained for evaluation, debugging, and optional filtering only. The main OPSD training loss does not use supervised CE.

## 4. Validate JSONL

```bash
python opsd/scripts/validate_opsd_jsonl.py \
  --jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --image_root /path/to/coco/train2017
```

## 5. No-Training Baseline Sweep

```bash
python opsd/scripts/sweep_pruned_baselines.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --eval_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /path/to/coco/train2017 \
  --output_dir /scratch/enmingzz/temp/opsd_runs/baseline_sweep_llava1k \
  --pruners random,grid,divprune_lite,vscan_stage1 \
  --keep_ratios 1.0,0.5,0.25,0.125 \
  --student_input_mode drop_tokens \
  --max_new_tokens 32 \
  --limit 1000 \
  --seed 42 \
  --attn_implementation flash_attention_2
```

## 6. Baseline Report

```bash
python opsd/scripts/report_pruned_baseline_sweep.py \
  --sweep_dir /scratch/enmingzz/temp/opsd_runs/baseline_sweep_llava1k
```

## 7. Teacher-Rollout KL Warmup, Single Ratio 0.25

```bash
python opsd/scripts/train_qwen25vl_prune_distill.py \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_single_r025.yaml \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /path/to/coco/train2017 \
  --output_dir /scratch/enmingzz/temp/opsd_runs/divprune_kl_single_r025 \
  --max_steps 1000 \
  --save_every 500 \
  --eval_every 500
```

## 8. Teacher-Rollout KL Warmup, Multi-Ratio

```bash
python opsd/scripts/train_qwen25vl_prune_distill.py \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_multiratio.yaml \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /path/to/coco/train2017 \
  --output_dir /scratch/enmingzz/temp/opsd_runs/divprune_kl_multiratio \
  --max_steps 1000 \
  --save_every 500 \
  --eval_every 500
```

## 9. Mixed On-Policy Continuation

```bash
python opsd/scripts/train_qwen25vl_prune_distill.py \
  --config opsd/configs/prune_distill/qwen25vl_7b_lora_llava_divprune_onpolicy.yaml \
  --adapter_path /scratch/enmingzz/temp/opsd_runs/divprune_kl_multiratio/step_1000 \
  --train_jsonl /scratch/enmingzz/temp/opsd_data/llava20k_train.jsonl \
  --val_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /path/to/coco/train2017 \
  --output_dir /scratch/enmingzz/temp/opsd_runs/divprune_kl_multiratio_onpolicy \
  --max_steps 1000 \
  --save_every 500 \
  --eval_every 500
```

## 10. Evaluation After Training

```bash
python opsd/scripts/eval_qwen25vl_pruned_student.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter_path /scratch/enmingzz/temp/opsd_runs/divprune_kl_multiratio/step_1000 \
  --eval_jsonl /scratch/enmingzz/temp/opsd_data/llava1k_val.jsonl \
  --image_root /path/to/coco/train2017 \
  --output_jsonl /scratch/enmingzz/temp/opsd_runs/divprune_kl_multiratio/eval_r025.jsonl \
  --keep_ratio 0.25 \
  --pruner divprune_lite \
  --student_input_mode drop_tokens \
  --max_new_tokens 32 \
  --limit 1000 \
  --attn_implementation flash_attention_2
```

Also evaluate keep ratios `0.5`, `0.25`, and `0.125` for raw pruned base, single-ratio distilled student, multi-ratio distilled student, and mixed on-policy continued student.

## Compare Eval Outputs

```bash
python opsd/scripts/compare_eval_outputs.py \
  --full_predictions /path/to/full_token.jsonl \
  --pruned_base_predictions /path/to/pruned_base.jsonl \
  --distilled_predictions /path/to/distilled_student.jsonl \
  --output_dir /scratch/enmingzz/temp/opsd_runs/compare_eval
```

## Current Limits

- Single image per sample.
- Batch size 1.
- No video support.
- No trainable pruner.
- Main OPSD method is pure KL, not supervised CE.
- Keep `vscan_merge_dropped=false` unless you explicitly want the pruner to mutate visual embeddings before pruned forward.
