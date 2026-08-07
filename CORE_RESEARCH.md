# Core Research Code

This document indexes the minimal code needed to reproduce the current
Qwen2.5-VL + VisionZip research workflow. Checkpoints, datasets, generated
rollouts, benchmark predictions, plots, Slurm logs, and result tables are not
part of this release.

## Scope

The release contains four components:

1. pass@k generation and judging;
2. LLM-only SFT, OPSD, and budget-sensitivity-weighted OPSD training;
3. JSD budget-sensitivity measurement across checkpoints;
4. merged-checkpoint evaluation on MME, MMStar, MathVista, and MMMU-Pro.

The model is `Qwen/Qwen2.5-VL-7B-Instruct`. Vision pruning uses the official
VisionZip Qwen2.5-VL implementation. The evaluation wrapper targets Armen's
VLMEvalKit commit `51682a6baab948d3dbb4b867a3eab178504ac3f5` plus the patch in
`patches/vlmevalkit_armen51682_cleanenv_qwen25vl_lora_visionzip.patch`.

## Layout

### Training

Core implementation:

- `visionzip_aokvqa/train.py`: SFT and OPSD training loop, DDP, checkpointing,
  EMA teacher, per-sample ratio sampling, cached rollouts, and weighted losses.
- `visionzip_aokvqa/qwen_wrapper.py`: Qwen2.5-VL loading, official VisionZip
  forwarding, generation, LoRA setup, and teacher/student context handling.
- `visionzip_aokvqa/losses.py`: forward KL, per-token KL, JSD, CE, and related
  memory-safe loss primitives.
- `visionzip_aokvqa/native_budget_weighting.py`: native-budget probe and token
  weighting functions.
- `visionzip_aokvqa/trajectory_weighting.py`: effective-batch trajectory
  weighting, including JSD/current-KL variants.
- `visionzip_aokvqa/paired_sampling.py`: deterministic paired ratio and rollout
  seeds used for matched comparisons.

Representative configs:

- SFT: `experiments/llm_only/lcot20k_retrain_dropout0_20260803/configs/sft20k.yaml`
- original OPSD matched control:
  `experiments/llm_only/teacher_gap_persistence_opsd_pilot_20260801/autoresearch/trajectory_curriculum/configs/matched_vanilla_probe_exact_9984.yaml`
- JSD/current-KL weighted OPSD:
  `experiments/llm_only/jsd_trajectory_reweight_dropout0_20260806/configs/jsd_over_current_kl_10240.yaml`
- direct inverse JSD/current-KL:
  `experiments/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/configs/train_10240.yaml`
- softmax inverse JSD/current-KL:
  `experiments/llm_only/jsd_current_kl_softmax_dropout0_20260806/configs/train_T010_10240.yaml`

The configs intentionally preserve the exact cluster paths used by the runs.
Change `dataset.name`, `dataset.image_root`, `experiment.output_root`, and local
VisionZip/VLMEvalKit paths for another machine.

Run a four-GPU training job from the directory containing this repository:

```bash
torchrun --nproc-per-node=4 --master-port=29500 \
  opsd/visionzip_aokvqa/train.py \
  --config opsd/experiments/llm_only/lcot20k_retrain_dropout0_20260803/configs/sft20k.yaml \
  --output_dir /path/to/output
```

Use the same command with one of the OPSD configs above. All released configs
use LLM-only LoRA, bf16, LoRA dropout 0, and do not train the vision encoder,
projector, or VisionZip parameters.

### OPSD objective

For a student distribution `p_b` at retained visual-token budget `b` and a
full-token teacher distribution `q_full`, the implemented OPSD direction is:

```text
KL(q_full || p_b)
```

The teacher sees the same student-generated textual prefix and full visual
tokens. OPSD does not receive the ground-truth answer in its teacher prompt.

### JSD budget sensitivity

For a native VisionZip budget intervention `b -> b_plus`, the metric uses the
same generated textual prefix and independently evaluates VisionZip at both
budgets:

```text
m_t   = 0.5 * (p_b,t + p_b_plus,t)
JSD_t = 0.5 * KL(p_b,t || m_t) + 0.5 * KL(p_b_plus,t || m_t)
```

The trajectory metric is the valid-response-token mean. The current weighted
OPSD variants remove ratio difficulty with the corresponding teacher KL and
then normalize trajectory weights over the effective batch. Probe branches and
weights are detached; the optimized token loss remains `KL(q_full || p_b)`.

Metric implementation and checkpoint evaluator:

- `hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/run_checkpoint.py`
- `hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/aggregate.py`
- `hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/compare_sft_opsd.py`
- `hypothesis_validate/opsd_checkpoint_kl_trajectory/tests/test_metrics.py`

Example:

```bash
python hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/run_checkpoint.py \
  --config hypothesis_validate/opsd_checkpoint_kl_trajectory/config_max1024.json \
  --checkpoint-label step_1024 \
  --adapter-path /path/to/adapter \
  --output-dir /path/to/jsd_metrics \
  --limit 100 \
  --merge-student \
  --full-reference self_checkpoint
```

This stores raw per-token metrics before aggregation.

### pass@k

The pass@k path always writes raw model outputs and token IDs before parsing or
judging. The fixed-random-mask runner reuses one mask for all 64 rollouts of a
sample, while different samples may use different deterministic masks.

Core scripts:

- `hypothesis_validate/scripts/run_raw_rollout_experiment.py`
- `hypothesis_validate/scripts/run_random_fixed_mask_rollout_experiment.py`
- `hypothesis_validate/scripts/judge_raw_rollout_experiment_qwen36_27b.py`
- `hypothesis_validate/scripts/summarize_qwen36_judge_only_sweep.py`
- `hypothesis_validate/scripts/validate_raw_rollout_sweep.py`

Example VisionZip pass@64 generation:

```bash
python hypothesis_validate/scripts/run_raw_rollout_experiment.py \
  --samples /path/to/mmstar_numeric_clean100.jsonl \
  --output-dir /path/to/raw_rollouts/r020 \
  --benchmark MMStar_OpenEnded \
  --decode-mode sample \
  --pruning visionzip \
  --retention-ratio 0.20 \
  --num-rollouts 64 \
  --temperature 0.7 \
  --top-p 0.95 \
  --top-k 50 \
  --max-new-tokens 1024
```

Run a separate greedy command with `--decode-mode greedy`. Correctness judging
is deliberately a separate stage so changing a parser or judge never requires
regenerating the expensive raw rollouts.

### Benchmark evaluation

The current merged-model launcher is under:

```text
experiments/llm_only/jsd_weighted_opsd_five_eval_main4_20260807/
```

Its `eval_one_case.sbatch` supports:

- datasets: `mme`, `mmstar`, `mathvista`, `mmmupro`;
- ratios: `noprune`, `r010`, `r020`, `r030`;
- merged LLM-only LoRA checkpoints;
- raw prediction preservation;
- strict output validation;
- Qwen post-processing where configured.

Submit one case with:

```bash
sbatch experiments/llm_only/jsd_weighted_opsd_five_eval_main4_20260807/eval_one_case.sbatch \
  METHOD_TAG mmstar r020
```

Set method/checkpoint paths in that campaign's `common.sh`. The launcher checks
adapter hashes, merged-model provenance, LLM-only scope, LoRA dropout, and
checkpoint step before evaluation.

## Validation

CPU checks:

```bash
python -m py_compile \
  visionzip_aokvqa/*.py \
  hypothesis_validate/scripts/run_raw_rollout_experiment.py \
  hypothesis_validate/scripts/run_random_fixed_mask_rollout_experiment.py \
  hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/*.py

pytest -q \
  tests/test_token_budget_stability.py \
  tests/test_trajectory_weighting.py \
  hypothesis_validate/tests/test_random_fixed_mask_rollout.py \
  hypothesis_validate/opsd_checkpoint_kl_trajectory/tests/test_metrics.py
```

GPU smoke tests should be run through the cluster environment that supplies the
official VisionZip Qwen2.5-VL module and the aligned Transformers checkout.

## Deliberately excluded

This release does not include benchmark outputs, judged answers, plots,
checkpoint snapshots, merged models, training logs, Slurm submission records,
PDF reviews, or local datasets. Those artifacts are experiment outputs rather
than executable research code.
