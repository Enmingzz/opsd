# Core Research Code

This document indexes the minimal code needed to reproduce the current
Qwen2.5-VL + VisionZip research workflow. Checkpoints, datasets, generated
rollouts, benchmark predictions, plots, Slurm logs, and result tables are not
part of this release.

## Scope

The release contains four components:

1. pass@k generation and judging;
2. LLM-only SFT, OPSD, budget-sensitivity weighting, and F/projection
   partition controls;
3. JSD budget-sensitivity measurement across checkpoints;
4. merged-checkpoint evaluation on MME, MMStar, MathVista, MathVerse, and
   MMMU-Pro.

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
- global trajectory-F affine/curriculum controls:
  `experiments/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818/`
- hard top/bottom trajectory-F partitions:
  `experiments/llm_only/opsd_r010_f_trajectory_partition_delta002_dropout0_20260817/`
- hard token-F partitions and deterministic random-drop control:
  `experiments/llm_only/opsd_r010_f_token_partition_delta002_dropout0_20260818/`
- projection-mass VA-style grouped weighting:
  `experiments/llm_only/opsd_r010_projection_mass_va_group_dropout0_20260818/`

### LLaVA-CoT 20K training

The complete 20K training contract is in
`experiments/llm_only/lcot20k_retrain_dropout0_20260803/`. It contains:

- fresh LLM-only SFT over all 20,000 ordered examples;
- fresh two-stage progressive OPSD over all 20,000 examples;
- smoke configs, Slurm launchers, resume validation, and data-order tests.

The original random-ratio OPSD trajectory can be extended exactly from step
9,984 to step 20,000 with
`experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/`.
That path restores LoRA, optimizer, EMA teacher, data cursor, and rank-specific
RNG state rather than warm-starting a second run.

Validate the fresh 20K configs before submission:

```bash
python experiments/llm_only/lcot20k_retrain_dropout0_20260803/validate_configs.py
```

The ordered JSONL and image bank are deliberately not committed. Their exact
paths, SHA256 values, construction metadata, and decontamination contract are
recorded under
`data/openmmreasoner_llava_cot_train20k_ordered_decontam_v1_seed42/`. The
dataset construction entry point is `scripts/data/build_ordered_llava_cot_20k.py`.

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

### Projection and F controls

The current mechanism controls run native VisionZip at `b` and `b_plus` on
the same generated textual prefix. Given teacher/student JSD `A`, native
budget JSD `B`, and teacher/expanded-budget JSD `C`, the projection mass is
`P = (A + B - C) / 2` and the projection fraction is `F = P / A` with the
configured numerical guard. Implementations live in
`visionzip_aokvqa/native_budget_weighting.py` and
`visionzip_aokvqa/trajectory_weighting.py`; integration and detached probe
execution live in `visionzip_aokvqa/train.py`.

The deterministic random-token-drop control preserves the same valid token
pool and loss mask as the token-F partition experiment, but drops 20% of
eligible response tokens using a stable hash of the global seed, sample ID,
rollout seed, and token index. It is reproducible across DDP ranks and does not
depend on incidental RNG consumption.

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

The proven step-20K merged-model launchers are under:

- `experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/eval_step20000_qwen5_20260804/`;
- `experiments/llm_only/lcot20k_retrain_dropout0_20260803/eval_sft20k_qwen5_20260804/`;
- `experiments/llm_only/lcot20k_retrain_dropout0_20260803/eval_progressive20k_qwen5_20260805/`.

Their `eval_one_case.sbatch` launchers support:

- datasets: `mme`, `mmstar`, `mathvista`, `mathverse`, `mmmupro`;
- ratios: `noprune`, `r010`, `r020`, `r030`;
- merged LLM-only LoRA checkpoints;
- raw prediction preservation;
- strict output validation;
- Qwen post-processing where configured.

Strict ground-truth-aware fallback post-processing is implemented separately
for MathVista and MathVerse in:

- `scripts/eval/postprocess_mathvista_strict_gt.py`;
- `scripts/eval/postprocess_mathverse_strict_gt.py`.

Both preserve raw predictions and apply deterministic extraction first. Only
unresolved examples are sent to the configured Qwen judge.

Submit all five benchmarks at all four evaluation ratios with the campaign's
`submit_all.sh`, or submit one case directly:

```bash
sbatch experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/eval_step20000_qwen5_20260804/eval_one_case.sbatch \
  mmstar r020
```

Each launcher checks adapter hashes, merged-model provenance, LLM-only scope,
LoRA dropout, checkpoint step, benchmark row count, and raw-output presence
before accepting a result.

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
  tests/test_f_partition_weighting.py \
  tests/test_projection_mass_grouped_weighting.py \
  tests/test_token_outlier_exclusion.py \
  tests/test_mathvista_strict_gt_postprocess.py \
  tests/test_mathverse_strict_gt_postprocess.py \
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
