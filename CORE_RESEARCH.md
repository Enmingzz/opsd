# Core Research Workflow

This is the maintained public entry point for the current Qwen2.5-VL +
VisionZip work. Generated datasets, checkpoints, predictions, judge outputs,
plots, Slurm logs, and result tables are intentionally excluded from Git.

## Current scope

The only active training budget published here is **10% retained visual
tokens (`r010`)**. The repository keeps:

1. original r010-only OPSD training on the ordered, decontaminated LLaVA-CoT
   data;
2. projection fraction (`F`) measurement and the current r010-only
   trajectory/token ablations;
3. pass@k generation and judging with raw rollout preservation;
4. merged-checkpoint evaluation on MME, MMStar, MathVista, MathVerse, and
   MMMU-Pro.

Older random-ratio, progressive, and JSD-weighted training launchers are not
part of this public active workflow. Generic implementation hooks remain in
the shared trainer where the r010 ablations depend on them.

The main research objective is **not yet complete**: identify a simple,
principled projection-fraction reweighting rule or modified OPSD objective
that improves over original r010-only OPSD on the benchmark suite.

## Status (2026-08-24)

### Completed

- Built the ordered, decontaminated 20K LLaVA-CoT training manifest and fixed
  image/token constraints.
- Completed and audited the 10,240-sample original r010-only OPSD run.
- Verified LLM-only LoRA scope: 392 tensors, 40,370,176 trainable parameters,
  no vision-encoder or projector adapter tensors.
- Implemented native VisionZip `b -> b_plus` probes on the same generated
  textual prefix and logged the projection geometry.
- Implemented and tested trajectory-F top-20% and bottom-80% partition
  controls; both 10,240-sample runs completed.
- Implemented token-F partitioning, deterministic random-token-drop control,
  affine/curriculum trajectory-F controls, and projection-mass grouped losses.
- Implemented pass@k generation with raw token/output preservation and fixed
  random masks per sample.
- Implemented merged-model evaluation with raw prediction sidecars and the
  canonical parser-first MathVista/MathVerse strict fallback protocols.
- Completed the fixed-teacher LCOT-1K divergence sweep over 11 r010-only
  checkpoints. The reproducible scorer stores exact FP32 JSD and both KL
  directions for r010 to r011/r012/r015 on checkpoint-specific rollouts.
- Added explicit projection-fraction grouped loss controls, bottom-F token
  dropping, KL-floor filtering, and a deterministic random-token-drop control.

### Current launchers

- Bottom-20% F grouped loss with lambda 0.30, r010 to r011 intervention, and
  no KL eligibility floor.
- Deterministic random-drop-10% response-token control matched to the r010 to
  r011 probe configuration.
- Five-benchmark evaluation of completed r010 ablations.

The repository publishes executable configurations, not cluster job state.
Checkpoints, logs, and evaluation results remain outside Git.

### Not completed

- Fresh r010-only OPSD training over all 20,000 examples. The resumable
  two-stage configuration is ready, but this run has not been executed.
- Complete five-benchmark results for every r010 ablation.
- Selection of a final reweighting equation or new OPSD objective.
- Multi-seed confirmation and statistical significance for a final method.
- A paper claim that projection fraction causally identifies the optimal
  training examples or tokens. Existing evidence is diagnostic and ablation
  evidence only.

## Fixed training contract

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Pruner | native VisionZip |
| Student retention | `0.10` only |
| Teacher | full-token EMA teacher, decay `0.9999` |
| Teacher ground-truth access | disabled |
| OPSD direction | `KL(q_full || p_r010)` |
| Trainable scope | language-decoder LoRA only |
| LoRA | rank 16, alpha 32, dropout 0 |
| Learning rate | fixed `2e-5` |
| Effective batch size | 32 |
| Rollout | greedy, KV cache, at most 512 tokens |
| Data order | deterministic, no shuffle |

The canonical training directory is:

```text
experiments/llm_only/opsd_random_r010_only_dropout0_20260815/
```

### 10,240-sample baseline

```bash
python experiments/llm_only/opsd_random_r010_only_dropout0_20260815/validate_config.py \
  --config experiments/llm_only/opsd_random_r010_only_dropout0_20260815/configs/train_10240.yaml \
  --output /tmp/r010_preflight.json --overwrite

sbatch experiments/llm_only/opsd_random_r010_only_dropout0_20260815/slurm/train_4l40s_20h.sbatch
```

### Resumable 20K run

Both stages declare `max_steps: 20000`, so their checkpoint contracts are
identical. Stage 2 restores LoRA, optimizer, EMA teacher, rank-specific RNG,
and the ordered data cursor from step 10,240 in the same output directory.

```bash
stage1=$(sbatch --parsable \
  experiments/llm_only/opsd_random_r010_only_dropout0_20260815/slurm/train_20000_stage1_4l40s_24h.sbatch)

sbatch --dependency=afterok:${stage1} \
  experiments/llm_only/opsd_random_r010_only_dropout0_20260815/slurm/train_20000_stage2_4l40s_24h.sbatch
```

## Projection fraction

For the full-token teacher `q`, the r010 student `p_b`, and an expanded native
VisionZip branch `p_b+` scored on the same student-generated prefix:

```text
A = JSD(q, p_b)
B = JSD(p_b, p_b+)
C = JSD(q, p_b+)
P = (A + B - C) / 2
F = P / A
```

For grouped trajectory analysis, sums are taken over valid generated tokens
before the division. For token partition controls, `F_t = P_t / A_t` is
computed per valid generated token with the configured numerical guard. Probe
branches and all selection/weight signals are detached; the optimized OPSD
loss remains differentiable `KL(q_full || p_r010)`.

Core implementation:

- `visionzip_aokvqa/native_budget_weighting.py`
- `visionzip_aokvqa/trajectory_weighting.py`
- `visionzip_aokvqa/losses.py`
- `visionzip_aokvqa/train.py`
- `tests/test_f_partition_weighting.py`
- `tests/test_projection_mass_grouped_weighting.py`
- `tests/test_token_kl_floor_filter.py`
- `tests/test_token_outlier_exclusion.py`

Current r010 ablations:

- `experiments/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818/`
- `experiments/llm_only/opsd_r010_f_trajectory_partition_delta002_dropout0_20260817/`
- `experiments/llm_only/opsd_r010_f_token_partition_delta002_dropout0_20260818/`
- `experiments/llm_only/opsd_r010_projection_mass_va_group_dropout0_20260818/`
- `experiments/llm_only/opsd_r010_f_bottom20_l030_d001_nofloor_dropout0_20260824/`
- `experiments/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824/`

## Fixed-teacher budget diagnostics

The canonical r010-only LCOT-1K scorer generates a fresh greedy rollout for
each checkpoint, then replays the same text under r010, expanded native
VisionZip budgets, and the adapter-disabled full-token base teacher:

```text
analysis/r010_only_lcot1k_fixed_teacher_deltas_20260823/
```

The smaller RandomPruner control uses the first 200 holdout samples, steps
0/1024/2048/3072, and a deterministic nested r010 to r011 random ranking:

```text
analysis/random_pruner_original_opsd_lcot200_fixed_teacher_d01_20260824/
```

Generated rollouts, token metrics, and plots are intentionally ignored by
Git; both directories contain their runner, validator, analysis, and Slurm
entry points.

## Five-benchmark evaluation

The clean launcher supports MME, MMStar, MathVista MINI, MathVerse MINI Vision
Only, and MMMU-Pro 4-choice at `noprune`, `r010`, `r020`, and `r030`. It uses a
merged BF16 model and always saves the pre-parser raw prediction sidecar.

```bash
bash experiments/llm_only/opsd_random_r010_only_dropout0_20260815/eval_five_benchmarks/merge_model.sh
bash experiments/llm_only/opsd_random_r010_only_dropout0_20260815/eval_five_benchmarks/submit_all.sh
```

Set `CHECKPOINT_STEP=20000` after the 20K run finishes. MathVista and
MathVerse paper scores must follow `scripts/eval/MATHVISTA_STANDARD.md` and
`scripts/eval/MATHVERSE_STANDARD.md`; changing the judge consumes saved raw
outputs and does not rerun inference.

## pass@k

The pass@k path writes raw outputs and token IDs before any parser or judge.
For random pruning, one deterministic mask is reused across all rollouts of a
sample.

Core files:

- `hypothesis_validate/scripts/run_raw_rollout_experiment.py`
- `hypothesis_validate/scripts/run_random_fixed_mask_rollout_experiment.py`
- `hypothesis_validate/scripts/judge_raw_rollout_experiment_qwen36_27b.py`
- `hypothesis_validate/scripts/validate_raw_rollout_sweep.py`

## Validation

```bash
pytest -q \
  experiments/llm_only/opsd_random_r010_only_dropout0_20260815/tests/test_r010_20k_resume_contract.py \
  tests/test_f_partition_weighting.py \
  tests/test_projection_mass_grouped_weighting.py \
  tests/test_token_kl_floor_filter.py \
  tests/test_token_outlier_exclusion.py \
  tests/test_trajectory_weighting.py \
  analysis/r010_only_lcot1k_fixed_teacher_deltas_20260823/tests/test_metrics.py \
  tests/test_mathvista_strict_gt_postprocess.py \
  tests/test_mathverse_strict_gt_postprocess.py \
  hypothesis_validate/tests/test_random_fixed_mask_rollout.py
```

GPU smoke tests require the cluster environment supplying the official
VisionZip Qwen2.5-VL implementation and pinned clean-Armen Transformers tree.

The exact Killarney L40S package versions, patched source paths, model/data
hashes, and evaluator runtime are recorded in
`docs/reproducibility/CORE_TRAIN_EVAL_ENVIRONMENT_20260824.md`.
The two distinct dirty VLMEvalKit/Transformers trees used for training and
evaluation can be reconstructed and hash-verified from
`third_party_patches/vlmevalkit_51682/`.

## Claim boundary

Completed engineering and completed ablations are reproducible from this
tree, but no final projection-based method is claimed yet. A method will be
promoted only after it beats original r010 OPSD under matched data, seeds,
trainable scope, optimizer steps, merged evaluation, and all five benchmark
post-processing standards.
