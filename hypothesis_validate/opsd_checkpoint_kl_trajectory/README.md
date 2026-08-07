# OPSD LLM-only Checkpoint KL Trajectory

This diagnostic follows the decontaminated LLaVA-CoT 10K, LLM-only OPSD run
through training. For every saved student checkpoint, it:

1. generates one greedy reasoning rollout with VisionZip r020;
2. teacher-forces that exact fixed response prefix under r020 and r025;
3. scores the same prefix with the fixed, adapter-disabled
   `Qwen/Qwen2.5-VL-7B-Instruct` full-token model;
4. computes exact full-vocabulary token-level divergences.

The three primary metrics are:

- `KL(p_student_r025 || p_student_r020)`;
- `KL(p_base_full || p_student_r020)`;
- `KL(p_base_full || p_student_r025)`.

The fixed base model is the reference for every checkpoint. The saved EMA
teacher is intentionally not used. A symmetric r020/r025 JS divergence and a
nested-r025 add-back control are also retained for diagnosis.

Intermediate checkpoints are loaded as their original PEFT LoRA adapters so
the same in-memory model can score the student and then disable the adapter for
the fixed base teacher. The clean Armen VisionZip backend floors token counts
after floating-point ratio conversion; the worker reads the actual retained
count from the backend instead of assuming an idealized percentage. Official
r025 recomputes its contextual merge centers, while the nested-r025 control
keeps every r020 token identity and adds ranked tokens to the same realized
budget.

As a load-form sanity check, the final PEFT adapter was compared with its
independently merged BF16 model on sample 812 at r020. Greedy generation was
token-for-token identical (70/70 tokens), and shared-prefix top-1 agreement was
100%. The mean JS divergence was `2.4077e-4`, consistent with small BF16 merge
rounding rather than a behavioral loading mismatch. The complete check is in
`outputs/merged_equivalence_final_sample812.json`.

The current primary run uses the manually audited MMStar metric-specific
No-OCR Clean-100 cohort. It preserves all four official answer choices and the
clean-Armen reasoning prompt, while excluding OCR, charts, tables, documents,
tiny-detail reading, duplicate images, and visually ambiguous labels. The
frozen sample file is:

`../manual_review/mmstar_metric_noocr_clean100_seed42/samples.jsonl`

Its SHA256 is
`2490bf326f6b21586423559b0178513944d503d5649d0e179fd291425823cc3f`.
Generation uses `max_new_tokens=1024`. This remains a mechanism diagnostic,
not an official MMStar accuracy evaluation.

`config.json` and the earlier 128-token outputs are retained only as legacy
diagnostics. Their high truncation rate makes them unsuitable for the final
checkpoint-trajectory claim.

## Direct run

```bash
source /project/6101803/enmingzz/env/vsi-official.sh
python hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/run_checkpoint.py \
  --config hypothesis_validate/opsd_checkpoint_kl_trajectory/config_max1024.json \
  --checkpoint-label step_0 \
  --adapter-path __BASE__
```

## Slurm sweep

```bash
bash hypothesis_validate/opsd_checkpoint_kl_trajectory/slurm/submit_all_max1024_30m.sh
```

## Aggregate

```bash
python hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/aggregate.py \
  --config hypothesis_validate/opsd_checkpoint_kl_trajectory/config_max1024.json
```

## Matched SFT trajectory

The matching LLM-only SFT run uses the same base initialization, decontaminated
10K training set, LoRA scope, learning rate, effective batch size, uniform
r010/r020/r030/r040 training schedule, checkpoint steps, Clean-100 diagnostic
cohort, and evaluation settings. Only the training objective differs.

```bash
bash hypothesis_validate/opsd_checkpoint_kl_trajectory/slurm/submit_all_sft_max1024_30m.sh

source /project/6101803/enmingzz/env/vsi-official.sh
python hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/aggregate.py \
  --config hypothesis_validate/opsd_checkpoint_kl_trajectory/config_sft_max1024.json
python hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/compare_sft_opsd.py
```

The comparison uses ordinary sample-balanced means as its primary aggregation.
Within-sample token trimming is retained only as an optional diagnostic.

## Four-pair budget-gap sweep

The expanded sweep evaluates four low/high VisionZip retention pairs:

- 15% to 17.5%;
- 15% to 20%;
- 17.5% to 20%;
- 20% to 22.5%.

Each pair uses all 11 OPSD checkpoints and all 11 SFT checkpoints, for 88
independent one-L40S jobs and 8,800 sample-checkpoint records. The low-retention
checkpoint generates the greedy prefix; low retention, official high retention,
nested high retention, and the fixed full-token base teacher all score that same
prefix. Legacy JSON metric keys containing `r020` and `r025` are retained for
backward compatibility, but every record, plot, table, and report stores and
displays the actual configured pair.

Create or refresh configs and submit:

```bash
python hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/create_ratio_pair_configs.py \
  --overwrite
bash hypothesis_validate/opsd_checkpoint_kl_trajectory/slurm/submit_ratio_pairs_88_jobs.sh
```

After all jobs finish, aggregate and audit:

```bash
bash hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/finalize_ratio_pair_sweep.sh
python hypothesis_validate/opsd_checkpoint_kl_trajectory/scripts/validate_ratio_pair_sweep.py
```

The combined report and figures are under:

`outputs/ratio_pairs_mmstar_clean100_max1024/analysis/`

The clean evaluation environment does not include a pandas Parquet engine, so
the per-token tables use lossless compressed CSV (`per_token_metrics.csv.gz`)
and record that fallback in `per_token_metrics_format.json`. Aggregate CSVs and
all numerical results are unchanged by this storage choice.
