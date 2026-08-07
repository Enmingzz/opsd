# JSD trajectory reweighting, dropout 0

This experiment compares two trajectory-level OPSD weighting signals while
holding initialization, data order, sampled VisionZip ratios, rollout seeds,
optimizer settings, and LLM-only LoRA scope fixed.

The verified 20k exact-resume dataset is used with `shuffle: false`, and these
runs stop at sample 10,240. Its first 10,000 rows exactly match the complete
seed-42 shuffled sequence of the original 10k dropout-0 run. The final 240
rows come from the verified extension set and pass the same decontamination,
text-length, and image-token checks.

The trajectory contains step 0 plus full resumable checkpoints at steps
1,024 through 10,240, for 11 checkpoints in total. Lightweight LoRA
evaluation snapshots are also saved every 256 samples.

For a rollout sampled at budget `b`, native VisionZip is rerun at `1.25 * b`
on the same generated text prefix. The symmetric FP32 JSD between these two
student distributions is averaged over the generated trajectory.

The two sensitivity definitions are:

1. `jsd_over_current_kl`: `JSD(p_b, p_1.25b) / KL(q_full || p_b)`.
2. `jsd_over_step0_kl`: `JSD(p_b, p_1.25b) / K0[b]`, where `K0[b]` is a
   frozen ratio-specific checkpoint-0 calibration from the independent 1k
   LLaVA-CoT holdout probe.

Within each effective batch of 32 trajectories, lower sensitivity gets higher
detached weight:

```text
tau = median(sensitivity)
raw_i = (tau + eps) / (tau + sensitivity_i + eps)
w_i = raw_i / sum_j(raw_j)
```

Therefore `sum_i(w_i) = 1`. No KL-loss-mass correction is applied. A no-grad
probe first computes all 32 signals and caches rollout token IDs on CPU. The
differentiable pass then replays those exact prefixes one at a time, avoiding
32 simultaneous computation graphs while preserving exact batch-32
normalization.

Full outputs are written to:

- `/scratch/enmingzz/outputs/llm_only/jsd_trajectory_reweight_dropout0_20260806/jsd_over_current_kl`
- `/scratch/enmingzz/outputs/llm_only/jsd_trajectory_reweight_dropout0_20260806/jsd_over_step0_kl`
