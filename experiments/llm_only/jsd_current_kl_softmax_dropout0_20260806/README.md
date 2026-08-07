# Softmax JSD/Current-KL Trajectory Reweighting

These two LLM-only OPSD runs test detached entropy-regularized trajectory
allocation at temperatures 0.05 and 0.1. They are exactly paired with the
direct-inverse and prior median-inverse runs.

For trajectory `i` in a synchronized effective batch of size `B=32`:

```text
s_i = mean_token_JSD(p_b, p_b_plus) / mean_token_KL(q_full || p_b)
probability_i = softmax_i(-s_i / T)
multiplier_i = B * probability_i
loss = (1 / B) * sum_i multiplier_i * mean_token_KL(q_full || p_b)
```

The weights are detached. Their probability sum is one and their mean
multiplier is one, so uniform sensitivities exactly recover vanilla OPSD.
Neither run uses `tau`.

## Paired settings

- Qwen2.5-VL-7B-Instruct
- LLM-only LoRA: 392 tensors / 40,370,176 trainable parameters
- LoRA dropout: 0
- fixed learning rate: 2e-5
- EMA teacher decay: 0.9999
- effective batch: 4 GPUs x 8 accumulation x 1 sample = 32
- identical ordered data, sample IDs, ratio seed, rollout seed, and paired
  sampling namespace
- native VisionZip ratios sampled from 0.10, 0.20, 0.30, 0.40
- relative probe budget: `b_plus = 1.25 * b`

## Validation

CPU formula tests, detached-gradient tests, four-rank DDP scaling tests, and
real-model four-trajectory smoke runs passed for both temperatures. Both
real-model runs had zero fixed-prefix probe/replay KL error.

| Temperature | Multiplier range | Weighted/unweighted KL |
|---:|---:|---:|
| 0.05 | 0.535–1.508 | 1.005 |
| 0.10 | 0.743–1.247 | 1.016 |

Smoke outputs:

```text
/scratch/enmingzz/outputs/llm_only/jsd_current_kl_softmax_dropout0_20260806/smoke_T005
/scratch/enmingzz/outputs/llm_only/jsd_current_kl_softmax_dropout0_20260806/smoke_T010
```

## Full runs

- `T=0.05`: Slurm job `4636680`
- `T=0.10`: active Slurm job `4636754`

Both request 4 L40S GPUs, 32 CPUs, 256 GB RAM, and 20 hours on `aip-btaati`.

The first `T=0.10` submission, `4636681`, was assigned to `kn002` and failed
before the shell script entered Python with `/usr/bin/env: bash: Too many
levels of symbolic links`. No output directory or checkpoint was created. The
launcher shebang was changed to `/bin/bash`, and the replacement job excludes
`kn002`.
