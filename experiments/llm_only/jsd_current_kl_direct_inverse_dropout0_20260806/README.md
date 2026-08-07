# Direct-Inverse JSD/Current-KL OPSD

This run is paired with
`jsd_trajectory_reweight_dropout0_20260806/jsd_over_current_kl` and changes
only the trajectory-weight transform.

For trajectory `i`:

```text
s_i = mean_token_JSD(p_b, p_b_plus) / mean_token_KL(q_full || p_b)
raw_weight_i = 1 / (s_i + 1e-8)
weight_i = raw_weight_i / sum_j(raw_weight_j)
loss = sum_i weight_i * mean_token_KL(q_full || p_b)
```

There is no median pivot (`tau`), temperature, clipping, or fitted ratio
coefficient. Weights are detached and normalized over the synchronized
effective batch of 32 trajectories. Uniform signals exactly recover the
ordinary OPSD batch mean.

## Pairing

- Qwen2.5-VL-7B-Instruct
- LLM-only LoRA: 392 tensors / 40,370,176 trainable parameters
- LoRA dropout: 0
- fixed learning rate: 2e-5
- EMA teacher decay: 0.9999
- effective batch: 4 GPUs x 8 accumulation x 1 sample = 32
- identical ordered data, sample IDs, ratio seed, rollout seed, and paired
  sampling namespace to the previous current-KL run
- native VisionZip ratios sampled from 0.10, 0.20, 0.30, 0.40
- native relative probe budget: `b_plus = 1.25 * b`

## Validation

The following completed before submission:

```bash
python experiments/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/tests/run_direct_inverse_tests.py
torchrun --standalone --nproc-per-node=4 \
  experiments/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/tests/distributed_weighting_smoke.py
python visionzip_aokvqa/train.py \
  --config experiments/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/configs/smoke4.yaml \
  --output_dir /scratch/enmingzz/outputs/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/smoke4
```

The real-model smoke completed four trajectories and one optimizer update.
Its fixed-prefix replay KL error was zero, probability weights summed to one,
`trajectory_batch_tau` was null, and the logged transform was
`direct_inverse`.

## Full Run

```bash
CONFIG_PATH=/project/6101803/enmingzz/opsd/experiments/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/configs/train_10240.yaml \
OUT_DIR=/scratch/enmingzz/outputs/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/train_10240 \
sbatch --export=ALL,CONFIG_PATH="$CONFIG_PATH",OUT_DIR="$OUT_DIR" \
  experiments/llm_only/jsd_current_kl_direct_inverse_dropout0_20260806/slurm/train_4l40s_24h.sbatch
```

Submitted as Slurm job `4636577` on `aip-btaati`. Its pending time limit was
subsequently reduced from 24 hours to 20 hours with `scontrol update`; the
submission script records the updated limit for reproducibility.
