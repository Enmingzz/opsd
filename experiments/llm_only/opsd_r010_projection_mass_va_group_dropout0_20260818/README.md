# R010 Projection-Mass Token Reweighting

This experiment starts from the same Qwen2.5-VL-7B initialization and uses
the same ordered 10,240 samples, rollout seeds, optimizer, EMA teacher, and
LLM-only LoRA scope as `opsd_random_r010_only_dropout0_20260815`.

For every r010 student rollout, a no-grad native VisionZip r012 branch scores
the same textual prefix. Per generated token:

```
A_t = JSD(q_full, p_10)
B_t = JSD(p_10, p_12)
C_t = JSD(q_full, p_12)
P_t = (A_t + B_t - C_t) / 2
```

Tokens are ranked by `max(P_t, 0)`. The top 10% form the high projection-mass
group. The raw grouped objective is:

```
0.30 * mean(KL_high) + 0.70 * mean(KL_low)
```

At a 10% high-token fraction this gives approximately 3.86 times as much
per-token weight to the high group, close to the 4x high/low token emphasis
of VA-OPD's top-20%, lambda=0.5 grouping. The grouped objective is optimized
directly; no KL-dependent loss-mass correction is applied. Consequently, its
scalar value may differ from vanilla OPSD, as it should for a genuine grouped
loss.

The auxiliary branch and JSD-derived group assignment are no-grad. The OPSD KL
inside both groups remains differentiable. Only valid generated assistant
tokens enter the loss.

## Commands

```bash
bash run_smoke_1gpu.sh
sbatch slurm/train_4l40s_btaati.sbatch

# VA-OPD-matched grouping ablation
bash run_smoke_top20_lambda05_1gpu.sh
sbatch slurm/train_top20_lambda05_4l40s_btaati.sbatch
```

Outputs are written under:

`/scratch/enmingzz/outputs/llm_only/opsd_r010_projection_mass_va_group_dropout0_20260818/`

The top-20%, lambda=0.5 variant is isolated at:

`/scratch/enmingzz/outputs/llm_only/opsd_r010_projection_mass_top20_lambda05_dropout0_20260818/`
