# RandomPruner OPSD LCOT-200 fixed-teacher F trend

This diagnostic measures the 1-percentage-point budget intervention over the
first 200 samples of the decontaminated LCOT-1k holdout set for the first four
checkpoints of the matched RandomPruner original-OPSD run.

## Protocol

- checkpoints: `0`, `1024`, `2048`, `3072`
- student: active checkpoint LoRA, native RandomPruner at `r010`
- intervention: the same deterministic random ranking at `r011`; therefore
  the `r010` retained-token set must be a subset of the `r011` set
- prefix: each checkpoint generates its own greedy `r010` rollout; that exact
  token sequence is replayed for all three scoring branches at that checkpoint
- teacher: adapter-disabled Qwen2.5-VL-7B base model with full visual tokens
- divergence: exact full-vocabulary FP32 JSD at every generated-token position

For token `t`:

```text
A_t = JSD(q_fixed_full, p_r010)
B_t = JSD(p_r010, p_r011)
C_t = JSD(q_fixed_full, p_r011)
P_t = (A_t + B_t - C_t) / 2
```

The main reported statistic is `F_pooled = sum(P_t) / sum(A_t)`. We also
report `mean_i[sum_t(P_it) / sum_t(A_it)]` with sample-bootstrap confidence
intervals, matching the existing LCOT-1k analysis convention.

## Run

```bash
bash run_one.sh 0
sbatch --account=aip-gigor --export=ALL,STEP=1024 slurm/run_one.sbatch
python analyze.py
```
