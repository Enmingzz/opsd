# r010 F-weighting intervention ablation (`+0.02`)

This campaign retrains the existing r010 affine-F and intermediate-curriculum
OPSD variants with a smaller native VisionZip probe. It is a controlled
ablation of the probe intervention only:

- student rollout and OPSD branch: `b = 0.10`;
- native VisionZip probe: `b_plus = 0.12` (`budget_delta = 0.02`);
- model: `Qwen/Qwen2.5-VL-7B-Instruct`;
- trainable scope: LLM-only LoRA, 392 tensors / 40,370,176 parameters;
- LoRA dropout: 0;
- full-token EMA teacher, decay `0.9999`;
- fixed learning rate `2e-5`;
- 10,240 ordered trajectories, effective batch size 32;
- identical dataset order, sample IDs, rollout seed, and initialization.

The two unchanged weighting rules are:

- Affine: `w = 1 + (clip(F, 0, 1) - 0.20)`.
- Curriculum: `w = 4 * clip(F, 0, 1) * (1 - clip(F, 0, 1))`.

The affine center is `0.20`, selected before the full run after the matched
`+0.02` smoke showed mean `F` near `0.228`. This keeps the expected affine
loss scale near one instead of systematically downweighting the complete run.
Neither method uses batch renormalization.

## Commands

```bash
# Current single-GPU allocation
bash run_variant.sh global_f_affine smoke
bash run_variant.sh global_f_curriculum smoke

# Four-L40S training
sbatch --job-name=faff-d02 --export=ALL,VARIANT=global_f_affine \
  slurm/train_4l40s_gigor.sbatch
sbatch --job-name=fcurr-d02 --export=ALL,VARIANT=global_f_curriculum \
  slurm/train_4l40s_btaati.sbatch
```

Outputs are isolated under:

`/scratch/enmingzz/outputs/llm_only/opsd_r010_f_delta002_ablation_dropout0_20260818/`
