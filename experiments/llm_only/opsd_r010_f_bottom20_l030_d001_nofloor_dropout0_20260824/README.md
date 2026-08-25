# R010 F-bottom20 lambda-0.30 delta-0.01 no-floor OPSD

This run isolates removal of the F-ranking eligibility floor from the matched
`f_bottom20_l030_d001` experiment. All other training settings remain fixed:

- Qwen2.5-VL-7B-Instruct;
- LLM-only LoRA, dropout 0;
- VisionZip r010 student and native r011 probe;
- EMA full-token teacher;
- ordered decontaminated LLaVA-CoT first 10,240 examples;
- fixed learning rate 2e-5;
- 10,240 optimizer steps;
- effective batch size 32 on four L40S GPUs.

The only scientific change is `min_teacher_kl: 0.0`. The lowest 20% F tokens
among all valid generated tokens receive 30% of the group-level loss mass:

```text
loss = 0.30 * mean(KL_bottom20F) + 0.70 * mean(KL_complement)
```

Run preflight:

```bash
python experiments/llm_only/opsd_r010_f_bottom20_l030_d001_nofloor_dropout0_20260824/validate_config.py \
  --config experiments/llm_only/opsd_r010_f_bottom20_l030_d001_nofloor_dropout0_20260824/configs/train_10240.yaml \
  --expected-steps 10240
```

Submit training:

```bash
sbatch --account=aip-btaati \
  experiments/llm_only/opsd_r010_f_bottom20_l030_d001_nofloor_dropout0_20260824/slurm/train_4l40s_20h.sbatch
```

The run is resumable from the latest complete checkpoint under
`run/resume_checkpoints/` and saves lightweight evaluation snapshots every
256 steps.
