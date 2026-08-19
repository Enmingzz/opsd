# OPSD with 10% VisionZip retention only

This is the canonical active OPSD training path. Every training sample uses
native VisionZip at `0.10` retention; no multi-ratio or progressive schedule
is used.

Key invariants:

- base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- trainable scope: language-decoder LoRA only
- LoRA rank 16, alpha 32, dropout 0
- full-token EMA teacher, decay `0.9999`, no ground-truth access
- fixed learning rate `2e-5`
- effective batch size 32
- greedy rollout with KV cache and up to 512 new tokens
- ordered, decontaminated LLaVA-CoT data (10,240 completed; 20K staged)
- snapshots every 256 samples; resumable checkpoints every 1,024

The paired-sampling namespace is intentionally identical to the other
ratio-set ablations so sample order and rollout seeds remain paired.

## Status

- The 10,240-sample run completed and passed the training, checkpoint, and
  LLM-only adapter-scope audits.
- The fresh 20K two-stage configuration is validated but has not been run.
- Five-benchmark evaluation is available; the full ablation table is not yet
  complete.

## 10,240-sample run

```bash
python experiments/llm_only/opsd_random_r010_only_dropout0_20260815/validate_config.py --overwrite
bash experiments/llm_only/opsd_random_r010_only_dropout0_20260815/run_smoke_1gpu.sh
sbatch experiments/llm_only/opsd_random_r010_only_dropout0_20260815/slurm/train_4l40s_20h.sbatch
```

## Fresh 20K run

Both stages use one checkpoint contract and one output directory. Stage 2
restores the full trainer state saved by stage 1.

```bash
stage1=$(sbatch --parsable \
  experiments/llm_only/opsd_random_r010_only_dropout0_20260815/slurm/train_20000_stage1_4l40s_24h.sbatch)
sbatch --dependency=afterok:${stage1} \
  experiments/llm_only/opsd_random_r010_only_dropout0_20260815/slurm/train_20000_stage2_4l40s_24h.sbatch
```

## Evaluation

```bash
bash experiments/llm_only/opsd_random_r010_only_dropout0_20260815/eval_five_benchmarks/merge_model.sh
bash experiments/llm_only/opsd_random_r010_only_dropout0_20260815/eval_five_benchmarks/submit_all.sh
```

See `eval_five_benchmarks/README.md` for the 20K checkpoint switch and strict
MathVista/MathVerse post-processing requirements.
