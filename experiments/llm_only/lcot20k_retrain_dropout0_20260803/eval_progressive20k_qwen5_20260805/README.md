# Progressive OPSD 20K clean-Armen evaluation

This campaign evaluates the merged LLM-only Progressive OPSD checkpoint at
step 20,000. It is isolated from all prior SFT, original OPSD, and legacy
evaluation outputs.

## Matrix

- Benchmarks: MME, MMStar, MathVista MINI, MathVerse MINI Vision Only, MMMU-Pro 4-option.
- Retention settings: no pruning, 10%, 20%, and 30%.
- Prompt: reasoning mode.
- Decoding: greedy, temperature 0, KV cache enabled, 2,048 generated tokens.
- Image range: 1,280 to 4,096 Qwen visual tokens.
- Model form: LoRA merged into Qwen/Qwen2.5-VL-7B-Instruct.
- Evaluation runtime: the clean-Armen-aligned `ckpt_eval_trainenv`.
- Postprocessing: the established Qwen-assisted benchmark pipeline; raw model
  predictions and validation sidecars are retained.

## Dependency chain

Training job `4599261` must finish and pass its final audit. A one-GPU merge
job then validates the step-20,000 adapter, merges it, records its SHA-256,
and writes immutable merge metadata. Twenty independent four-GPU evaluation
jobs run only after that merge succeeds.

Submit with:

```bash
bash submit_all.sh
```

Collect completed results with:

```bash
python collect_results.py
```

The score summary is written beside this README. Raw outputs live under:

`/scratch/enmingzz/outputs/llm_only/eval/progressive_opsd20k_dropout0_step20000_merged_qwen5_20260805`
