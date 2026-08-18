# SFT20K merged reasoning evaluation

This campaign evaluates the audited LLM-only SFT checkpoint trained on the
ordered, decontaminated LLaVA-CoT 20K dataset.

Frozen settings:

- checkpoint: step 20,000, merged into Qwen2.5-VL-7B-Instruct;
- load form: merged full checkpoint with an empty adapter path;
- prompt: reasoning mode;
- pruner: native VisionZip;
- ratios: no-prune, 10%, 20%, and 30% retention;
- image budget: 1,280 to 4,096 Qwen visual tokens;
- generation: greedy, 2,048 maximum new tokens, KV cache enabled;
- evaluation environment: Armen VLMEvalKit commit `51682a6`;
- raw model predictions are retained before parsing;
- Qwen2.5-7B post-processing is used where the established campaign requires
  semantic extraction or judging.

Benchmarks are MME, MMStar, MMMU-Pro 4-choice, MathVista MINI, and
MathVerse MINI Vision-Only. Jobs are independent four-L40S jobs on
`aip-gigor`; no Slurm array is used.
