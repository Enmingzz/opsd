# Original OPSD Dropout-0 Step-20000 Evaluation

This campaign evaluates the state-preserving 20k continuation of the original
LLM-only OPSD run.

## Artifact

- Adapter: `/scratch/enmingzz/outputs/llm_only/original_opsd_dropout0_exact_resume_to20k_20260803/run/resume_checkpoints/step_020000`
- Merged model: `/scratch/enmingzz/outputs/llm_only/merged_models/original_opsd_dropout0_20k_20260804/original_opsd_dropout0_step20000_qwen25vl7b_merged_bf16`
- Scope: language decoder LoRA only
- LoRA dropout: 0.0
- Evaluation: clean-Armen commit `51682a6baab948d3dbb4b867a3eab178504ac3f5`
- Prompt: reasoning mode
- Pixels: `min=1280*28*28`, `max=4096*28*28`
- Decoding: greedy, temperature 0, maximum 2048 new tokens

## Benchmarks and post-processing

- MME: official Yes/No extraction, with Qwen fallback for unresolved outputs.
- MMStar: official MCQ extraction, with the established Qwen fallback.
- MathVista MINI: deterministic extraction, with Qwen fallback.
- MathVerse MINI Vision Only: Qwen extraction and grading.
- MMMU-Pro 4c: official score plus an additional all-row Qwen answer-extraction score.

The MMMU-Pro Qwen extractor is not given the ground-truth answer. It extracts
the option selected by the model, after which correctness is computed by exact
comparison to the benchmark answer. When a complete final `<answer>` tag exists,
only that model-produced span and the option map are shown to Qwen so an
incorrect rationale cannot make the extractor override the model's explicit
selection. Untagged responses are judged from the complete raw response.
Official and Qwen-assisted scores are kept separately.

All runs preserve raw model output sidecars.

## Validation and submission

- Python compilation and shell syntax checks passed.
- Artifact, merged-checkpoint provenance, ratio-map, and scope preflights passed
  for all 20 benchmark/ratio cases.
- A mixed 86-row MMMU-Pro smoke covered complete answer tags, untagged boxed
  answers, truncated outputs, and repetitive outputs. Qwen extracted 83 rows;
  the three `UNKNOWN` cases had no selected answer.
- A real merged-checkpoint generation smoke passed on MMStar sample 0 at r020.
  The runtime reported native VisionZip backend ratio 0.85, KV cache enabled,
  2048 maximum new tokens, and a complete tagged reasoning response.
- Submitted on 2026-08-04 under `aip-btaati` as jobs `4597222` through
  `4597241`. The exact case mapping is in `submission.tsv`.
