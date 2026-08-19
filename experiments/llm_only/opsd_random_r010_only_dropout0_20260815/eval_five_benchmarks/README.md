# r010-only merged evaluation

This launcher evaluates the r010-only LLM-LoRA OPSD checkpoint on MME,
MMStar, MathVista MINI, MathVerse MINI Vision Only, and MMMU-Pro 4-choice at
no pruning and 10%, 20%, and 30% retained visual-token budgets. Every run
uses the merged BF16 checkpoint, reasoning mode, greedy decoding, KV cache,
the pinned clean-Armen VLMEvalKit checkout, and raw-prediction preservation.

The default is the completed step-10240 checkpoint:

```bash
bash eval_five_benchmarks/merge_model.sh
bash eval_five_benchmarks/submit_all.sh
```

To evaluate the two-stage 20K run after it completes:

```bash
export CHECKPOINT_STEP=20000
bash eval_five_benchmarks/merge_model.sh
bash eval_five_benchmarks/submit_all.sh
```

Generation and scoring are separate artifacts. MME uses the strict Qwen
yes/no fallback, MMStar uses the integrated parser-first Qwen fallback, and
MMMU-Pro uses the official multiple-choice evaluator. MathVista and MathVerse
generation also writes the legacy score for diagnostics, but paper tables
must use the canonical parser-first strict protocols documented in
`scripts/eval/MATHVISTA_STANDARD.md` and
`scripts/eval/MATHVERSE_STANDARD.md`. Those strict postprocessors consume the
saved workbook and raw sidecar, so judge changes never require regeneration.

No result files or checkpoints are committed to GitHub.
