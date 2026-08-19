# Canonical MathVista Evaluation Protocol

Protocol ID: `mathvista-strict-gt-v1.0`

Effective date: 2026-08-14

This is the default MathVista post-processing protocol for new OPSD results.
Scores produced by the older unconstrained Qwen2.5-7B answer extractor must be
labelled `legacy_qwen7_extraction` and must not be mixed with canonical scores
in one comparison table.

## Required inputs

- The original MathVista workbook from the pinned clean-Armen VLMEvalKit
  evaluation (`51682a6b`).
- The saved raw model response. Generation must not be rerun solely because
  post-processing changes.
- `question`, `answer`, `answer_option`, `choices`, `question_type`, and
  `answer_type` from the benchmark row.

The candidate passed to post-processing is the evaluator's `prediction`
field: a closed `<answer>...</answer>` span when one exists, otherwise the
visible raw response fallback. This preserves the model's explicit final
answer while retaining malformed or truncated responses for strict review.

## Decision order

1. Run the pinned deterministic MathVista parser.
2. If it resolves an answer, accept that parser result even when it is wrong.
   Do not let an LLM repair a deterministically parseable wrong answer.
3. Only unresolved responses enter the fallback judge.
4. The fallback judge receives the question, ground-truth answer, optional
   correct option letter, and candidate response. It does not receive the
   image.
5. The fallback judge outputs exactly `CORRECT` or `INCORRECT`.

The canonical fallback model is the local
`Qwen/Qwen3.6-27B-FP8` checkpoint, decoded greedily with thinking disabled.
The vLLM request uses constrained choice decoding over exactly `CORRECT` and
`INCORRECT`; free-form judge text is not accepted.
The system prompt explicitly forbids solving, completion, reference-based
repair, endpoint selection from ranges, and accepting unfinished or
contradictory responses.

## Safety rules

- A truncated or unfinished response without an explicit committed answer is
  incorrect.
- A range or multiple alternatives is incorrect when one scalar is required.
- A refusal or claim of insufficient information is incorrect unless that is
  itself the reference answer.
- Units, comma separators, and equivalent numeric formatting may be
  normalized.
- Correctness judging is answer-level only. It does not establish visual
  grounding or rationale correctness because the judge does not see the
  image.
- Every output must record the source workbook hash, prompt hash, judge model,
  per-row decision source, raw judge output, and Slurm job ID.

## Canonical command

```bash
python scripts/eval/postprocess_mathvista_strict_gt.py \
  --result-file /path/to/Qwen_MathVista_MINI.xlsx \
  --output-dir /path/to/canonical_postprocess \
  --method METHOD \
  --ratio r020 \
  --base-url http://127.0.0.1:PORT/v1
```

The output directory contains `judgments.jsonl`, the canonical workbook,
category-level score CSV, and `summary.json`. Existing generation and legacy
post-processing files are never overwritten.
