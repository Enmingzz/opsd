# MathVerse Vision-Only Scoring Standard

Canonical protocol: `mathverse-vo-strict-gt-v1.0`.

For every saved `MathVerse_MINI_Vision_Only` prediction:

1. Use the answer saved by the clean-Armen VLMEvalKit generation parser. If a
   fallback value still contains a complete `<answer>...</answer>` span,
   extract that span using the same concatenate-all-matches behavior as the
   Qwen wrapper.
2. Reload `question_for_eval`, question type, and ground truth by `index` from
   the canonical registered TSV. Never trust an XLSX answer cell because a
   leading `=` can be interpreted as a spreadsheet formula.
3. Accept only a trimmed exact match without a judge.
4. Send every non-exact candidate to Qwen3.6-27B with the canonical question,
   answer choices, ground truth, and parser candidate.
5. The judge must use temperature 0, thinking disabled, and constrained binary
   output over `CORRECT` and `INCORRECT`. It may recognize equivalent answer
   forms but may not repair missing, ambiguous, contradictory, or wrong model
   answers using the reference.
6. Preserve the source workbook, parser candidate, complete judge prompt, raw
   judge output, final label, hashes, and protocol version.

This is a final-answer accuracy protocol. It is not MathVerse CoT-E and does
not score the quality of the generated rationale.

CLI:

```bash
python scripts/eval/postprocess_mathverse_strict_gt.py \
  --result-file /path/to/Qwen_MathVerse_MINI_Vision_Only.xlsx \
  --output-dir /path/to/strict_output \
  --method METHOD \
  --ratio r020 \
  --base-url http://127.0.0.1:8000/v1
```
