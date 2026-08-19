# Ordered LLaVA-CoT 20K

This artifact preserves the approved historical 10K as rows 1-10,000 and appends a
strictly disjoint additional 10K as rows 10,001-20,000.

## Training files

- Continue without replaying old data: `additional10k_decontam_qwentok512_imgtok1152_seed42.jsonl`
- Full ordered dataset: `train20k_old10k_then_additional10k_qwentok512_imgtok1152_seed42.jsonl`
- Continuation begins at one-based row `10,001` in the ordered dataset.

Use the standalone additional 10K file for a continuation run. This avoids relying on
sampler offset semantics and guarantees that the approved historical 10K is not replayed.

## Checks

- Valid `<think>...</think><answer>...</answer>` target for every row.
- Qwen target length <= 512 tokens.
- Recorded image-prompt length <= 1152 tokens.
- No sample/source/question/image-path identity overlap between old and new splits.
- No identity overlap with the strict 1K validation holdout.
- The old 10K prefix and new 10K suffix are preserved byte-for-byte.
- Independent 15-benchmark leakage audit passed with zero matches.
- Qwen processor check passed on 256 samples with zero token-count mismatches.

See `manifest.json` for hashes and `boundary_preview.md` for the ordering boundary.
