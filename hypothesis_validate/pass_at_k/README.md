# pass@k Evaluation

This directory contains the launchers for the current raw-rollout pass@k
pipeline. Generated rollouts and judged outputs are intentionally excluded
from version control.

## Protocol

- model: `Qwen/Qwen2.5-VL-7B-Instruct`;
- pruners: native VisionZip or deterministic random pruning;
- retention ratios: 10%, 20%, 30%, 40%, and no pruning;
- cohort: 100 manually audited numeric/open-ended MMStar samples;
- stochastic rollouts: 64 per sample;
- decoding: temperature 0.7, top-p 0.95, top-k 50;
- generation ceiling: 1024 tokens;
- greedy decoding is run separately;
- raw text, raw token IDs, seeds, mask metadata, and generation settings are
  saved before parsing or judging.

For random pruning, every rollout of the same sample and ratio uses the same
fixed visual-token mask. Different samples receive deterministic masks derived
from their sample ID and the global seed.

## Core scripts

Raw VisionZip/no-prune generation:

```text
hypothesis_validate/scripts/run_raw_rollout_experiment.py
```

Random fixed-mask generation:

```text
hypothesis_validate/scripts/run_random_fixed_mask_rollout_experiment.py
```

Post-generation Qwen judging and aggregation:

```text
hypothesis_validate/scripts/judge_raw_rollout_experiment_qwen36_27b.py
hypothesis_validate/scripts/summarize_qwen36_judge_only_sweep.py
hypothesis_validate/scripts/validate_raw_rollout_sweep.py
```

Sample-selection code:

```text
hypothesis_validate/scripts/prepare_mmstar_numeric_clean100.py
hypothesis_validate/scripts/finalize_mmstar_metric_noocr_clean100.py
```

## Example

```bash
python hypothesis_validate/scripts/run_raw_rollout_experiment.py \
  --samples /path/to/mmstar_numeric_clean100.jsonl \
  --output-dir /path/to/pass64/r020 \
  --benchmark MMStar_OpenEnded \
  --decode-mode sample \
  --pruning visionzip \
  --retention-ratio 0.20 \
  --num-rollouts 64 \
  --rollout-batch-size 16 \
  --temperature 0.7 \
  --top-p 0.95 \
  --top-k 50 \
  --max-new-tokens 1024
```

Use `--decode-mode greedy` for the paired greedy run. Evaluation is a separate
stage: explicit `<answer>...</answer>` content is extracted when present, and
otherwise the preserved raw response is sent to the configured judge. This
separation allows parser or judge changes without rerunning model generation.

Standard unbiased pass@k is computed from `n=64` rollouts and `c` judged-valid
successes:

```text
pass@k = 1 - C(n-c, k) / C(n, k)
```

The reasoning-valid analysis counts a rollout as successful only when its
visual evidence and reasoning are judged coherent, not merely when its final
answer happens to match.
