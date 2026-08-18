# Random-drop-20% replacement

- Cancelled job: `4863104` (`token_top20`, never started)
- Replacement job: `4876473` (`token_random_drop20`)
- Account: `aip-gigor`
- Resources: 4 x L40S, 20 hours
- Training ratio: VisionZip r010
- Diagnostic probe: native VisionZip r010 to r012
- Eligible response tokens: valid generated tokens with OPSD KL >= `1e-5`
- Selection: deterministically drop `ceil(0.2 * eligible_count)` tokens using
  SHA256 scores keyed by `seed:sample_id:rollout_seed:token_index`
- Optimization: mean forward OPSD KL over the remaining eligible tokens
- Random-drop seed: `42`
- All model, data-order, rollout, EMA-teacher, LoRA, optimizer, and checkpoint
  settings match the token-F partition runs.

Validation completed before submission:

- Python compilation passed.
- Config fail-closed validation passed.
- 23 focused unit tests passed.
