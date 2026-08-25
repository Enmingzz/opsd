# r010 OPSD: random drop 10% response-loss tokens

This is the deterministic random-token control matched to the existing
`drop bottom 10% F` ablation. The student rollout and OPSD loss use native
VisionZip at r010, while a no-grad native r011 pass uses the same generated
text prefix. For each trajectory, exactly `ceil(10% * valid_tokens)` valid
assistant-response positions are selected by a deterministic SHA256 ordering
and removed from the OPSD objective.

The training loss is:

```text
loss = mean(KL(q_full || p_r010) over non-dropped valid response tokens)
```

There is no KL eligibility floor, group coefficient, token-weight
normalization, or loss-mass restoration. Therefore this run differs from the
bottom-10% F ablation only in which 10% of response tokens are removed.

The reused core implementation has the legacy internal mode name
`token_random_drop20`; the configured `top_fraction: 0.10` is authoritative,
and strict smoke/final audits verify the actual dropped count.

```bash
bash experiments/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824/run_smoke_1gpu.sh
sbatch --account=aip-btaati \
  experiments/llm_only/opsd_r010_random_drop10_tokens_delta001_dropout0_20260824/slurm/train_4l40s_18h.sbatch
```
