# R010-only LCOT-1K fixed-teacher delta sweep

This diagnostic evaluates the VisionZip r010-only OPSD trajectory across
checkpoints `0, 1024, ..., 10240` on the decontaminated LLaVA-CoT holdout set.

For every checkpoint and sample:

1. The current student checkpoint generates one greedy rollout at r010.
2. The generated token IDs are replayed unchanged under r010, r011, r012,
   r015, and the full-token fixed base teacher.
3. The student LoRA remains active for every pruned branch. It is disabled
   only for the full-token teacher branch, so the teacher is always the
   Qwen2.5-VL-7B step-0/base model.
4. Exact full-vocabulary FP32 divergences are saved for every generated-token
   position.

For each intervention `d` in `{0.01, 0.02, 0.05}`:

```text
A = pair(q_teacher_full, p_student_r010)
B = pair(p_student_r010, p_student_r010_plus_d)
C = pair(q_teacher_full, p_student_r010_plus_d)
```

Every pair stores JSD and both KL directions. This is nine scalar arrays per
intervention and 27 arrays per token in total. The duplicated A arrays are
intentional: the wide Parquet schema is self-contained for every intervention.
This is a strict superset of the requested 18 directional-KL values: the table
also keeps all nine JSD arrays, so later geometry definitions do not require a
new model forward or rollout.

The ordered pairs and KL directions are:

| Pair | First | Second | `forward_kl` | `reverse_kl` |
|---|---|---|---|---|
| A | fixed base teacher, full tokens | checkpoint student, r010 | KL(q || p10) | KL(p10 || q) |
| B | checkpoint student, r010 | checkpoint student, r010+d | KL(p10 || pplus) | KL(pplus || p10) |
| C | fixed base teacher, full tokens | checkpoint student, r010+d | KL(q || pplus) | KL(pplus || q) |

Outputs are resumable. A rollout is written atomically before scoring, so an
interrupted scoring job never needs to regenerate a completed rollout.

## Output layout

```text
outputs/
  step_000000/
    rollouts/samples/*.json
    scores/samples/*.json
    rollouts.jsonl
    scores.jsonl
    per_token_metrics.parquet
    manifest.json
```

`per_token_metrics.parquet` is the canonical analysis table. `rollouts.jsonl`
preserves the exact question, prompt, image path, generated text, generated
token IDs, and generation metadata.

After all checkpoint shards complete, run:

```bash
source /project/6101803/enmingzz/env/vsi-official.sh
python analysis/r010_only_lcot1k_fixed_teacher_deltas_20260823/consolidate_campaign.py
```

The validated campaign-level artifacts are written under
`outputs/consolidated/`, including one Parquet table across all 11 checkpoints
and the complete rollout/score JSONL files.
