# Original OPSD Dropout-0 Exact Resume to 20k

This experiment extends the original LLM-only OPSD `dropout=0` trajectory
from global sample step 9,984 to global sample step 20,000. It is an exact
stateful resume over an ordered 20k dataset, not a fresh additional-10k stage.

## Exact continuation contract

- Parent checkpoint: `step_009984` from the original dropout-0 OPSD run.
- Student LoRA, EMA teacher, AdamW moments/step, and all four rank RNG states
  are inherited byte-for-byte.
- Parent optimizer step: 312; expected final optimizer step: 625.
- Data cursor: global dataset index 9,984.
- Data order: the exact seed-42 order consumed by the parent run first,
  followed by a deterministic seed-42 order of the new decontaminated 10k.
- Loader shuffle: disabled for this preordered resume file. The first 9,984
  loaded sample IDs are checked one-by-one against the four parent rank logs.
- Remaining data: the final 16 original rows followed by all 10,000 new rows.
- Global target step: 20,000 (10,016 resumed samples, 313 optimizer updates).
- Learning rate: fixed `2e-5`; weight decay: `0`; LoRA dropout: `0.0`.
- Effective batch size: 4 GPUs x microbatch 1 x accumulation 8 = 32.
- Trainable scope: 40,370,176 parameters in 392 language-decoder LoRA tensors.
- Frozen scope: vision encoder, visual merger/projector, VisionZip, and base weights.
- Objective: original unweighted forward OPSD KL with EMA decay `0.9999`.
- Rollout and ratio randomness continue from restored per-rank RNG state.

The bridge checkpoint in each new output directory hard-links the immutable
parent state files and rewrites only the trainer metadata needed to bind that
state to the ordered 20k dataset and new target step. Parent logs are copied
before appending, so the original run remains unchanged.

## Inputs

- Parent checkpoint:
  `/scratch/enmingzz/outputs/llm_only/native_budget_weighting_dropout0_pair_20260730/original_opsd_dropout0/resume_checkpoints/step_009984`
- Ordered 20k dataset:
  `data/openmmreasoner_llava_cot_train20k_ordered_decontam_v1_seed42/train20k_exact_resume_order_qwentok512_imgtok1152_seed42.jsonl`
- Dataset SHA256:
  `9f1cf9dfbff291ee0ce7f34a236820c7da907f096bdb54ec0165eed845f3516f`
- Training manifest:
  `data/openmmreasoner_llava_cot_train20k_ordered_decontam_v1_seed42/train20k_exact_resume_training_manifest.json`
- Ordering manifest:
  `data/openmmreasoner_llava_cot_train20k_ordered_decontam_v1_seed42/train20k_exact_resume_order_manifest.json`

## Outputs

- 32-sample exact-resume smoke:
  `/scratch/enmingzz/outputs/llm_only/original_opsd_dropout0_exact_resume_to20k_20260803/smoke32`
- Full exact resume:
  `/scratch/enmingzz/outputs/llm_only/original_opsd_dropout0_exact_resume_to20k_20260803/run`

## Current execution

- Exact-resume smoke job `4563216`: completed in 4m53s; post-training audit passed.
- Full job `4563217`: started 2026-08-03 17:40 EDT on 4 x L40S.
- Expected completion from measured and parent-run throughput: approximately
  2026-08-04 07:45 EDT.
- Full-run state at launch: global step 9,984, AdamW step 312.
- Expected final state: global step 20,000, AdamW step 625.

## Commands

Prepare or verify the exact-resume bridge:

```bash
source /project/6101803/enmingzz/env/vsi-official.sh
python experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/prepare_exact_resume.py \
  --config experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/configs/train_exact_resume_to20k.yaml \
  --output-dir /scratch/enmingzz/outputs/llm_only/original_opsd_dropout0_exact_resume_to20k_20260803/run \
  --check-only
```

Submit the audited smoke and dependent full run:

```bash
SMOKE_JOB=$(sbatch --parsable \
  experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/slurm/smoke_exact_resume_4l40s.sbatch)
sbatch --dependency=afterok:${SMOKE_JOB} \
  experiments/llm_only/original_opsd_dropout0_continuation_additional10k_20260803/slurm/train_exact_resume_4l40s.sbatch
```

The common launcher validates the bridge before training and audits the
completed state, log continuity, optimizer step, parameter scope, snapshots,
and final checkpoint afterward.

## Superseded attempt

The earlier warm-start design reset AdamW and trained a standalone additional
10k stage. Its pending Slurm job `4561811` was cancelled before allocation or
execution. Files under `manifests/` with `warm_start` in their status are kept
only as an audit trail and must not be used as evidence for this exact-resume
run.

The first exact-resume smoke attempt (`4562827`) exposed that the generic
JSONL loader reshuffled the entire physical 20k file. It and its dependent
full job (`4562828`) were cancelled before any full-run training. Their output
is retained under
`superseded_global_reshuffle_4562827/`. The corrected preordered dataset and
explicit `dataset.shuffle: false` prevent that global reshuffle.
