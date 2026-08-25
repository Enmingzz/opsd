# LLM-Only Experiments

This is the clean working root for all new language-decoder-only experiments.
No legacy adapter or merged model from the vision-encoder-plus-LLM experiments
may be used as initialization here.

The only active training launchers published in Git are the native VisionZip
`r010` OPSD baseline and its projection-fraction ablations. Historical
multi-ratio, progressive, and JSD-weighted launchers are deliberately excluded
from the maintained GitHub workflow. See `../../CORE_RESEARCH.md`.

## Scope Contract

Every new training configuration must:

1. start from the original base model or an adapter produced under this root;
2. set `training.lora_layers_pattern: layers`;
3. explicitly set `training.lora_layers_to_transform` to language decoder
   layer indices (Qwen2.5-VL-7B has layers 0 through 27);
4. save all checkpoints, evaluation outputs, and logs under
   `/scratch/enmingzz/outputs/llm_only/`;
5. verify every saved adapter has zero tensor names containing `.visual.`.

The machine-readable requirements are in `scope_contract.json`.

## Layout

- `opsd_random_r010_only_dropout0_20260815/`: canonical baseline, resumable
  20K configuration, and five-benchmark evaluator.
- `opsd_r010_f_delta002_ablation_dropout0_20260818/`: global trajectory-F
  affine and intermediate-curriculum controls.
- `opsd_r010_f_trajectory_partition_delta002_dropout0_20260817/`: hard
  trajectory top/bottom-F controls.
- `opsd_r010_f_token_partition_delta002_dropout0_20260818/`: token top/bottom-F
  and matched random-drop controls.
- `opsd_r010_projection_mass_va_group_dropout0_20260818/`: projection-mass
  grouped-loss controls.
- `opsd_r010_f_bottom20_l030_d001_nofloor_dropout0_20260824/`: bottom-20% F
  grouped-loss run with lambda 0.30, r010 to r011, and no KL ranking floor.
- `opsd_r010_random_drop10_tokens_delta001_dropout0_20260824/`: deterministic
  random-drop-10% response-token control matched to the r010 to r011 probe.
- `../scripts/verify_lora_scope.py`: shared fresh-model and saved-adapter scope
  verifier.

The existing evaluation implementation is shared code and is not copied into
this directory. Results written by new evaluations must still use this root.

## Current Training Defaults

For new training runs created on or after 2026-08-03, use LoRA dropout `0.0`
and a fixed learning rate of `2e-5` with no scheduler unless an explicitly
named ablation requires otherwise. Record any exception in that experiment's
README and config. Legacy configs and checkpoints remain unchanged.
