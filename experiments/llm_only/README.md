# LLM-Only Experiments

This is the clean working root for all new language-decoder-only experiments.
No legacy adapter or merged model from the vision-encoder-plus-LLM experiments
may be used as initialization here.

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

- `first5_sft_then_opsd/`: first mechanism experiment, currently containing
  the first-five-layer SFT stage.
- `random_sft_opsd/`: full-decoder SFT and OPSD random-retention controls.
- `../scripts/verify_lora_scope.py`: shared fresh-model and saved-adapter scope
  verifier for every experiment in this root.
- `outputs`: link to the clean scratch output root. Formal `checkpoints` and
  `eval` start empty; `smoke` contains the verified first-five-layer preflight
  and one-step adapter.

The existing evaluation implementation is shared code and is not copied into
this directory. Results written by new evaluations must still use this root.

## Current Training Defaults

For new training runs created on or after 2026-08-03, use LoRA dropout `0.0`
and a fixed learning rate of `2e-5` with no scheduler unless an explicitly
named ablation requires otherwise. Record any exception in that experiment's
README and config. Legacy configs and checkpoints remain unchanged.
