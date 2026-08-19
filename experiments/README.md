# Experiment Scope Layout

Training artifacts are separated by the parameter scope updated by LoRA.

- `vision_encoder_plus_llm/`: compatibility archive for all earlier A-OKVQA
  and LLaVA-CoT experiments. Their adapters updated both the visual encoder
  MLPs and the language decoder.
- `llm_only/`: clean root for new experiments whose trainable adapter tensors
  are restricted to the language decoder.

Do not compare or combine checkpoints across these roots without explicitly
reporting the difference in trainable parameter scope.
