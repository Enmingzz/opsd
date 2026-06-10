# Official Qwen2.5-VL VisionZip VLMEvalKit Local Changes

These changes live outside this repository under
`/project/6101803/enmingzz/vlm/official_thinking_in_space/third_party`.
They are required for the pure VisionZip baseline to use the official
Qwen2.5-VL VisionZip implementation instead of `opsd.visionzip_aokvqa`.

## Files Modified

### VLMEvalKit wrapper

`VLMEvalKit/vlmeval/vlm/opsd_qwen25vl.py`

- Added the official VisionZip Qwen2.5-VL directory to `sys.path`.
- Imported `bootstrap_qwen25` from `opsd.visionzip_aokvqa.qwen_wrapper`.
- Added `model_mode="official_visionzip_base"`.
- In official mode:
  - call `bootstrap_qwen25()` before importing `qwen2_5vl_visionzip`
  - load `Qwen2_5_VLForConditionalGeneration` from official VisionZip
  - set `visionzip_contextual_ratio = 0.05`
  - set `visionzip_dominant_ratio = max(retention_ratio - 0.05, 0.0)`
  - call normal `model.generate(**pixel_values...)`; do not call our
    `generate_pruned`

### VLMEvalKit config

`VLMEvalKit/vlmeval/config.py`

- The four direct VisionZip baselines use `model_mode="official_visionzip_base"`:
  - `opsd_qwen25vl_visionzip_base_r005`
  - `opsd_qwen25vl_visionzip_base_r010`
  - `opsd_qwen25vl_visionzip_base_r020`
  - `opsd_qwen25vl_visionzip_base_r030`
- The four MMStar reasoning VisionZip baselines also use
  `model_mode="official_visionzip_base"`.

### Official VisionZip Qwen2.5-VL implementation

`VisionZip/Qwen2_5_VL/qwen2_5vl_visionzip.py`

- Parameterized the official hard-coded ratio lines:
  - default `visionzip_dominant_ratio = 0.65`
  - default `visionzip_contextual_ratio = 0.05`
- The default behavior remains official 70% retention: 65% dominant + 5%
  contextual.
- Our 5%, 10%, 20%, 30% baselines set dominant ratio to
  `retention_ratio - 0.05` while keeping contextual ratio at `0.05`.

## Failure Mode Fixed

The previous attempt imported official `qwen2_5vl_visionzip.py` after VLMEvalKit
had already loaded an older local transformers package. That produced:

```text
No module named 'transformers.models.qwen2_5_vl'
```

VLMEvalKit still wrote `status=done`, but metrics were empty. The fixed script
now runs an import preflight and fails if any dataset status contains
`error_message` or missing metrics.

