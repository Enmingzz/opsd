# VisionZip Implementation And Evaluation Audit

Date: 2026-06-08

## Verdict

The pure VisionZip baseline path now uses the official Qwen2.5-VL VisionZip
implementation directly:

- official file:
  `/project/6101803/enmingzz/vlm/official_thinking_in_space/third_party/VisionZip/Qwen2_5_VL/qwen2_5vl_visionzip.py`
- VLMEvalKit model mode: `official_visionzip_base`
- registered model names:
  `opsd_qwen25vl_visionzip_base_r005`,
  `opsd_qwen25vl_visionzip_base_r010`,
  `opsd_qwen25vl_visionzip_base_r020`,
  `opsd_qwen25vl_visionzip_base_r030`

The official implementation had hard-coded ratios. I parameterized those two
ratio lines with default values unchanged (`0.65` dominant, `0.05` contextual)
so the same official implementation can run our 5%, 10%, 20%, and 30% settings.

The previous VisionZip baseline results under
`/project/6101803/enmingzz/outputs/visionzip_aokvqa_reasoning/eval_vlmevalkit/visionzip_base_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030`
were complete, but they used the old contextual split (`target * 1/14`). Treat
them as old-split sanity results, not final official VisionZip numbers.

## Fix Applied

- Updated `visionzip_aokvqa/visionzip.py` so default `contextual_fraction` is
  `0.05`.
- Updated contextual count to use original visual token count instead of target
  token count.
- Allowed 5% retention to have zero dominant tokens, matching the official
  Qwen2.5-VL VisionZip behavior.
- Updated the VLMEvalKit baseline registrations so pure VisionZip uses the
  official Qwen2.5-VL VisionZip model class instead of our local pruning wrapper.
- Parameterized the official Qwen2.5-VL VisionZip class by
  `visionzip_dominant_ratio` and `visionzip_contextual_ratio`, while preserving
  the official default behavior when those attributes are not set.

## Official Code Alignment

Official local reference:

- `/project/6101803/enmingzz/vlm/official_thinking_in_space/third_party/VisionZip/Qwen2_5_VL/qwen2_5vl_visionzip.py`
- `/project/6101803/enmingzz/vlm/official_thinking_in_space/third_party/VisionZip/Qwen2_5_VL/README.md`

The official README states that 70% retention is 65% dominant + 5% contextual,
and 50% retention is 45% dominant + 5% contextual. The Qwen2.5-VL implementation
uses the same pattern in code:

- `dominant_num = int(0.65 * attn_logits.size(0))`
- `contextual_num = max(int(0.05 * attn_logits.size(0)), 1)`

Our wrapper already matched the official attention/key extraction path:

- last visual layer attention is averaged and spatial-merged before pruning
- visual keys are spatial-merged and used as contextual merge metric
- dropped contextual tokens are merged into nearest kept contextual token

## Evaluation Setup

VLMEvalKit registration is correct for the pure VisionZip baseline:

- model mode: `visionzip_base`
- no LoRA adapter loaded
- ratios: 5%, 10%, 20%, 30%
- direct-answer prompt mode
- max generation: MME/POPE 16, MMStar 32

New official-split evaluation has been submitted:

- Slurm job: `3909846`
- Account: `aip-btaati`
- GPU request: `4*l40s`
- Script: `/project/6101803/enmingzz/opsd/scripts/eval_visionzip_base_official_split_vlmevalkit_mme_mmstar_pope.sh`
- Sbatch: `/project/6101803/enmingzz/opsd/slurm_jobs/eval_visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_btaati_8h.sbatch`
- Work dir: `/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/eval_vlmevalkit/visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030`
- Project summary: `/project/6101803/enmingzz/opsd/report/visionzip_base_official_split_vlmevalkit_mme_mmstar_pope_ratios_005_010_020_030.md`

## Checks Run

- `python -m py_compile` on VisionZip, Qwen wrapper, pruning forward, and VLMEvalKit wrapper/config.
- `python -m py_compile` on the official Qwen2.5-VL VisionZip implementation
  after ratio parameterization.
- CPU torch sanity check for:
  - official split counts across 5%, 10%, 20%, 30%, and 100%
  - dominant top-k selection when dominant budget is nonzero
  - drop-token input construction
  - attention mask, position ids, kept image positions, and student prompt length mapping
- VLMEvalKit config import check confirms all four pure VisionZip model names
  are registered with `model_mode="official_visionzip_base"`:
  - `opsd_qwen25vl_visionzip_base_r005`
  - `opsd_qwen25vl_visionzip_base_r010`
  - `opsd_qwen25vl_visionzip_base_r020`
  - `opsd_qwen25vl_visionzip_base_r030`

## Old-Split Scores For Reference Only

| Ratio | MME total | MME perception | MME reasoning | MMStar (%) | POPE |
|---:|---:|---:|---:|---:|---:|
| 005 | 1420.36 | 1129.64 | 290.71 | 33.93 | 66.56 |
| 010 | 1653.58 | 1361.44 | 292.14 | 39.67 | 78.82 |
| 020 | 1830.41 | 1524.70 | 305.71 | 45.87 | 83.14 |
| 030 | 1941.92 | 1586.56 | 355.36 | 49.33 | 85.18 |
