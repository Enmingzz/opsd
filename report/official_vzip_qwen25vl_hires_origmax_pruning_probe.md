# Official VisionZip Pruning Probe

- MME TSV: `/scratch/enmingzz/vlmevalkit_data/MME.tsv`
- MME row: `0`
- image: `/scratch/enmingzz/vlmevalkit_data/images/MME/OCR/0001.jpg`
- generation cap for probe: `VISIONZIP_MAX_NEW_TOKENS=1`

| Model | Dominant | Contextual | Original image tokens | Expected kept | Actual kept | Actual keep % | Original seq | Pruned seq | 1-token response |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| opsd_qwen25vl_visionzip_flashattn2_hires_origmax_r030 | 0.25 | 0.05 | 1296 | 388 | 388 | 29.94 | 1334 | 426 | `Yes` |
| opsd_qwen25vl_visionzip_flashattn2_hires_origmax_r020 | 0.15 | 0.05 | 1296 | 258 | 258 | 19.91 | 1334 | 296 | `Yes` |
| opsd_qwen25vl_visionzip_flashattn2_hires_origmax_r010 | 0.05 | 0.05 | 1296 | 128 | 128 | 9.88 | 1334 | 166 | `Yes` |
| opsd_qwen25vl_visionzip_flashattn2_hires_origmax_r005 | 0.00 | 0.05 | 1296 | 64 | 64 | 4.94 | 1334 | 102 | `Yes` |
