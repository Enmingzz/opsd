# OPSD VisionZip Experiments

This repository contains the Qwen2.5-VL A-OKVQA training and evaluation workflow for OPSD experiments with the official VisionZip token pruning implementation.

The current experiment stack is:

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Pruning backend: official VisionZip Qwen2.5-VL runtime
- Training dataset: `HuggingFaceM4/A-OKVQA`
- Evaluation: VLMEvalKit on MME, POPE, and MMStar
- Attention backend: `flash_attention_2`
- Adapter training: LoRA
- Training ratios: 10%, 20%, 30%, and 40%, sampled uniformly unless a config says otherwise

Large artifacts such as datasets, checkpoints, logs, and raw evaluation outputs are intentionally ignored. The repo keeps source code, configs, launcher scripts, external patches, and compact result summaries.

## Repository Layout

| Path | Purpose |
|---|---|
| `visionzip_aokvqa/` | A-OKVQA training code, official VisionZip model loading, SFT, EPIC, and OPSD losses |
| `pruning_distill/` | Shared pruned forward and distillation utilities |
| `configs/visionzip_aokvqa/` | A-OKVQA experiment configs |
| `scripts/` | AutoDL training and VLMEvalKit evaluation launchers |
| `patches/` | Patches for the external official VisionZip and VLMEvalKit checkouts |
| `reports/` | Small CSV/Markdown summaries of aligned evaluation results |

## External Code Alignment

Training and evaluation require the official VisionZip implementation, not the older local fallback. The local training wrapper imports the official Qwen2.5-VL VisionZip source through `VISIONZIP_QWEN25VL_ROOT`.

The aligned external checkouts used for the current runs were:

| Project | Upstream | Commit |
|---|---|---|
| VisionZip | `https://github.com/JIA-Lab-research/VisionZip.git` | `8f86b55c6f000eb033e6912538af2dd7dcb30502` |
| VLMEvalKit | `https://github.com/open-compass/VLMEvalKit.git` | `58fdeb6b980bda22096d912d70d1c858dedc84fd` |

Apply the stored patches to those external repos:

```bash
cd /root/autodl-tmp/opsd_eval/VisionZip
git apply /root/autodl-tmp/opsd_eval/opsd/patches/visionzip_qwen25vl_official_runtime.patch

cd /root/autodl-tmp/opsd_eval/VLMEvalKit
git apply /root/autodl-tmp/opsd_eval/opsd/patches/vlmevalkit_official_visionzip_eval.patch
```

The VLMEvalKit patch adds the Qwen2.5-VL official VisionZip model aliases used by the launcher scripts, including no-prune baseline, VisionZip ratios, and trained LoRA adapter variants.

## AutoDL Environment

The scripts assume this directory layout on AutoDL:

```text
/root/autodl-tmp/opsd_eval/
  opsd/
  VisionZip/
  VLMEvalKit/
```

Recommended environment variables:

```bash
export BASE_ROOT=/root/autodl-tmp/opsd_eval
export OPSD_ROOT=${BASE_ROOT}/opsd
export VLMEVALKIT_ROOT=${BASE_ROOT}/VLMEvalKit
export VISIONZIP_QWEN25VL_ROOT=${BASE_ROOT}/VisionZip/Qwen2_5_VL
export PYTHONPATH=${VLMEVALKIT_ROOT}:${OPSD_ROOT}:${PYTHONPATH:-}
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Install the normal Qwen2.5-VL training stack in the active environment, including PyTorch, Transformers with Qwen2.5-VL support, `qwen-vl-utils`, `peft`, `accelerate`, `flash-attn`, `datasets`, `Pillow`, `PyYAML`, and VLMEvalKit dependencies.

## Training

Run the four primary official-VisionZip training jobs in sequence:

```bash
cd /root/autodl-tmp/opsd_eval/opsd
bash scripts/run_aokvqa_first4_officialvz_autodl.sh
```

The script runs:

| Stage | Method | Config |
|---:|---|---|
| 1 | SFT | `configs/visionzip_aokvqa/aokvqa_sft_gbs8.yaml` |
| 2 | EPIC TCD | `configs/visionzip_aokvqa/aokvqa_epic_tcd_gbs8.yaml` |
| 3 | EMA OPSD no GT | `configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_gbs8.yaml` |
| 4 | Freeze SFT Teacher OPSD no GT | generated from `configs/visionzip_aokvqa/aokvqa_opsd_sft_teacher_freeze_nogt_gbs8.yaml` |

Current training defaults:

- `torchrun --nproc-per-node=4`
- per-GPU micro batch size: 2
- global batch size: 8
- LoRA rank: 16
- generation max new tokens: 512
- EMA decay for EMA teacher configs: 0.9999
- teacher input: full visual tokens
- student input: official VisionZip pruned visual tokens

Additional requested experiment configs are also present under `configs/visionzip_aokvqa/`, including progressive-ratio variants and ground-truth variants.

## Evaluation

The main VLMEvalKit launcher uses 12 workers across 4 GPUs by default, so each GPU hosts 3 evaluation workers:

```bash
cd /root/autodl-tmp/opsd_eval/opsd
bash scripts/run_vlmevalkit_autodl_12worker.sh
```

To evaluate the first four trained methods on MME, MMStar, and POPE:

```bash
cd /root/autodl-tmp/opsd_eval/opsd
RUN_ID=aokvqa_first4_officialvz_20260610_154040 \
DATASETS="MME MMStar POPE" \
RATIOS="r030 r020 r010 r005" \
NPROC_PER_NODE=12 \
bash scripts/eval_first4_officialvz_vlmevalkit_autodl.sh
```

The no-prune Qwen2.5-VL baseline and official VisionZip aliases are provided by the VLMEvalKit patch. Keep `flash_attention_2` enabled for aligned comparisons.

## Current Aligned Results

Baseline scores used as 100%:

| Model | MME | POPE | MMStar |
|---|---:|---:|---:|
| Qwen2.5-VL no-prune | 2300.55 | 86.29 | 61.47 |

Best average percentage by ratio across MME, POPE, and MMStar:

| Ratio | Best method | Avg % |
|---:|---|---:|
| 30% | Freeze SFT Teacher OPSD no GT | 100.60 |
| 20% | EPIC | 99.48 |
| 10% | SFT | 97.49 |
| 5% | Freeze SFT Teacher OPSD no GT | 91.74 |

Average across evaluated ratios:

| Method | Avg % |
|---|---:|
| VisionZip official | 91.14 |
| SFT | 96.97 |
| EPIC | 96.95 |
| EMA OPSD no GT | 95.74 |
| Freeze SFT Teacher OPSD no GT | 97.31 |

See `reports/first4_officialvz_mme_pope_mmstar_relative_summary_latest.md` for the full combined table.

## Notes

- Do not commit checkpoints, raw datasets, model weights, or full VLMEvalKit work directories.
- Keep training and evaluation on the same official VisionZip runtime for comparable numbers.
- The old local `visionzip_aokvqa/visionzip.py` fallback has been removed so experiments fail fast if the official VisionZip checkout is missing.
