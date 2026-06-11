# OPSD VisionZip Experiment Runbook

This repository contains the OPSD A-OKVQA training and VLMEvalKit evaluation workflow for Qwen2.5-VL-7B with the official VisionZip Qwen2.5-VL pruning implementation.

The intended use is to clone this repo on any GPU cluster, clone the two external upstream repos, apply the patches in `patches/`, and run the experiment launchers with cluster-specific paths and GPU settings.

## What This Runs

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Training data: `HuggingFaceM4/A-OKVQA`
- Pruning backend: official VisionZip Qwen2.5-VL implementation
- Training methods: SFT, EPIC TCD, EMA OPSD no GT, Freeze SFT Teacher OPSD no GT
- Evaluation: VLMEvalKit on MME, POPE, and MMStar
- Attention backend: `flash_attention_2`
- Adapter type: LoRA
- Training ratios: 10%, 20%, 30%, and 40%, sampled uniformly in the main configs
- Evaluation ratios: 5%, 10%, 20%, and 30%

Large artifacts are intentionally ignored by git: model weights, datasets, checkpoints, logs, raw VLMEvalKit outputs, and caches.

## GPU Requirements

The default scripts assume one multi-GPU node.

| Task | Minimum that should work | Recommended for current configs | Notes |
|---|---:|---:|---|
| First4 training, `gbs8` configs | 4 GPUs with 48GB each | 4 GPUs with 80GB or 96GB each | Current first4 configs use per-GPU micro batch 2, global batch 8. |
| Larger `mb8` configs | 4 GPUs with 80GB each | 4 GPUs with 96GB each | Per-GPU micro batch 8, global batch 32. Reduce `micro_batch_size` if OOM. |
| Larger `mb16` configs | 4 GPUs with 96GB each | 4 GPUs with 96GB each | Per-GPU micro batch 16, global batch 64. These are memory aggressive. |
| VLMEvalKit eval, fast mode | 4 GPUs with 48GB each | 4 GPUs with 80GB or 96GB each | Default is 12 workers across 4 GPUs, so 3 eval workers per GPU. |
| VLMEvalKit eval, safer mode | 1 to 4 GPUs with 24GB+ each | 4 GPUs with 48GB+ each | Set `NPROC_PER_NODE` equal to GPU count for one worker per GPU. |

Other practical requirements:

- CUDA-compatible PyTorch with `flash-attn`
- 150GB+ disk for model/cache/checkpoints, more if keeping raw eval outputs
- 64GB+ system RAM
- Stable network access to Hugging Face, or a local mirror/cache

Effective optimizer batch size is:

```text
world_size * micro_batch_size * gradient_accumulation_steps
```

For the main first4 run, this is `4 * 2 * 1 = 8`.

## Cluster Layout

Pick a shared experiment root on your cluster:

```bash
export BASE_ROOT=/path/to/opsd_exp
export OPSD_ROOT=${BASE_ROOT}/opsd
export VISIONZIP_ROOT=${BASE_ROOT}/VisionZip
export VLMEVALKIT_ROOT=${BASE_ROOT}/VLMEvalKit
export OUT_ROOT=${BASE_ROOT}/outputs/visionzip_aokvqa_reasoning
```

Clone the repos:

```bash
mkdir -p "${BASE_ROOT}"
git clone https://github.com/Enmingzz/opsd.git "${OPSD_ROOT}"
git clone https://github.com/JIA-Lab-research/VisionZip.git "${VISIONZIP_ROOT}"
git clone https://github.com/open-compass/VLMEvalKit.git "${VLMEVALKIT_ROOT}"
```

The currently aligned upstream commits are:

| Project | Commit |
|---|---|
| VisionZip | `8f86b55c6f000eb033e6912538af2dd7dcb30502` |
| VLMEvalKit | `58fdeb6b980bda22096d912d70d1c858dedc84fd` |

Apply the official-runtime patches:

```bash
cd "${VISIONZIP_ROOT}"
git checkout 8f86b55c6f000eb033e6912538af2dd7dcb30502
git apply "${OPSD_ROOT}/patches/visionzip_qwen25vl_official_runtime.patch"

cd "${VLMEVALKIT_ROOT}"
git checkout 58fdeb6b980bda22096d912d70d1c858dedc84fd
git apply "${OPSD_ROOT}/patches/vlmevalkit_official_visionzip_eval.patch"
```

## Environment

Activate your cluster Python environment first. It should contain PyTorch, Transformers with Qwen2.5-VL support, `qwen-vl-utils`, `peft`, `accelerate`, `datasets`, `Pillow`, `PyYAML`, `flash-attn`, and VLMEvalKit dependencies.

Set the runtime variables:

```bash
export BASE_ROOT=/path/to/opsd_exp
export OPSD_ROOT=${BASE_ROOT}/opsd
export VLMEVALKIT_ROOT=${BASE_ROOT}/VLMEvalKit
export VISIONZIP_QWEN25VL_ROOT=${BASE_ROOT}/VisionZip/Qwen2_5_VL
export OUT_ROOT=${BASE_ROOT}/outputs/visionzip_aokvqa_reasoning
export PYTHONPATH=${BASE_ROOT}:${VLMEVALKIT_ROOT}:${PYTHONPATH:-}
export HF_HOME=${BASE_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

For clusters in China, also set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Run The Main Training Experiment

The main run trains the four methods used in the current MME/POPE/MMStar comparison:

1. SFT
2. EPIC TCD
3. EMA OPSD no GT
4. Freeze SFT Teacher OPSD no GT

Run:

```bash
cd "${OPSD_ROOT}"

RUN_GROUP=aokvqa_first4_officialvz_$(date +%Y%m%d_%H%M%S) \
BASE_ROOT="${BASE_ROOT}" \
OPSD_ROOT="${OPSD_ROOT}" \
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_aokvqa_first4_officialvz_autodl.sh
```

The script name is historical; it is now path-configurable and can run on any cluster.

Outputs are written to:

```text
${OUT_ROOT}/checkpoints/${RUN_GROUP}/
${OUT_ROOT}/logs/train/${RUN_GROUP}/
```

The first4 configs are:

| Stage | Method | Config | Per-GPU batch | Global batch on 4 GPUs |
|---:|---|---|---:|---:|
| 1 | SFT | `configs/visionzip_aokvqa/aokvqa_sft_gbs8.yaml` | 2 | 8 |
| 2 | EPIC TCD | `configs/visionzip_aokvqa/aokvqa_epic_tcd_gbs8.yaml` | 2 | 8 |
| 3 | EMA OPSD no GT | `configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_gbs8.yaml` | 2 | 8 |
| 4 | Freeze SFT Teacher OPSD no GT | `configs/visionzip_aokvqa/aokvqa_opsd_sft_teacher_freeze_nogt_gbs8.yaml` | 2 | 8 |

To use fewer GPUs, change `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, and make sure each config's `training.max_steps` is divisible by:

```text
NPROC_PER_NODE * training.micro_batch_size
```

To use bigger GPUs more aggressively, edit `training.micro_batch_size` in the YAML config. The training CLI currently exposes `gradient_accumulation_steps` as an override, but not `micro_batch_size`.

## Run Additional Training Queues

The repo also contains launchers for older ordered runs and the six requested variants:

```bash
cd "${OPSD_ROOT}"

BASE_ROOT="${BASE_ROOT}" OUT_ROOT="${OUT_ROOT}" NPROC_PER_NODE=4 \
bash scripts/run_aokvqa_ordered_train_autodl.sh

BASE_ROOT="${BASE_ROOT}" OUT_ROOT="${OUT_ROOT}" NPROC_PER_NODE=4 \
bash scripts/run_aokvqa_requested_6train_autodl.sh
```

Before running configs that use an SFT teacher, check `opsd.teacher_adapter_path` in the corresponding YAML. If it points to an old checkpoint root, update it to your cluster's SFT checkpoint path.

## Run Evaluation

First make sure the VLMEvalKit patch has been applied. Then evaluate the trained first4 run:

```bash
cd "${OPSD_ROOT}"

RUN_ID=<your RUN_GROUP from training> \
BASE_ROOT="${BASE_ROOT}" \
OPSD_ROOT="${OPSD_ROOT}" \
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
VISIONZIP_QWEN25VL_ROOT="${VISIONZIP_QWEN25VL_ROOT}" \
DATASETS="MME MMStar POPE" \
RATIOS="r030 r020 r010 r005" \
NPROC_PER_NODE=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_first4_officialvz_vlmevalkit_autodl.sh
```

`NPROC_PER_NODE=12` means 12 VLMEvalKit workers total. With 4 GPUs, that is 3 workers per GPU. If evaluation is unstable or memory is tight, use:

```bash
NPROC_PER_NODE=4
```

The eval launcher exports these variables for patched VLMEvalKit aliases:

```bash
OPSD_BASE_ROOT=${BASE_ROOT}
OPSD_FIRST4_RUN_ID=${RUN_ID}
OPSD_FIRST4_CKPT_ROOT=${OUT_ROOT}/checkpoints/${RUN_ID}
VISIONZIP_QWEN25VL_ROOT=${BASE_ROOT}/VisionZip/Qwen2_5_VL
```

Evaluation outputs go to:

```text
${OUT_ROOT}/eval_vlmevalkit/
${OUT_ROOT}/logs/full/
${OUT_ROOT}/reports/
```

## Useful Model Aliases

The VLMEvalKit patch adds aliases for aligned runs:

| Alias pattern | Meaning |
|---|---|
| `opsd_qwen25vl_official_7b_flashattn2_mnt32` | Qwen2.5-VL no-prune baseline |
| `opsd_qwen25vl_visionzip_flashattn2_r030` | Official VisionZip baseline at 30% |
| `opsd_qwen25vl_visionzip_flashattn2_r020` | Official VisionZip baseline at 20% |
| `opsd_qwen25vl_visionzip_flashattn2_r010` | Official VisionZip baseline at 10% |
| `opsd_qwen25vl_visionzip_flashattn2_r005` | Official VisionZip baseline at 5% |
| `opsd_first4_officialvz_sft_r030` | SFT adapter, official VisionZip, 30% |
| `opsd_first4_officialvz_epic_r030` | EPIC adapter, official VisionZip, 30% |
| `opsd_first4_officialvz_ema_nogt_r030` | EMA OPSD no GT adapter, official VisionZip, 30% |
| `opsd_first4_officialvz_freeze_sftteacher_nogt_r030` | Freeze SFT Teacher OPSD no GT adapter, official VisionZip, 30% |

For MMStar-specific aligned aliases, use the corresponding `_mmstar_` names added by the patch.

## Reference Results

The current aligned baseline scores used as 100% are:

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

Full summary:

```text
reports/first4_officialvz_mme_pope_mmstar_relative_summary_latest.md
```

## Troubleshooting

- If training cannot import VisionZip, check `VISIONZIP_QWEN25VL_ROOT`.
- If VLMEvalKit cannot find model aliases, re-apply `patches/vlmevalkit_official_visionzip_eval.patch`.
- If `flash_attention_2` fails, verify CUDA, PyTorch, and `flash-attn` versions. Falling back to SDPA changes the comparison setup.
- If evaluation OOMs, lower `NPROC_PER_NODE`.
- If training OOMs, lower `training.micro_batch_size` in the YAML.
- If a teacher checkpoint is missing, verify `RUN_GROUP`, `RUN_ID`, and any `opsd.teacher_adapter_path` values.
