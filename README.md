# OPSD VisionZip A-OKVQA Runbook

This repository contains the A-OKVQA training and VLMEvalKit evaluation code used for the Qwen2.5-VL + VisionZip OPSD experiments.

The current public checkpoint bundle contains three PEFT LoRA adapters:

| Method | Hugging Face subfolder |
|---|---|
| SFT | `sft/` |
| EPIC | `epic/` |
| OPSD-512 | `opsd512/` |

Checkpoint repo:

```text
enmingzhangzz/opsd-aokvqa-qwen25vl-lora-cleanenv-20260621
```

Base model:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

The adapters are LoRA weights only. They do not include the base Qwen2.5-VL model.

## Current Experiment Settings

Training:

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Dataset | `HuggingFaceM4/A-OKVQA` |
| Adapter | LoRA |
| Main GPU setup | `4 * L40S` |
| Effective batch size | `32` |
| OPSD EMA decay | `0.9999` |
| OPSD ground truth use | No GT in OPSD loss |
| OPSD-512 rollout cap | `512` generated tokens |
| Reasoning target format | `<think>...</think><answer>...</answer>` |

Evaluation:

| Item | Value |
|---|---|
| Eval codebase | Armen VLMEvalKit commit `51682a6b` plus local patch |
| Benchmarks | MME, POPE, MMStar |
| Image pixels | `min_pixels = 1280 * 28 * 28`, `max_pixels = 4096 * 28 * 28` |
| Decoding | Greedy, `temperature = 0.0` |
| Max new tokens | VLMEvalKit/Qwen default `2048` |
| Attention | `flash_attention_2` |
| KV cache | Enabled in the current clean eval launcher |

VisionZip ratio mapping:

| Retention tag | Intended retained visual tokens | Launcher `visionzip_ratio` |
|---|---:|---:|
| `r005` | 5% | `1.00` |
| `r010` | 10% | `0.95` |
| `r020` | 20% | `0.85` |
| `r030` | 30% | `0.75` |

Do not pass `0.05`, `0.10`, `0.20`, or `0.30` directly to the current launcher. In this code path, `visionzip_ratio` is the pruning-side argument used by the patched Qwen2.5-VL VisionZip implementation, and the mapping above is the corrected mapping used for the reported results.

## Download The Adapters

Log in to Hugging Face if the repo is private:

```bash
hf auth login
```

Download all adapters:

```bash
hf download enmingzhangzz/opsd-aokvqa-qwen25vl-lora-cleanenv-20260621 \
  --local-dir ./opsd_aokvqa_loras
```

The downloaded layout should be:

```text
opsd_aokvqa_loras/
  sft/
    adapter_config.json
    adapter_model.safetensors
  epic/
    adapter_config.json
    adapter_model.safetensors
  opsd512/
    adapter_config.json
    adapter_model.safetensors
    ema_shadow.pt
```

For inference/evaluation, only `adapter_config.json` and `adapter_model.safetensors` are required. `ema_shadow.pt` is kept for training-state provenance.

## Load An Adapter With PEFT

```python
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

base_model = "Qwen/Qwen2.5-VL-7B-Instruct"
adapter_repo = "enmingzhangzz/opsd-aokvqa-qwen25vl-lora-cleanenv-20260621"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# Choose one: subfolder="sft", "epic", or "opsd512"
model = PeftModel.from_pretrained(model, adapter_repo, subfolder="opsd512")
model.eval()

processor = AutoProcessor.from_pretrained(base_model)
```

For local adapters:

```python
model = PeftModel.from_pretrained(model, "./opsd_aokvqa_loras/opsd512")
```

## Set Up The Clean VLMEvalKit

The reported clean results used Armen's VLMEvalKit at commit `51682a6b` with a small local patch for:

- PEFT adapter loading through `adapter_path`
- VisionZip enable/ratio arguments
- `max_pixels = 4096 * 28 * 28`
- strict failure mode during distributed eval
- cached pruned-input debugging state

Clone the eval kit:

```bash
mkdir -p third_party
git clone https://github.com/armenjeddi/VLMEvalKit.git third_party/VLMEvalKit_armen51682
cd third_party/VLMEvalKit_armen51682
git checkout 51682a6baab948d3dbb4b867a3eab178504ac3f5
git apply ../../patches/vlmevalkit_armen51682_cleanenv_qwen25vl_lora_visionzip.patch
```

The patch file is:

```text
patches/vlmevalkit_armen51682_cleanenv_qwen25vl_lora_visionzip.patch
```

On the Vector cluster, the already-aligned working copy is:

```text
/project/6101803/enmingzz/ckpt_eval_trainenv/VLMEvalKit_armen51682
```

The current canonical local launcher is:

```text
/project/6101803/enmingzz/ckpt_eval_trainenv/eval_one.sh
```

The GitHub repo also contains older job wrappers under `scripts/` and `slurm_jobs/`. For exact cleanenv reproduction on the current cluster, prefer the `ckpt_eval_trainenv/eval_one.sh` launcher above or port its environment setup into your own cluster.

## Run A Single Evaluation Locally

Example: OPSD-512, reasoning mode, 20% retention, MME + POPE:

```bash
export CKPT_EVAL_ROOT=/project/6101803/enmingzz/ckpt_eval_trainenv
export VLM_ROOT=${CKPT_EVAL_ROOT}/VLMEvalKit_armen51682
export ADAPTER_TAG=opsd512
export RATIO_TAG=r020
export adapter_path=/path/to/opsd_aokvqa_loras/opsd512
export visionzip_ratio=0.85
export enable_thinking=True
export enable_visionzip=True
export temperature=0.0
export EVAL_DATASETS="MME POPE"
export EVAL_NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash /project/6101803/enmingzz/ckpt_eval_trainenv/eval_one.sh
```

Direct mode is the same except:

```bash
export enable_thinking=False
```

No-prune baseline:

```bash
export ADAPTER_TAG=baseline
export RATIO_TAG=noprune
export adapter_path=none
export enable_visionzip=False
export visionzip_ratio=0.0
```

VisionZip baseline without adapter:

```bash
export ADAPTER_TAG=baseline
export RATIO_TAG=r020
export adapter_path=none
export enable_visionzip=True
export visionzip_ratio=0.85
```

## Common Evaluation Commands

Adapter paths after `hf download`:

```bash
SFT_ADAPTER=./opsd_aokvqa_loras/sft
EPIC_ADAPTER=./opsd_aokvqa_loras/epic
OPSD512_ADAPTER=./opsd_aokvqa_loras/opsd512
```

Ratio helper:

```bash
case "${RATIO_TAG}" in
  r005) visionzip_ratio=1.00 ;;
  r010) visionzip_ratio=0.95 ;;
  r020) visionzip_ratio=0.85 ;;
  r030) visionzip_ratio=0.75 ;;
  *) echo "Unknown ratio ${RATIO_TAG}" >&2; exit 1 ;;
esac
export visionzip_ratio
```

Run SFT r030 reasoning on all three benchmarks:

```bash
export ADAPTER_TAG=sft
export RATIO_TAG=r030
export adapter_path="${SFT_ADAPTER}"
export visionzip_ratio=0.75
export enable_thinking=True
export EVAL_DATASETS="MME MMStar POPE"
bash /project/6101803/enmingzz/ckpt_eval_trainenv/eval_one.sh
```

Run EPIC r030:

```bash
export ADAPTER_TAG=epic
export RATIO_TAG=r030
export adapter_path="${EPIC_ADAPTER}"
export visionzip_ratio=0.75
export enable_thinking=True
export EVAL_DATASETS="MME MMStar POPE"
bash /project/6101803/enmingzz/ckpt_eval_trainenv/eval_one.sh
```

Run OPSD-512 r030:

```bash
export ADAPTER_TAG=opsd512
export RATIO_TAG=r030
export adapter_path="${OPSD512_ADAPTER}"
export visionzip_ratio=0.75
export enable_thinking=True
export EVAL_DATASETS="MME MMStar POPE"
bash /project/6101803/enmingzz/ckpt_eval_trainenv/eval_one.sh
```

## Output Locations

Current cluster defaults:

```text
/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/eval_vlmevalkit_trainenv/
/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/full/
```

Each eval run creates a directory like:

```text
${OUT_ROOT}/eval_vlmevalkit_trainenv/${RUN_GROUP}/${ADAPTER_TAG}_${RATIO_TAG}/Qwen/TYYYYMMDD_G51682a6b/
```

Important score files:

```text
Qwen_MME_score.csv
Qwen_MMStar_acc.csv
Qwen_POPE_score.csv
```

For MME, the total score is:

```text
perception + reasoning
```

For POPE, VLMEvalKit's `Overall` column is F1, not accuracy. The same CSV also reports `acc`, `precision`, and `recall`.

## Current Reference Scores

Reasoning mode, `r030`, cleanenv:

| Method | MME | POPE F1 | POPE Acc | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Baseline | 2189.280 | 80.592 | 83.400 | 96.998 | 68.933 |
| SFT | 2247.017 | 82.406 | 80.144 | 73.979 | 93.000 |
| EPIC | 2120.064 | 79.308 | 76.533 | 72.133 | 88.067 |
| OPSD-512 | 2250.260 | 82.187 | 84.433 | 96.326 | 71.667 |

Direct mode, `r005/r010/r020`, cleanenv:

| Retention | Method | MME | MMStar | POPE |
|---|---|---:|---:|---:|
| 5% | Baseline | 1830.391 | 43.800 | 74.058 |
| 5% | SFT | 2055.261 | 49.333 | 81.040 |
| 5% | EPIC | 2092.244 | 50.533 | 81.554 |
| 5% | OPSD-512 | 2090.979 | 48.733 | 78.941 |
| 10% | Baseline | 2097.136 | 51.400 | 81.020 |
| 10% | SFT | 2233.061 | 55.933 | 85.429 |
| 10% | EPIC | 2235.535 | 57.533 | 85.493 |
| 10% | OPSD-512 | 2216.427 | 55.933 | 84.015 |
| 20% | Baseline | 2242.759 | 56.333 | 84.145 |
| 20% | SFT | 2303.393 | 58.733 | 86.943 |
| 20% | EPIC | 2299.381 | 59.067 | 86.911 |
| 20% | OPSD-512 | 2285.487 | 57.733 | 85.084 |

## Train The Main Methods

The main 4-L40S submitter is:

```bash
scripts/submit_aokvqa_reasoning_ebs32_4l40s.sh
```

Dry run:

```bash
DRY_RUN=1 bash scripts/submit_aokvqa_reasoning_ebs32_4l40s.sh
```

Submit with defaults:

```bash
bash scripts/submit_aokvqa_reasoning_ebs32_4l40s.sh
```

Useful overrides:

```bash
ACCOUNT=aip-btaati \
PARTITION=gpubase_l40s_b3 \
TIME_LIMIT=16:00:00 \
CPUS_PER_TASK=32 \
bash scripts/submit_aokvqa_reasoning_ebs32_4l40s.sh
```

Stages submitted by this launcher:

| Stage | Config |
|---|---|
| SFT | `configs/visionzip_aokvqa/aokvqa_sft_reasoning_ebs32_4l40s.yaml` |
| Pure OPSD no GT | `configs/visionzip_aokvqa/aokvqa_opsd_nogt_ema_reasoning_ebs32_4l40s.yaml` |
| Freeze SFT-teacher OPSD no GT | `configs/visionzip_aokvqa/aokvqa_opsd_nogt_freeze_sft_teacher_reasoning_ebs32_4l40s.yaml` |
| SFT EMA-teacher OPSD no GT | `configs/visionzip_aokvqa/aokvqa_opsd_nogt_sft_ema_teacher_reasoning_ebs32_4l40s.yaml` |
| EPIC | `configs/visionzip_aokvqa/aokvqa_epic_tcd_reasoning_ebs32_4l40s.yaml` |

Outputs:

```text
/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/checkpoints/${RUN_GROUP}/
/scratch/enmingzz/outputs/visionzip_aokvqa_reasoning/logs/train/${RUN_GROUP}/
```

## Notes And Common Pitfalls

- The HF adapters are LoRA checkpoints. Always load them on top of `Qwen/Qwen2.5-VL-7B-Instruct`.
- `opsd512/ema_shadow.pt` is not needed for inference.
- For POPE, `Overall` is F1. Use the `acc` column if you want accuracy.
- The corrected VisionZip mapping is `r005 -> 1.00`, `r010 -> 0.95`, `r020 -> 0.85`, `r030 -> 0.75`.
- The cleanenv eval uses `max_pixels = 4096 * 28 * 28`. Older local scripts may still mention `16384 * 28 * 28`; do not mix those results.
- The current exact cleanenv launcher lives outside the repo at `/project/6101803/enmingzz/ckpt_eval_trainenv/eval_one.sh`. The reproducibility-critical code changes are captured in `patches/vlmevalkit_armen51682_cleanenv_qwen25vl_lora_visionzip.patch`.
