# Core OPSD Training and Evaluation Environment

Captured on 2026-08-24 from an Alliance Canada Killarney L40S node. This is
the runtime used by the current Qwen2.5-VL OPSD/SFT experiments and the main
benchmark evaluation pipeline.

## Reproduction warning

The OPSD core files listed below are versioned together with this document.
Clone the latest `main` revision and verify the recorded file hashes before
reproducing a run:

```bash
git rev-parse HEAD
sha256sum \
  visionzip_aokvqa/train.py \
  visionzip_aokvqa/losses.py \
  visionzip_aokvqa/native_budget_weighting.py
```

The repository still depends on separately patched VisionZip and VLMEvalKit
trees. The two exact dirty VLMEvalKit states are captured as verified patches
under `third_party_patches/vlmevalkit_51682/`; reconstruct those checkouts and
match the source-path precedence below before reproducing a run.

## Hardware and system runtime

| Component | Runtime value |
|---|---|
| GPU | NVIDIA L40S, 46,068 MiB |
| NVIDIA driver | 580.159.03 |
| Python | 3.11.5, GCC 12.3.1 |
| CUDA module | 12.6 |
| PyTorch CUDA build | 12.6 |
| NCCL module | 2.26.2 |
| Arrow module | 23.0.1 |
| OpenCV module | 4.11.0 |
| Precision | BF16; TF32 allowed |
| Attention implementation | FlashAttention 2 |

The canonical activation entry point is:

```bash
source /project/6101803/enmingzz/env/vsi-official.sh
```

It activates:

```text
/scratch/enmingzz/temp/venvs/vsi-official/bin/python
```

and swaps the cluster runtime to CUDA 12.6 and Arrow 23.0.1.

## Core training packages

These are imported runtime versions, which take precedence over potentially
stale `pip show` metadata.

| Package | Version |
|---|---:|
| torch | 2.9.1+computecanada |
| torchvision | 0.24.1+computecanada |
| transformers | 4.57.0, locally patched source |
| tokenizers | 0.22.2 |
| huggingface_hub | 0.34.3 |
| peft | 0.18.1+computecanada |
| accelerate | 1.13.0+computecanada |
| flash-attn | 2.8.3+torch29.computecanada |
| triton | 3.6.0+computecanada |
| qwen-vl-utils | 0.0.14+computecanada |
| datasets | 4.8.5 |
| safetensors | 0.7.0+computecanada |
| numpy | 2.4.2+computecanada |
| pandas | 3.0.0+computecanada |
| pyarrow | 23.0.1 |
| scipy | 1.17.0+computecanada |
| matplotlib | 3.10.8+computecanada |
| PyYAML | 6.0.3+computecanada |
| openpyxl | 3.1.5+computecanada |
| tqdm | 4.67.3+computecanada |
| Pillow, imported runtime | 9.5.0.post2 |

Important: the installed Pillow distribution metadata reports 12.1.0, but
the module actually imported by the canonical runtime reports 9.5.0.post2.
Use the imported runtime value when reproducing evaluator behavior.

## Required source overrides

The pruning implementation depends on a patched Qwen2.5-VL Transformers
source. A stock `transformers==4.57.0` is not equivalent.

Training uses:

```bash
export ARMEN_TRANSFORMERS_SRC="$OPSD_ROOT/third_party/VLMEvalKit_armen51682/transformers/src"
export HF_HUB034_ROOT=/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2
export TOKENIZERS_QWEN25_ROOT=/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only
export VISIONZIP_QWEN25VL_ROOT="$OPSD_ROOT/third_party/VisionZip/Qwen2_5_VL"
export PYTHONPATH=/project/6101803/enmingzz
```

`opsd.visionzip_aokvqa.qwen_wrapper.bootstrap_qwen25()` deliberately puts the
patched Transformers, Hugging Face Hub 0.34.3, and Tokenizers 0.22.2 ahead of
the base virtual environment. Do not remove this bootstrap or import a
different Transformers package before it.

Main benchmark evaluation uses:

```bash
source /project/6101803/enmingzz/ckpt_eval_trainenv/env_train_runtime.sh
```

That script places these paths first:

```text
/project/6101803/enmingzz/ckpt_eval_trainenv/VLMEvalKit_armen51682
/project/6101803/enmingzz/ckpt_eval_trainenv/VLMEvalKit_armen51682/transformers/src
/scratch/enmingzz/cache/uv/archive-v0/DGthIN4hMUv1qyt2
/scratch/enmingzz/temp/pydeps_armen_clean_tokenizers_only
```

## Repository identities

| Repository/source | Identity |
|---|---|
| OPSD | latest synchronized `main`; record `git rev-parse HEAD` with every run |
| VisionZip checkout | `8f86b55c6f000eb033e6912538af2dd7dcb30502` |
| VLMEvalKit clean base | `51682a6baab948d3dbb4b867a3eab178504ac3f5` plus local evaluator patches |
| Qwen2.5-VL-7B-Instruct HF revision | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |

The training and evaluation VLMEvalKit checkouts are intentionally different.
Their local changes include the Qwen2.5-VL VisionZip model wrapper,
FlashAttention training correction, raw-output preservation, image-MCQ
parsing, and inference/evaluator hooks. Reconstruct both from the exact
upstream commit with:

```bash
bash third_party_patches/vlmevalkit_51682/apply_and_verify.sh \
  train /absolute/path/to/VLMEvalKit_train
bash third_party_patches/vlmevalkit_51682/apply_and_verify.sh \
  eval /absolute/path/to/VLMEvalKit_eval
```

## Core file hashes

Use these to verify that the second machine is running the same code.

| File | SHA256 |
|---|---|
| `visionzip_aokvqa/train.py` | `579c971345987fee54f0d559c30690bcc3a4e9ccc5c7aba96f0e30ab133c9119` |
| `visionzip_aokvqa/qwen_wrapper.py` | `e973daced0492bec8ac9ed60fa8ae61f2df31d053b3b13c08cce3a2a09bac31f` |
| `visionzip_aokvqa/losses.py` | `e1402543263251373584e0f2404836aed3cc27cd54254eefb8603e4c3c9edb27` |
| `visionzip_aokvqa/native_budget_weighting.py` | `b57b9f59c8fb1654b6b320683860d83a4a359a2b5d85dc388aebe31be069a0c7` |
| `pruning_distill/pruners.py` | `c11d455816005b83685976a8ab171621baf9206ee683b65fe54376a8270a442c` |
| `visionzip_aokvqa/prompting.py` | `ce6fde4a9d7d1d8256d4a8e9e80eb875078a244d336fde34f71f3735f850533e` |
| training patched `modeling_qwen2_5_vl.py` | `4a207f07cd6d3d20fdd9afd33a3933cffc3f1d4ab1d386900579eba09b9e558a` |
| eval patched `modeling_qwen2_5_vl.py` | `97d0f146fc760162039a6883a9a0a787b7cafe3d550e737210b9194294f7b683` |
| eval `vlmeval/vlm/qwen2_vl/model.py` | `a903369dd1c6df7749b3a53f2e847fef876a0e5c6ede4da21e445d4e6abce72d` |

The training and evaluation model files intentionally come from two patched
trees and do not have the same hash. Preserve the path selection shown above.

## Core model and data hashes

| Artifact | SHA256 or revision |
|---|---|
| Qwen2.5-VL-7B config | `77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43` |
| ordered decontaminated LCOT-20k JSONL | `9f1cf9dfbff291ee0ce7f34a236820c7da907f096bdb54ec0165eed845f3516f` |
| LCOT-20k training manifest | `2b668e412fa51c159bbe5e9dfd4605da3555412fef55035df084ca258d39ae87` |
| decontaminated LCOT-1k metric JSONL | `704e33c056e9382b2deec1adcc86906d65a6cc2ce6e95b9b58d96760e795d2de` |

## Main benchmark evaluator

The canonical runner is:

```text
/project/6101803/enmingzz/ckpt_eval_trainenv/eval_one.sh
```

Its verified runtime is:

| Package | Version |
|---|---:|
| VLMEvalKit | commit `51682a6baab948d3dbb4b867a3eab178504ac3f5` plus local patches |
| torch | 2.9.1, CUDA 12.6 |
| transformers | 4.57.0, eval patched source |
| tokenizers | 0.22.2 |
| huggingface_hub | 0.34.3 |
| peft | 0.18.1 |
| accelerate | 1.13.0 |
| flash-attn | 2.8.3 |
| pyarrow | 23.0.1 |
| Pillow | 9.5.0.post2 |
| OpenCV | 4.11.0 |

The Qwen wrapper defaults used by the audited main evaluator are:

```text
min_pixels       = 1280 * 28 * 28 = 1,003,520
max_pixels       = 4096 * 28 * 28 = 3,211,264
max_new_tokens   = 2048
enable_thinking  = true
temperature      = 0.0
use_kv_cache     = true
raw outputs      = saved
```

Merged BF16 checkpoints are the default evaluation form unless an experiment
explicitly states otherwise.

## Strict MathVista and MathVerse fallback judge

Official parsing/exact matching is applied first. Only unresolved candidates
are passed to the strict ground-truth judge.

Judge model:

```text
/scratch/enmingzz/models/Qwen3.6-27B-FP8
served name: Qwen/Qwen3.6-27B-FP8
config SHA256: 885e6830f8d6883fefd63e3608c267452da2e6ce353a3494f42a7aa3d70c8434
```

Isolated judge runtime:

| Package | Version |
|---|---:|
| Python | 3.11 |
| vLLM | 0.25.1 |
| torch | 2.11.0+cu130 |
| transformers | 5.14.1 |
| tokenizers | 0.22.2 |
| numpy | 2.3.5 |
| pandas | 3.0.3 |
| openai | 2.46.0 |

Canonical setup and postprocessors:

```text
opsd/scripts/eval/qwen36_judge_runtime.sh
opsd/scripts/eval/postprocess_mathvista_strict_gt.py
opsd/scripts/eval/postprocess_mathverse_strict_gt.py
```

Their SHA256 values are, respectively:

```text
cfaaca487cb51edf770c6fea1388393983cf27042e2c2fdd9ecc6f348a01a125
d047055d20ab5823471ad1e925055335107d990bb921d7e36944237101f0b0f4
86292ffeb354b27a2d2bce0a7141658a732838080ab2338f3fcc7cd097311e78
```

Do not mix the judge vLLM environment with the training environment; their
Torch/CUDA stacks are intentionally different.

## Minimal setup on another L40S machine

1. Use Python 3.11, CUDA 12.6, and a driver new enough for CUDA 12.6.
2. Install the training package versions listed above, including a
   CUDA-12.6 build of PyTorch 2.9.1 and FlashAttention 2.8.3.
3. Synchronize the current OPSD working tree and the patched VLMEvalKit tree.
4. Set the three source overrides before launching training:

```bash
export PROJECT_ROOT=/path/to/enmingzz
export OPSD_ROOT="$PROJECT_ROOT/opsd"
export ARMEN_TRANSFORMERS_SRC="$OPSD_ROOT/third_party/VLMEvalKit_armen51682/transformers/src"
export HF_HUB034_ROOT=/path/to/huggingface_hub_0.34.3
export TOKENIZERS_QWEN25_ROOT=/path/to/tokenizers_0.22.2
export VISIONZIP_QWEN25VL_ROOT="$OPSD_ROOT/third_party/VisionZip/Qwen2_5_VL"
export PYTHONPATH="$PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

5. Run a preflight that prints imported versions and file paths. It must show
   patched Transformers 4.57.0, Tokenizers 0.22.2, Hugging Face Hub 0.34.3,
   Torch 2.9.1/CUDA 12.6, and FlashAttention 2.8.3.
6. Verify all core file and data hashes before starting a long run.

Example version check:

```bash
python - <<'PY'
from opsd.visionzip_aokvqa.qwen_wrapper import bootstrap_qwen25
bootstrap_qwen25()
import accelerate, flash_attn, huggingface_hub, peft, tokenizers, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.__file__)
print("transformers", transformers.__version__, transformers.__file__)
print("tokenizers", tokenizers.__version__, tokenizers.__file__)
print("huggingface_hub", huggingface_hub.__version__, huggingface_hub.__file__)
print("peft", peft.__version__, peft.__file__)
print("accelerate", accelerate.__version__, accelerate.__file__)
print("flash_attn", flash_attn.__version__, flash_attn.__file__)
PY
```
