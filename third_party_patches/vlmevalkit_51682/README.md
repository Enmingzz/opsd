# Killarney VLMEvalKit Patch Bundle

The OPSD training and benchmark evaluation runtimes use two intentionally
different dirty checkouts of VLMEvalKit. Both start from the exact upstream
revision:

```text
repository: https://github.com/open-compass/VLMEvalKit.git
commit:     51682a6baab948d3dbb4b867a3eab178504ac3f5
```

There were no untracked files in either checkout when these patches were
captured on 2026-08-24.

## Why there are two patches

`train_checkout.patch` contains the four training-side modifications used by
the OPSD trainer:

- patched Qwen2.5-VL VisionZip forward hooks;
- the FlashAttention visual-gradient correction;
- the local Qwen2.5-VL/VisionZip model registration and wrapper settings.

`eval_checkout.patch` contains the seven evaluation-side modifications used by
the audited main-table pipeline:

- merged/adapted Qwen2.5-VL VisionZip loading and generation hooks;
- raw prediction preservation and inference plumbing;
- image-MCQ and multiple-choice parsing fixes;
- evaluator entry-point and model configuration changes.

Do not apply both patches to one checkout. The two Qwen2.5-VL model files are
different by design and have different target hashes.

## Restore on another machine

Create two clean clones at the pinned commit:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git VLMEvalKit_train
git -C VLMEvalKit_train checkout 51682a6baab948d3dbb4b867a3eab178504ac3f5

git clone https://github.com/open-compass/VLMEvalKit.git VLMEvalKit_eval
git -C VLMEvalKit_eval checkout 51682a6baab948d3dbb4b867a3eab178504ac3f5
```

From the OPSD repository root, apply and verify each patch:

```bash
bash third_party_patches/vlmevalkit_51682/apply_and_verify.sh \
  train /absolute/path/to/VLMEvalKit_train

bash third_party_patches/vlmevalkit_51682/apply_and_verify.sh \
  eval /absolute/path/to/VLMEvalKit_eval
```

The script requires a clean checkout at the exact base commit, validates the
patch SHA256 before applying it, runs `git apply --check`, verifies the exact
changed-file set, and checks every resulting file hash. It is idempotent for a
checkout that already matches the captured Killarney tree.

Package versions, source-path precedence, model/data identities, and runtime
settings are documented in
`docs/reproducibility/CORE_TRAIN_EVAL_ENVIRONMENT_20260824.md`.
