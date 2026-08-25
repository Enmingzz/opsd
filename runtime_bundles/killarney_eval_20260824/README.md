# Killarney evaluation runtime snapshot

This directory archives the exact shell entry points and runtime evidence for
the Qwen2.5-VL baseline evaluation used on Killarney on 2026-08-24. It contains
no checkpoints, benchmark data, generated outputs, credentials, or caches.

## Baseline represented here

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Resolved model revision | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Adapter | none |
| VisionZip | disabled |
| `visionzip_ratio` | `0.0` |
| Thinking prompt | enabled |
| Temperature | `0.0` |
| `max_new_tokens` | `2048` |
| Dataset | MME |
| Expected MME rows | `2374` |

The copied `eval_one.sh` also asserts the effective Qwen wrapper settings,
including `min_pixels=1280*28*28`, `max_pixels=4096*28*28`, the local
Transformers checkout, and the imported package versions.

## Requested artifacts

- `env_train_runtime.sh`: byte-identical Killarney environment entry point.
- `eval_one.sh`: byte-identical Killarney evaluator entry point.
- `pip-freeze-all.txt`: raw output of `python -m pip freeze --all` after sourcing
  `env_train_runtime.sh`.
- `wheels/Pillow_SIMD-9.5.0.post2+computecanada-cp311-cp311-linux_x86_64.whl`:
  the exact wheel whose `PIL/` files are imported at evaluation time.

Additional evidence is in `runtime-imports.json`, `dataset-manifest.json`,
`module-list.txt`, and `pillow-imaging-ldd.txt`. The dataset manifest verifies
that the actual `MME.tsv` has 2,375 lines (one header plus 2,374 examples) and
records its SHA256. The three small scripts under `activation_chain/` record
the site-specific activation chain used by `env_train_runtime.sh`.

## Important Pillow metadata conflict

The raw freeze contains both the consequences of overlapping installations and
reports `pillow==12.1.0+computecanada`. That line does **not** describe the code
actually imported by the evaluator. Runtime inspection gives:

```text
PIL.__version__ = 9.5.0.post2
PIL.__file__ = /scratch/enmingzz/temp/venvs/vsi-official/lib/python3.11/site-packages/PIL/__init__.py
Pillow-SIMD distribution = 9.5.0.post2+computecanada
Pillow distribution metadata = 12.1.0+computecanada
```

All 102 `PIL/` files present in the archived wheel match the installed files
byte for byte. There are zero mismatches and zero missing files. Therefore,
`runtime-imports.json` and the archived Pillow-SIMD wheel are authoritative for
runtime behavior; `pip-freeze-all.txt` is an evidence snapshot, not a directly
installable lockfile.

On a compatible Alliance Gentoo 2023 Python 3.11 environment, install the
archived Pillow-SIMD wheel last so a later Pillow install cannot overwrite
`PIL/`:

```bash
python -m pip uninstall -y Pillow Pillow-SIMD
python -m pip install --no-deps --force-reinstall \
  wheels/Pillow_SIMD-9.5.0.post2+computecanada-cp311-cp311-linux_x86_64.whl
```

Its dynamic dependencies are recorded in `pillow-imaging-ldd.txt`. The wheel
is cluster-targeted and should not be assumed portable to an arbitrary Linux
distribution.

## VLMEvalKit and Transformers

The evaluator imports Transformers from the dirty evaluation VLMEvalKit
checkout rather than from distribution metadata. Reconstruct that checkout
first using:

```bash
third_party_patches/vlmevalkit_51682/apply_and_verify.sh eval /path/to/VLMEvalKit
```

Then set `VLM_ROOT` to that checkout. The raw freeze's Transformers line is not
sufficient to reproduce the imported code; `runtime-imports.json` records the
actual import path and version.

## Verification

Static artifact hashes:

```bash
bash verify_bundle.sh
```

After activating the reconstructed runtime, verify imported versions, the
Pillow wheel contents, and the cached model revision:

```bash
VERIFY_ACTIVE_RUNTIME=1 bash verify_bundle.sh
```

If the model is not already cached, materialize the pinned revision and use the
returned local snapshot path as `BASE_MODEL_LOCAL_PATH`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
))
PY
```

## Baseline MME invocation

Set the site-specific roots before sourcing the archived environment script.
In particular, `PROJECT_ROOT` must contain the reconstructed activation chain,
and `VLM_ROOT` must point to the patched evaluation VLMEvalKit checkout.

```bash
export PROJECT_ROOT=/path/to/project
export CKPT_EVAL_ROOT=/path/to/eval-runtime
export VLM_ROOT=/path/to/patched/VLMEvalKit
export BASE_MODEL_LOCAL_PATH=/path/to/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
source baseline_mme.env
bash eval_one.sh
```

The exact Killarney scripts contain Killarney defaults. Override those paths on
another machine; do not edit the archived copies if byte-level provenance is
needed.
