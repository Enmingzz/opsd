# Cross-machine Runtime Fingerprint

The Killarney export is packaged as
`killarney_runtime_fingerprint_20260901.tar.gz` with SHA256:

```text
d7872b7334e6acf16e33b3f42976c7985aa7c7ab46c40f7729d621c00089d0ea
```

`export_runtime_fingerprint.py` runs the actual MME row-0 VisionZip-r010
generation path and captures:

- the exact processor-produced `pixel_values` tensor;
- `image_grid_thw`, input IDs, and attention mask;
- the VisionZip visual attention score for every original visual token;
- dominant indices in score order and model sequence order;
- contextual anchor indices;
- the final retained and dropped indices;
- contextual-token merge assignments;
- package paths/versions, source hashes, model revision, and raw rollout.

The tensor values are stored as portable `.npy` files. `fingerprint.json`
contains readable index lists, summaries, and both content and file hashes.

## Export on each machine

Activate that machine's actual evaluation runtime first. On Killarney:

```bash
export VLM_ROOT=/project/6101803/enmingzz/ckpt_eval_trainenv/VLMEvalKit_armen51682
source /project/6101803/enmingzz/ckpt_eval_trainenv/env_train_runtime.sh

python /project/6101803/enmingzz/opsd/analysis/mme_first_sample_rollout_20260901/export_runtime_fingerprint.py \
  --mme-tsv /scratch/enmingzz/vlmevalkit_data/MME.tsv \
  --model-path /scratch/enmingzz/.cache/huggingface/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --output-dir /path/to/fingerprint_killarney
```

Run the same script after activating the Vulcan evaluator, changing only the
site-specific paths and output directory.

## Compare

```bash
python compare_runtime_fingerprints.py \
  /path/to/fingerprint_killarney \
  /path/to/fingerprint_vulcan \
  --output /path/to/comparison.json
```

Interpretation order:

1. `pixel_values.exact_equal=false`: preprocessing/runtime mismatch precedes VisionZip.
2. Equal pixels but unequal attention scores: model weights or vision-kernel mismatch.
3. Equal scores but unequal selected indices: top-k/tie-breaking mismatch.
4. Equal retained indices but unequal merge assignments: visual-key or merge mismatch.
5. All visual artifacts equal but rollout differs: inspect decoder logits/kernels.
