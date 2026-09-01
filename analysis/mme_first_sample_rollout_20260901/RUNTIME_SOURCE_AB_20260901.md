# Runtime Source A/B: MME Row 0

## Question

Does the post-archive change in the patched Qwen2.5-VL source explain the
different VisionZip-r010 rollout observed on Vulcan?

## Controlled setup

- Host: `kn142` (Killarney L40S)
- Slurm job: `5156820`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- MME row: `0`
- Image SHA256: `f39869e8e7fdd167b08a088f6c221a0f308a7b42c961bb8786e3dbb364a86e06`
- Decoding: greedy, `max_new_tokens=2048`, KV cache enabled
- Image bounds: `min_pixels=1003520`, `max_pixels=3211264`
- VisionZip r010: evaluator prune argument `0.95`

Only the patched evaluator source was changed between the two independent
processes. The model, image, prompt, Python environment, GPU, and decoding
settings were held fixed.

## Source conditions

| Condition | Qwen model SHA256 | FlashAttention SHA256 | Wrapper SHA256 |
|---|---|---|---|
| Archived eval (2026-08-24) | `97d0f146fc760162039a6883a9a0a787b7cafe3d550e737210b9194294f7b683` | `274a6a261437673efcb4ba76f4a78304ed6245e3a1f200376137090bdcb9c8de` | `a903369dd1c6df7749b3a53f2e847fef876a0e5c6ede4da21e445d4e6abce72d` |
| Current Killarney eval | `d8fd8a3d37be4b39cdedcc6f60b1e32706a50469dcdb9b9f59ca2263792e004b` | `125edba14f33026622ddbb147495718c26853cac8ba9d34d1d5cfddac286399e` | `a903369dd1c6df7749b3a53f2e847fef876a0e5c6ede4da21e445d4e6abce72d` |

The archived checkout was reconstructed from VLMEvalKit commit
`51682a6baab948d3dbb4b867a3eab178504ac3f5` using the verified patch bundle
from OPSD commit `25abe78`.

## Result

| Check | Archived source | Current source | Equal |
|---|---:|---:|---:|
| Initial visual tokens | 1296 | 1296 | yes |
| Retained r010 visual tokens | 128 | 128 | yes |
| Full generated tokens | 59 | 59 | yes |
| r010 generated tokens | 92 | 92 | yes |
| Full raw rollout | byte-identical | byte-identical | yes |
| r010 raw rollout | byte-identical | byte-identical | yes |

The r010 raw rollout in both conditions was:

```text
<think>
The logo features a large, stylized letter "A" in the center, which is part of the word "ANGIE'S". The word "ANGIE'S" is written in a cursive font, with the apostrophe indicating possession. The design is consistent with a brand name, and the apostrophe suggests it is a possessive form, likely referring to a person's name.
</think>
<answer>
Yes
</answer>
```

## Conclusion

The source-hash drift is real, but it does not explain the Vulcan rollout
difference for this sample. The model-file diff changes image-placeholder
handling to use the active image/video placeholder; for this single-image MME
input it is behaviorally equivalent. The next comparison must fingerprint the
preprocessed pixel tensor, model shard files, selected VisionZip indices, and
the logits at the first divergent generated token across machines.

