# r010 trajectory-F partition with a 2-point intervention

This isolated pair repeats the trajectory top-20% versus bottom-80% F
partition ablation with native VisionZip budgets `0.10 -> 0.12`. It preserves
the initialization, 20k ordered dataset, rollout seed, LLM-only LoRA scope,
EMA teacher, optimizer, and 10,240-step training contract of the cancelled
`0.10 -> 0.175` runs.

The earlier 7.5-point outputs are not resumed or overwritten.
