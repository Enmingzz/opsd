# Trajectory bottom-80% F evaluation

This campaign evaluates the completed `trajectory_bottom80_delta002` run at
step 10240. It exactly matches the trajectory top-20% campaign on model merge,
reasoning-mode prompts, VisionZip ratios, MME/MMStar/MathVista datasets, raw
prediction preservation, and strict MathVista post-processing.

The source adapter is the LLM-only, LoRA-dropout-0 checkpoint trained with
native VisionZip r010 and the r010-to-r012 projection-fraction probe.
