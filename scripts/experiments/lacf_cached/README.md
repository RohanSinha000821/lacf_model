# Cached-feature LACF F4

Owner-authorized F4 switch, 11 November 2026. Run `python -m scripts.experiments.lacf_cached.train` only through the recorded local launch gates.

Results remain at `outputs/lacf10s_optimized_20261009/f4/1234/`, as requested. The interrupted live-encoder attempt is preserved separately; never overwrite or resume it. The configuration explicitly records cached execution: centered plus two fixed random native unpadded training views and centered development. These differ from the fresh random padded training inputs of original F1–F3. Original batch-one backend development scoring is retained. Do not claim identical fold recipes.

Reuse the existing A9 backend, original 3+5 prototypes, 35-D relations and all three losses, 661,479 fresh trainables. Frozen encoder features are validated once and only the last WavLM layer/CLAP arrays are loaded. Shared sampling, training, FP32-sensitive math, metrics and calibration remain canonical. Seed1234/batch132/AdamW3e-4/wd1e-4/constant/max30/patience5, strict source macro EER/ties earlier. No encoder training, MoE, target evaluation or automatic retries.

Saved checkpoints contain the adapter, classifier and fixed prototypes, not frozen encoder weights. Future evaluation must recognize this checkpoint format and reproduce the recorded source execution before target access; this trainer does not authorize or implement target evaluation.
