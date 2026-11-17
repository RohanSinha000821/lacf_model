# Cached-feature LACF F4

Owner-authorized F4 switch, 11 November 2026. Run `python -m scripts.experiments.lacf_cached.train` only through the recorded local launch gates.

Results remain at `outputs/lacf10s_optimized_20261009/f4/1234/`, as requested. The interrupted live-encoder attempt is preserved separately; never overwrite or resume it. The configuration explicitly records cached execution: centered plus two fixed random native unpadded training views and centered development. These differ from the fresh random padded training inputs of original F1–F3. Original batch-one backend development scoring is retained. Do not claim identical fold recipes.

Reuse the existing A9 backend, original 3+5 prototypes, 35-D relations and all three losses, 661,479 fresh trainables. Frozen encoder features are validated once and only the last WavLM layer/CLAP arrays are loaded. Shared sampling, training, FP32-sensitive math, metrics and calibration remain canonical. Seed1234/batch132/AdamW3e-4/wd1e-4/constant/max30/patience5, strict source macro EER/ties earlier. No encoder training, MoE, target evaluation or automatic retries.

Saved checkpoints contain the adapter, classifier and fixed prototypes, not frozen encoder weights. Future evaluation must recognize this checkpoint format and reproduce the recorded source execution before target access; this trainer does not authorize or implement target evaluation.

## Completed F4 target evaluation

Owner authorized evaluation on14November. Use the guarded launch for
`python -u -m scripts.experiments.lacf_cached.evaluate`. The original source-only
run is preserved. Output: `outputs/lacf10s_optimized_20261009/evaluations/f4/1234/`;
logs: sibling `evaluations/launch/`. Fresh full-source scores must match exactly
before reusing the completed main-MoE ASV2019 target feature cache. Keep last-layer
WavLM input and native batch-one backend/BF16-sensitiveFP32 scoring. No encoder
execution, retraining, prototype/loss changes, overwrite or automatic retry.
Report every raw/transductive1/5/10 operating point and1000conditional intervals.
Cached F4 differs from original F1–F3 training views/padding; retain that disclosure.
