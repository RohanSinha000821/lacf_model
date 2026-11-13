# LACF-MoE

Main working three-expert model. It retains both frozen audio encoders through
the verified source feature cache, original eight concepts (three bona-fide,
five spoof), temperature0.07, the768→512→512 adapter and full35-D relation head.
Layer groups R0–R3,R4–R7,R8–R12 feed residual experts and a recording-dependent
three-way router. Only detection BCE is enabled;967,338 parameters train.

The implementation reproduces the completed original three-expert model's
state keys, seeded initialization, routing, forward/backward math and backend.
The legacy `C4` string in serialized model metadata is retained for compatibility;
it does not mean four experts. No four-expert code is imported by this model.

Folders: `src/audio_deepfake_detection/lacf_moe/`, `scripts/lacf_moe/`,
`tests/lacf_moe/`, and `outputs/lacf_moe/<fold>/1234/`.
Completed F1 is linked from its preserved historical run; its artifacts and
provenance are not rewritten. F2–F4 start fresh, sequentially, after source-only
finite-update/full-source scoring/reload checks. No target evaluation follows.

All folds use cached13-layer WavLM/CLAP features, centered plus two fixed random
up-to10s training views and centered source dev, native unpadded encoder inputs.
The shared cache remains at `outputs/experiments/lacf_diagnostics/frozen_features/`;
it is reused rather than copied. Domain/class-balanced replacement sampling,
seed1234,batch132,AdamW3e-4/wd1e-4/constantLR,BF16 with sensitiveFP32,max30/
patience5,strict equal-domain macro source EER and earlier checkpoint on ties
match original MoE F1. Feature batches need0workers and2CPUthreads.

Run only through the prepared guarded `outputs/lacf_moe/launch/run.sh` in tmux.
It preserves F1, verifies its complete source score vector using this model,
loads/checks source feature payloads once and passes only each fold's six source
partitions to training. Each fold exports fresh source scores/calibration,
routing diagnostics and hash-bound completion before advancing. Existing runs,
failed launches or checkpoints are never resumed, retried or overwritten.

This remains an exploratory study after prior target exposure. Adopting the
model as the main working direction does not make earlier target-informed
comparisons independent method selection. No new target tuning or future
ablations are authorized by the training queue.

CPU checks:

```bash
source activate.sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q tests/lacf_moe/
```

## Native P1 evaluation

Owner authorized all four completed main three-expert folds on12November2026.
The guarded `outputs/lacf_moe/evaluations/launch/run.sh` runs
`python -u -m scripts.lacf_moe.evaluate` in tmux `lacf-moe-eval`.
It verifies/reuses the original completed F1 evaluation through a relative link,
then evaluates F2(CFAD unseen)→F3(ASV5 eval)→F4(ASV2019 LA eval), stopping on
any failure. Source completion/calibration verification and fresh full-source
checkpoint scoring gate every new target. The main three-expert model is always
instantiated; existing source-reload/encoder-check helpers are shared with the
older evaluation implementation without using its four-expert model.

New native-unpadded centered10s WavLM13-layer/CLAP features are retained in each
`outputs/lacf_moe/evaluations/<fold>/1234/target_features/`. Source thresholds stay
frozen. Reports include EER/AUROC, raw and separately transductive z-score
APCER/BPCER/ACER at source1%,5%,10% with1000 conditional bootstrap resamples
(seed2026). No target results select changes. The existing32/64 extractor uses a
7GiB allocator cap; source-only memory/drift checks stop before target access
on failure. Existing training jobs and historical results remain untouched.

Live log: `outputs/lacf_moe/evaluations/launch/console.log`. Per-fold
`progress.json` distinguishes extraction/scoring, metrics and completion.
`evaluation_complete.json` and the final queue `launch/complete.json` are created
only after verification; an interrupted attempt is never overwritten or retried.
