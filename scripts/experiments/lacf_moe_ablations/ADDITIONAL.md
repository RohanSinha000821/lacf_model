# Additional LACF-MoE ablations — B1–B6

The owner authorized completing the six previously defined additional controls
on 14 November 2026. This supersedes the older B-series hold for these controls
only. Use seed 1234, all four full-source folds: 24 fresh source-only fits,
followed by their fixed-checkpoint native P1 evaluations. Existing A0–A6 runs,
ongoing evaluations, SSL F2, stopped/resumable SSL F3 and all historical artifacts
remain unchanged. No architecture/recipe is chosen from target results.

| ID | Input/change | Head input | Trainable parameters |
|---|---|---:|---:|
| B1 `wavlm_embedding` | Three-expert WavLM → original adapter → normalized embedding; no CLAP | 512 | 998,820 |
| B2 `clap_anchors` | Frozen CLAP → original eight concept probabilities; no WavLM | 8 | 1,777 |
| B3 `anchor_pair` | Same three-expert WavLM and CLAP → `[p_W,p_C]` | 16 | 966,084 |
| B4 `agreement_difference` | B3 + `abs(p_W-p_C)` and `p_W*p_C`; omit JS/entropies | 32 | 967,140 |
| B5 `global_layer_routing` | Same three layer experts and full relation head; three global learned weights replace recording-dependent layer router | 35 | 963,498 |
| B6 `fusion_moe_only` | Uniform raw 13-layer mean → original adapter/relations → same A5 three-expert fusion MoE; no layer MoE | 35 | 675,604 |

All heads retain pre-linear LayerNorm/64→16→1/GELU/dropout0.2. The original
3 bona-fide +5 spoof concepts, templates, frozen prototype bytes, temperature0.07,
FP32 normalization and CLAP double normalization are unchanged. Detection BCE
only, no semantic/BF losses, no augmentation. Head/adapter/expert initialization
matches the reference where comparable; changed input blocks use isolated RNG.
B5 is the existing C3 global-routing math with fresh full-fold fits, not reuse
of inner-domain diagnostic checkpoints. Capacity changes remain disclosed.

## Files and reuse

- Model: `src/audio_deepfake_detection/experiments/lacf_moe_ablations/additional_model.py`.
- Trainer: `scripts/experiments/lacf_moe_ablations/train_additional.py`.
- Evaluator: `scripts/experiments/lacf_moe_ablations/evaluate_additional.py`.
- Tests: `tests/experiments/lacf_moe_ablations/test_additional.py`.

The new entry points bind the six fixed B-model names, factories, dimensions and
parameter counts in their own Python process only. They reuse the unchanged
core fitting/preflight/export/evaluation functions. Bindings are scoped and
restored on exit, tested including exceptions, and pinned in run provenance.
No guarded source file, active Python process or A-series registry on disk is
changed. This avoids copying a second training loop or creating a new framework.

## Fixed execution and gates

Reuse existing verified frozen source features (center +two fixed random up-to10s
training views; centered development) and completed matching target feature caches.
Preserve native-unpadded encoder extraction semantics, limited short-audio crop
diversity and historical batching drift. No encoder loading/execution/download.
Load/hash/check the source bank once; each fit gets only its fold's three source
train/dev domains, never its target partition.

Batch132/accum1, equal source-domain/class replacement sampling, original epoch
recording count, AdamW3e-4/wd1e-4/constantLR, BF16 with sensitiveFP32 and reference
BF16-rounded logits, max30/patience5, strict macro source-dev EER/ties earlier.
Seeds remain1234 for this full-fold comparison; prior three-seed diagnostics are
unchanged. Featureworkers0, CPUthreads2,1GiB CUDA cap, >=2GiB GPUfree and >=75GiB
availableRAM before loading the source bank. Wait for resources without stopping
other jobs. Do not add workers to an in-memory head-only loader.

Each fit runs the unchanged discarded source-only finite-update/memory/full-dev
reload preflight. Fresh initialization follows; pilot weights are not used.
Selected models must reload with bit-exact full source scores, canonical exports,
exact source calibration and bound completion before advancing. No resume,
automatic retry or overwrite.

The separate evaluator waits for the PRODUCER'S VERIFIED queue-status entry,
not just a newly written completion filename. A failed/dead producer stops it.
It binds the selected checkpoint/completion after source training, reproduces
all source-dev scores freshly and exactly before target access, checks target
producer/crops/revisions/IDs/labels/payloads, then uses shared metrics. Keep raw
and separate transductive z-score1/5/10 APCER/BPCER/ACER,EER/AUROC and1000
conditional bootstrap intervals(seed2026). No target-guided design changes,
new seeds/external sets or release is authorized by this launch.

## Locations and commands

Training: `outputs/experiments/lacf_moe_ablations/additional/B1_wavlm_embedding/f1/1234/`, etc.
Launch/log/status: `additional/launch/`.
Evaluation: `additional/evaluations/B1_wavlm_embedding/f1/1234/`, etc.
Evaluation launch/log/status: `additional/evaluations/launch/`.

Use the guarded launch scripts; the underlying entry points are:

```bash
python -u -m scripts.experiments.lacf_moe_ablations.train_additional
python -u -m scripts.experiments.lacf_moe_ablations.evaluate_additional
```

Fixed B1→B2→B3→B4→B5→B6 order, each F1→F2→F3→F4. Training starts before
the evaluation waiter. This is exploratory after prior target exposure, not an
untouched method-selection study. Existing scores/results are not rescored.
