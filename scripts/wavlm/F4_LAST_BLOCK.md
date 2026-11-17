# Prepared WavLM-WA F4 — final encoder block

Owner decision, 16 November 2026: prepare F4 using only the final WavLM Base
Transformer block plus the existing weighted-average backend. Keep the existing
`wavlm_bs96` output family and model display name. This document records the initial preparation. The owner
later authorized the fresh F4 launch on16November; verify live progress in the
local handoff and run logs. The preparation record alone is not live status.

## Adaptation and unchanged recipe

- Pinned local `microsoft/wavlm-base` revision
  `efa81aae7ff777e464159e0f877d54eac5b84f81`; no downloads or previous-fold weights.
- Train block11 (twelfth/final block), 13 layer weights and the 768→2 classifier.
- **7,089,955 trainable parameters**: 7,088,404 encoder + 1,551 backend;
  94,383,487 total. Freeze convolution/frontend/blocks0–10. Disable forced
  input gradients in the frozen convolution stack; preserve original prefix
  dropout/layerdrop in train mode and existing no-SpecAugment setting.
- Same weighted 13-representation forward, masked temporal mean and spoof
  logit difference. State-dict keys and the standard evaluator remain compatible.
- Same seed1234, batch96/accum1, random4s training crop/full unpadded batch1
  source development, 16/4workers, prefetch2, BF16, weighted CE9:1,
  Adam encoder2e-5/backend5e-3, ExponentialLR0.95, max100/patience5.
- F4 source train/dev only: ASV5, CFAD, SpeechFake. Their equal-domain macro
  source EER selects the strictly best checkpoint; ties keep the earlier epoch.
- Fresh selected-model full source scores must match the selection pass
  bit-for-bit. Export canonical scores/calibration/hash-bound completion/manifest.
- F1–F3 were fully fine-tuned. F4 has a different adaptation scope, explicitly
  recorded in config/checkpoints/completion and passed to future evaluation
  reports. A shared folder/display name does not imply identical training scopes.

## Measured GPU preparation

26 CUDA-hidden CPU tests passed, including actual Transformers frozen-prefix
forward/gradient/update checks, old inference/state compatibility, CLI scope,
source-score tamper rejection/export registration and existing queue behavior.

Bounded real source probe: three discarded batch96×4s weighted-CE optimizer
updates, with layerdrop disabled only in the discarded stress probe to execute
all12blocks. Final training retains original layerdrop. Frozen prefixes had no
gradient graph, and only final-block/backend parameters had gradients.

| Measurement | Peak allocated | Peak reserved |
|---|---:|---:|
| Batch96 training, 4s | 6.41GiB | 7.61GiB |
| Full129.75475s source recording, batch1 | 8.29GiB | 10.28GiB |

Fresh full-model reload reproduced the long source score bit-exactly. All probe
updates were discarded. Use a **14GiB PyTorch budget**, requiring at least16GiB
live free before launch; CUDA overhead is outside the allocator budget. This is
representative measurement, not an exhaustive bound on every possible duration.
Current SSL/MoE jobs were preserved.

Evidence: `outputs/wavlm_bs96/f4/preparation/source_validation.json`,
`gpu_check.log`, `cpu_checks.log`, `authorization.json`.

## Launch entry point — requires explicit owner authorization

```bash
bash scripts/wavlm/train_f4_last.sh
```

The owner-authorized F4 run uses this entry point. Do not launch a duplicate.
The launcher locks the existing WavLM family and refuses any existing F4 run.
The trainer requires matching pinned source-preparation evidence and never
initializes from probe/past-fold weights. No automatic retry/resume/overwrite
or target evaluation. Output location:

`outputs/wavlm_bs96/f4/1234/`

Implementation: `sota/wavlm_wa_last.py`, existing `scripts/wavlm/train.py`
optional `--last-block-only`, `last_block_support.py`, `prepare_f4_last.py`.
The default full-fine-tuning route and all existing F1–F3 files are preserved.
