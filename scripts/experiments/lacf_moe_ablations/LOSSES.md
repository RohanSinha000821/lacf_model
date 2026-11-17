# LACF-MoE auxiliary-loss ablations — C0–C3

The owner requested these loss-only controls on 16 November 2026. Use the
unchanged main **three-expert** LACF-MoE and original 3 bona-fide + 5 spoof
concepts, not the four-expert model or revised banks.

| ID | Folder | Detection BCE | Semantic BCE weight | Bona-fide JS weight |
|---|---|---:|---:|---:|
| C0 | `C0_bce_only` (existing reference, linked) | 1 | 0 | 0 |
| C1 | `C1_bce_semantic` | 1 | 0.25 | 0 |
| C2 | `C2_bce_bonafide_js` | 1 | 0 | 0.10 |
| C3 | `C3_bce_semantic_bonafide_js` | 1 | 0.25 | 0.10 |

These C IDs are scoped to this loss study. They are distinct from the historical
C0–C4 layer-fusion diagnostics. C0 reuses completed source-selected main MoE
checkpoints/results: verify their identities and canonical source artifacts,
then link them without retraining, target rescoring or changing original files.

## What changes

Only the two auxiliary-loss switches/weights. All models still use all 35
relation features, including JS and both entropies, in the detection head.

- Detection: `BCEWithLogits(final spoof logit, binary label)`.
- Semantic: `BCE(sum(p_W[:, 3:8]), binary label)` over the original five spoof
  concepts. It does not require a TTS/VC subtype label.
- Bona-fide consistency: average `JS(p_W,p_C)` over bona-fide examples only;
  zero when a microbatch contains no bona-fide examples.
- Original `LACFLoss` handles all terms in FP32 under BF16 mixed precision.

C1/C2/C3 start fresh, with bit-identical initial model/prototype bytes and RNG
state within seed. Losses add no trainable parameters: **967,338** for every
condition, including C0. Frozen encoders are never loaded or executed.

## Fixed run and evaluation contract

F1–F4, seed 1234: **12 fresh fits and 12 dependent native P1 evaluations**.
Order C1→C2→C3, each F1→F2→F3→F4. Reuse the same verified frozen cache,
center+two fixed random up-to-10s training views, native-unpadded development,
CLAP repeat-padding, batch132/accum1, equal domain/class replacement sampling,
AdamW LR3e-4/weight decay1e-4/constant LR, max30/patience5. Strict source macro
EER selects checkpoints; ties retain the earlier epoch. The previous A/B and
main/SOTA source files and all active jobs remain unchanged.

The new trainer reuses the guarded core fitting/preflight/export loops through
explicit registry bindings confined to its own process. A new completion verifier
retains the core hash/canonical-score/calibration gates and strictly checks each
actual loss objective and full model configuration (instead of the core's fixed
BCE-only expectation). Bindings restore after exit or failure. The trainer's
START log filter replaces the inherited BCE-only label; config/history record
the exact objective and all raw loss components.

Each fit discards a bounded real-source GPU preflight and full source-dev reload
probe before fresh initialization. Selected full-source scores must reproduce
bit-exactly; canonical exports, exact calibration and completion bind advancement.
Evaluation waits for the producer's verified queue-status entry and matching
completion hash, then repeats all selected source scores bit-exact before target
cache access. Existing matching target features are checked and reused; no
encoder reruns or extra copies. Report EER/AUROC and raw/separate transductive
z-score APCER/BPCER/ACER at source **1%, 5%, 10%**, with 1,000 saved conditional
bootstrap resamples (seed2026). No target results choose a loss weight or recipe.

Each queue uses 1GiB GPU cap, two CPU threads and zero feature workers; initial
training-bank load requires >=75GiB available RAM and >=2GiB GPU free. Preserve
running SSL F3. No automatic retry/resume/overwrite, new seeds, external sets,
publication, or changes to main/SOTA code. Keep single-seed, prior-target-exposure,
BF16 rounding, extraction drift and short-recording crop-diversity disclosures.

## Files and outputs

- Model: `src/audio_deepfake_detection/experiments/lacf_moe_ablations/loss_model.py`
- Trainer: `scripts/experiments/lacf_moe_ablations/train_losses.py`
- Completion gate: `scripts/experiments/lacf_moe_ablations/verify_losses.py`
- Evaluator: `scripts/experiments/lacf_moe_ablations/evaluate_losses.py`
- CPU checks: `tests/experiments/lacf_moe_ablations/test_losses.py`

Training: `outputs/experiments/lacf_moe_ablations/losses/C1_bce_semantic/f1/1234/`, etc.
Evaluation: `outputs/experiments/lacf_moe_ablations/losses/evaluations/C1_bce_semantic/f1/1234/`, etc.
Launch/authorization/provenance/logs: `losses/launch/` and `losses/evaluations/launch/`.
Use their guarded `run.sh` launchers; underlying modules are
`scripts.experiments.lacf_moe_ablations.train_losses` and
`scripts.experiments.lacf_moe_ablations.evaluate_losses`.
