# Experiment folders

Primary SOTA models remain under `src/audio_deepfake_detection/sota/` and their
existing script/output families. Primary LACF stays under `lacf/`; its centered
source-development cache `outputs/lacf_frozen_dev_features/` remains there too.
The main three-expert LACF-MoE now lives outside experiments under `lacf_moe/`
in source/scripts/tests and `outputs/lacf_moe/`. The old diagnostics and controls
remain here; shared cached source features are reused without relocation.

| Work | Source | Scripts | Tests | Outputs |
|---|---|---|---|---|
| LACF diagnostics, probes and ablations | `src/audio_deepfake_detection/experiments/lacf_diagnostics/` | `scripts/experiments/lacf_diagnostics/` | `tests/experiments/lacf_diagnostics/` | `outputs/experiments/lacf_diagnostics/` |
| Layer fusion and readout controls | `src/audio_deepfake_detection/experiments/layer_fusion/` | `scripts/experiments/layer_fusion/` | `tests/experiments/layer_fusion/` | `outputs/experiments/layer_fusion/` |
| Main LACF-MoE ablations | `src/audio_deepfake_detection/experiments/lacf_moe_ablations/` | `scripts/experiments/lacf_moe_ablations/` | `tests/experiments/lacf_moe_ablations/` | `outputs/experiments/lacf_moe_ablations/` |
| Token fusion and encoder adaptation | `src/audio_deepfake_detection/experiments/token_fusion/` | `scripts/experiments/token_fusion/` | `tests/experiments/token_fusion/` | `outputs/experiments/token_fusion/` |
| TF2 with frozen cached CLAP | `src/audio_deepfake_detection/experiments/tf2_frozen_clap/` | `scripts/experiments/tf2_frozen_clap/` | `tests/experiments/tf2_frozen_clap/` | `outputs/experiments/tf2_frozen_clap/` |

Diagnostic outputs are grouped into `step1/`, `frozen_features/`, `probes_step3/`
and `ablations_step4/`. Experiment definitions, recipes and saved records have
not been changed by this filesystem organization. Tests are discovered
recursively by the existing pytest configuration.

## Compatibility and live-job exception (6 November 2026)

Relative symbolic links preserve the old source and output paths used by imports,
launch commands and historical configurations. They point to the same files,
not duplicate datasets or checkpoints. Existing imports/default cache paths remain
valid through these links. Do not remove them: preserved run records contain
absolute paths and source hashes. A historical hash-bound verifier still requires
its recorded code revision; the three inactive scripts whose ROOT depth changed
have original snapshots in the ignored migration record. Saved run records were
not rewritten to claim a different code revision or location.

TF2 full-F1 recovery and SSL F4 are active. Their source guards remain intact.
The TF2 source package, diagnostic `probes.py`/`diagnostic_cache.py`, shared
`train_probes.py`/`train_ablations.py`, token-fusion scripts, and TF2's active
run/launch folders stay physically at their original locations. The new
experiment paths link to them. Primary LACF components imported by experiments
remain in `lacf/`, without copying them.

Inactive source modules/tests/completed outputs were moved. Inactive `diagnose.py`,
`cache_diagnostics.py` and the layer-fusion launcher were moved, with only their
repository-root parent depth adjusted. Compatibility links retain their old paths.

## Deferred checklist

At the next post-training cleanup, first verify TF2/SSL completion and all source
guards. Then finish physical relocation of active/shared experiment files and the
active TF2 output directories, update imports/root calculations/provenance paths
for future runs, and run CUDA-hidden CPU checks again. Keep historical compatibility
paths and original code snapshots where recorded provenance requires them. Do not
stop/restart a job, change a recipe or launch evaluation as part of this cleanup.
This checklist is also in the local handoff; no timed reminder is scheduled.

## Output naming and cached-CLAP preparation (7 November 2026)

New experiments use concise descriptive method names; use fold/seed folders
and a meaningful suffix for any separately authorized repeat. Existing folders
keep their historical names. The new tf2_frozen_clap implementation is physically
under experiments/ (no new compatibility links). Preparation is CPU-only;
training/GPU validation are not launched. WavLM LoRA and TF2 train; matched
CLAP embeddings are read from the preserved source cache without loading CLAP.
See [its preparation guide](../scripts/experiments/tf2_frozen_clap/README.md).

## Adapted TF2 F1 evaluation

The isolated `scripts/experiments/token_fusion/evaluate_adapted.py` entry point
uses the preserved compatibility folder (physically `scripts/token_fusion/`).
It evaluates the selected full-F1 WavLM-LoRA plus adapted-CLAP TF2, after the
existing trainer's complete source reload and a bounded fresh evaluator check.
Outputs are `outputs/experiments/token_fusion/evaluations/tf2_adapted/f1/1234/`;
launch/status/logs are its family's `launch/` folder. `best_adaptation.pt` and
`source_training_complete.json` explicitly retain sparse-state/original-training
provenance. Shared metric calculations are reused; active source files and
compatibility links are not altered. No target frozen features are reused for
these adapted encoders. Target results remain exploratory and single-seed.

## Fully frozen TF2 all-fold training

`scripts/experiments/token_fusion/train_all_folds.py` preserves original F1 and
fits fresh F2→F3→F4, seed1234, using both frozen branches' existing cached
features and the unchanged original TF2 model/training math. Outputs live under
`outputs/experiments/token_fusion/tf2_frozen/{fold}/1234/`; `f1/1234` links to
the preserved original run. Its `launch/console.log` records live resource waits,
cache verification/loading, per-fold discarded GPU validation and training.
This is different from `tf2_frozen_clap`, which trains WavLM LoRA. No target
evaluation accompanies this new training queue. Compatibility links stay intact;
the new script physically resides under `scripts/token_fusion/`.

The owner separately authorizes F2–F4 evaluation through
`scripts/experiments/token_fusion/evaluate_frozen.py`. Its result family is
`outputs/experiments/token_fusion/evaluations/tf2_frozen/{fold}/1234/`, with
`launch/console.log` and queue records in that family. It reuses the completed
main MoE target caches after full source reload and producer/payload verification;
both frozen encoders remain unexecuted. Original TF2 F1 results stay preserved.
