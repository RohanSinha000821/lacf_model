# Experiment folders

Primary SOTA models remain under `src/audio_deepfake_detection/sota/` and their
existing script/output families. Primary LACF stays under `lacf/`; its centered
source-development cache `outputs/lacf_frozen_dev_features/` remains there too.

| Work | Source | Scripts | Tests | Outputs |
|---|---|---|---|---|
| LACF diagnostics, probes and ablations | `src/audio_deepfake_detection/experiments/lacf_diagnostics/` | `scripts/experiments/lacf_diagnostics/` | `tests/experiments/lacf_diagnostics/` | `outputs/experiments/lacf_diagnostics/` |
| Layer fusion and readout controls | `src/audio_deepfake_detection/experiments/layer_fusion/` | `scripts/experiments/layer_fusion/` | `tests/experiments/layer_fusion/` | `outputs/experiments/layer_fusion/` |
| Token fusion and encoder adaptation | `src/audio_deepfake_detection/experiments/token_fusion/` | `scripts/experiments/token_fusion/` | `tests/experiments/token_fusion/` | `outputs/experiments/token_fusion/` |

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
