## Target evaluation (owner authorized 14 November)

`python -u -m scripts.experiments.lacf_moe_ablations.evaluate` runs completed
A1–A6 sequentially, each F1→F2→F3→F4, seed1234. A0 uses the unchanged completed
main MoE evaluations. Source training does not authorize future targets; this
launch has its own explicit owner authorization and bound provenance.

Before each target, require unchanged selected checkpoint/config/completion,
canonical source exports/exact calibration and fresh full-source bit-exact
reload with the original batch132/BF16-sensitiveFP32 readout. Reuse existing
verified target features; F1's historical cache reader and F2–F4's producer
reader verify crops/revisions/IDs/labels/shard payloads. Encoders are never
loaded or executed. Preserve BF16-rounded scores and the original concept bank.

New output: `evaluations/A1_clap_embedding/f1/1234/`, etc.; logs/progress
under `evaluations/launch/`. Copied original training metrics remain
`source_metrics.json`, keeping their original completion hash meaning. Export
shared raw/separate transductive z-score1/5/10 errors, EER/AUROC and1000
conditional bootstrap intervals. Existing checkpoints/results are not rewritten.
Each failed source/payload/completion gate stops the queue; no automatic retry.
Two CPU threads, zero feature workers,1GiB CUDA cap. The later B1–B6 authorization and their separate queues are documented in [ADDITIONAL.md](ADDITIONAL.md).
