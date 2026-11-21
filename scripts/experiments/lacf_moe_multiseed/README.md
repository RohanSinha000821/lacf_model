# Full-fold BF16 MoE replication

Compare the original three- and four-expert models on F1--F4 using training
seeds1234/1235/1236. Eight completed seed1234 cases are verified and reused;
only16 new fits/evaluations are run. This is a full-source-fold replication,
not inner-source validation. All original concepts, feature caches, objectives,
batch132, optimizer, source selection and BF16 scoring stay unchanged.

Run `python -u -m scripts.experiments.lacf_moe_multiseed.train` using the guarded
launcher under `outputs/experiments/lacf_moe_multiseed/launch/`. One sequential
worker shares the GPU with preserved embedding-path/SOTA jobs, without stopping
or resizing them. Source-only native GPU proof is discarded before each fit;
selected full-source score parity and fixed calibration gate cached target use.

Each training seed controls initialization/dropout, domain-class sampler and
logical view choice. Physical cached crop intervals remain identical across
seeds. No encoders execute. Actual final logits compute in BF16; parameters and
sensitive operations remain FP32. Source and target precision policies match.

New outputs append to existing three/four-expert family `<fold>/<seed>/`
folders; target evaluations append to the corresponding `evaluations/` family.
All seed1234 and FP32 results stay unchanged. Study progress/status and final
`summary.json` are under `outputs/experiments/lacf_moe_multiseed/`.

Final summaries use all three seeds, per-fold mean and sample standard deviation
(ddof1). Bootstrap CIs remain conditional recording-level intervals, separate
from training-seed variability. Means never pool target populations. No target-
guided selection of seeds, grouping or architecture. Four experts differ from
three in group boundaries and capacity as well as count. No automatic retry,
resume, overwrite, other model/seed/external evaluation or publication.
