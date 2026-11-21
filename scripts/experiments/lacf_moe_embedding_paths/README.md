# Full-fold MoE embedding paths

Eight sequential fits, seed1234: CLAP path and WavLM matched control within
each F1--F4 fold. Train all three source datasets; no inner rotations.

Original frozen cached WavLM13x768 and CLAP512 inputs feed the same three-expert
MoE, adapter, eight concepts and35relation features. Add LN512->Linear512,64
->GELU on normalized CLAP audio or normalized adapted WavLM; concatenate35+64
and classify99->64->16->1. Both1005418trainables, BCEonly. New modules/head
weights match across arms; original adapter/experts/router/head tail preserved
at fresh initialization. Encoder weights are neither loaded nor executed.

Same FP32 fitting recipe as main: batch132, AdamW3e-4/wd1e-4, constantLR,
max30/patience5, source macroEER, seed1234, original sampling/cached views;
TF32 disabled. Source GPU preflights are discarded. Fresh selected full-source
reload/export/calibration/completion gates each fit. Baseline references link
to existing completed mainFP32 runs. Historical results remain unchanged.

Entrypoint: python -u -m scripts.experiments.lacf_moe_embedding_paths.train
Use guarded outputs/experiments/lacf_moe_embedding_paths/fp32/launch/run.sh
in tmux lacf-moe-embedding-paths. Outputs:
outputs/experiments/lacf_moe_embedding_paths/fp32/<variant>/<fold>/1234/.
Checkpoints, source score CSVs, calibration, metrics, routing and manifests
are saved there. Target evaluation is not queued by this training launcher.

Existing verified source-cache partitions are reused, never copied/re-extracted.
Only current-fold partitions stay in this worker's RAM; shared partitions are
retained across folds and excluded ones released. CPU2/workers0/1GiB GPU tensor
cap. Resource gates preserve existing jobs. No automatic retry/resume/overwrite.


The owner subsequently authorized evaluation of these eight fixed fits alongside
the 60 FP32 MoE ablations. The shared worker is
`scripts/experiments/lacf_moe_ablations/evaluate_fp32.py`; progress is in
`outputs/experiments/lacf_moe_fp32_evaluations/launch/`. Per-fit results append
under this family's `fp32/evaluations/<variant>/<fold>/1234/`. Completed producer
entries and fresh exact full-source FP32 scores gate cached target reuse.
Source calibration remains fixed; no target tuning or encoder execution.


## Historical BF16 comparison

The owner also requested matching the historical BF16 training and scoring for
both paths across F1--F4, seed1234: eight fresh fits and native evaluations.
Entrypoint: `python -u -m scripts.experiments.lacf_moe_embedding_paths.train_bf16`.
Outputs: `outputs/experiments/lacf_moe_embedding_paths/bf16/<variant>/<fold>/1234/`;
target results: `bf16/evaluations/<variant>/<fold>/1234/`. Train then evaluate
one fixed case before the next. Source selection/calibration and target logits
use historical BF16 autocast, sensitive calculations remain FP32. The model
asserts actual BF16 logits; no score-casting shortcut or learned initialization.
Source full-score parity and original target-cache validators precede exports.
Historical and FP32 results are preserved, including precision disclosures.

At the start of this embedding-path comparison, full-fold MoE and embedding-
path fits used seed1234. The earlier1234/1235/1236 diagnostics were inner-source
studies, and frozen token-fusion controls had three full-F1 seeds. Later full-
fold MoE replication is documented separately in
[the multiseed study](../lacf_moe_multiseed/README.md). This BF16 embedding-path
queue itself adds no seeds and does not authorize a final precision selection.
