# Project working instructions

## WavLM-WA F4 last-block training authorized (16 November 2026)

Owner now requests starting the prepared fresh F4 seed1234 run. This overrides
preparation-only hold for this final fit only. Use scripts/wavlm/train_f4_last.sh
and the passed current source-validation record, pinned official local Base,
7,089,955trainables (block11+WA weights/classifier), unchanged batch96/4s/BF16/
Adam2e-5+5e-3/ExponentialLR.95/CE9:1/100max/patience5/16train4devworkers/prefetch2.
Train ASV5/CFAD/SpeechFake only; source macroEER selects, exact full-source
selected-score reload/export/calibration/completion gates final success.
14GiB allocator cap after16GiB livefree. Fresh official initialization; no
probe/other-fold weights, resume/retry/overwrite or automatic target evaluation.
Keep output outputs/wavlm_bs96/f4/1234/ and WavLM-WA name, explicitly disclosing
last-block F4 vs full-FT F1–F3 in all saved configs/reports. Preserve running
SSL F3 and both MoE C-loss queues/source guards. No other fit, target/external
access, recipe change, download, commit or push.

## WavLM-WA F4 last-block preparation only (16 November 2026)

Owner requests preparing F4 with only final WavLM Transformer block/backend
trainable, using the existing wavlm_bs96 family/f4/1234 path. Do NOT launch
final training. Source-only CPU and discarded bounded GPU memory/reload checks
are within preparation; no target access. Keep the existing seed1234,batch96,
4s train/full batch1dev,CE9:1,Adam2e-5encoder/5e-3backend,ExponentialLR.95,
max100/patience5,workers16/4/prefetch2/source macro EER/ties earlier recipe.
Pinned official local WavLM Base efa81aae7ff777e464159e0f877d54eac5b84f81;
train block11+13layerweights+classifier:7089955trainable/94383487total.
Prefix parameters frozen; forced frontend input gradients disabled, existing
stochastic prefix behavior retained. F1–F3 are full-FT; disclose F4's last-block
scope in config/checkpoint/completion/evaluation despite shared name/folder.

New wavlm_wa_last.py and source-only support/preparation/launch files extend
existing trainer via an explicit prepared F4-only option. Preserve all active
SSL/MoE guards/jobs. Future training requires matching proof and fresh official
initialization, selected-checkpoint full-source score bit-exact reload/export/
calibration/completion before any later separately authorized evaluation.
Measured batch96train reserved7.61GiB; full129.75s source reserved10.28GiB;
14GiB allocator cap after16GiB live free. Representative, not exhaustive length
bound. Preparation passed26CPU tests and real source stress/reload checks.
See scripts/wavlm/F4_LAST_BLOCK.md. No final fit/evaluation, download or push.

## LACF-MoE C1–C3 loss ablations authorized (16 November 2026)

Owner requests BCE plus semantic alone(C1), bona-fide JS alone(C2), and both(C3).
Reuse original weights0.25semantic/0.10BFJS and unchanged main THREE-expert
MoE/35-D relations/3+5prototypes/967338params. Current IDs are scoped to losses,
not old C-layer diagnostics. Reuse verified existing BCE-only reference asC0;
12fresh F1–F4 seed1234 source fits and their12fixed-checkpoint native P1
evaluations. Same cached features/views/initialization/RNG/sampling/batch132/
AdamW3e-4/wd1e-4/constant/BF16-sensitiveFP32/max30/patience5/source macroEER.
No architecture change, extra controls/seeds, loss-weight/target-guided tuning.

New loss_model.py/train_losses.py/verify_losses.py/evaluate_losses.py stay in
experiments/lacf_moe_ablations; existing main/core/SOTA source bytes stay fixed.
Explicit process-local registries reuse core fitter/preflight/scorer/export.
Loss-specific completion verifier retains strict artifact/canonical source and
calibration gates, checks actual objective and whole model config. Discard each
real-source GPU finite-update/full-source reload probe before fresh fit. Require
selected full-source bit-exact reload/export/exact calibration and producer's
verified queue-status completion before evaluation; fresh full source check before
verified matching cached target reuse. No encoders executed/downloaded. Preserve
BF16 logits,raw/separate transductive1/5/10errors/EER/AUROC/1000bootstrapseed2026.

Outputs under experiments/lacf_moe_ablations/losses/, evaluation siblings in its
evaluations/. Eachqueue1GiBcap/2CPUthreads/featureworkers0; train bank requires
75GiB available RAM/2GiB GPUfree. Preserve active SSL F3 and previous artifacts.
Stop on failure, no auto retry/resume/overwrite, external data or publication.
See scripts/experiments/lacf_moe_ablations/LOSSES.md for definitions and caveats.

## SSL-last F2 native evaluation authorized (16 November 2026)

Owner requests evaluation of completed ssl_gating_last F2 seed1234, selected
epoch7/source macro EER5.7544199%, final10. This supersedes the earlier F2
target hold for this fixed checkpoint only. New evaluate_last_f2.py preserves
guarded F1/F3/shared files and reuses the same source/config/parity/scoring
helpers. Require successful producer exit, checkpoint/config/source-score/
calibration hashes, canonical source labels/counts/exact calibration, and fresh
first/tail plus129.75475s SpeechFake source score parity before CFAD unseen.
Full unpadded batch1/BF16-sensitiveFP32,4workers/prefetch2/CPU2,8GiB cap after
10GiB live free; no precision/input fallback. Retain EER/AUROC and raw/separate
transductive z-score APCER/BPCER/ACER at1%,5%,10%,1000conditional bootstraps
seed2026. Append evaluation artifacts to F2/1234 with launch proofs in
outputs/ssl_gating_last/f2/evaluation/. Preserve running F3 and all other results.
Stop on failure without automatic retry/resume/overwrite, other-fold/model or
external evaluation, training changes, downloads, commit or push.

## Additional MoE ablations B1–B6 authorized (14 November 2026)

Owner requests completing the remaining predefined additional controls. Run
B1–B6 on F1–F4 seed1234:24 fresh cached-feature source fits and their verified
native P1 evaluations. This overrides earlier B-series holds only. Definitions,
parameter counts, process-local reuse bindings and commands are in
scripts/experiments/lacf_moe_ablations/ADDITIONAL.md. No new controls or seeds.
Keep original core scientific recipe, prototypes, views, score precision,
source selection/calibration and shared raw/separate transductive reporting.

New additional_model.py, train_additional.py and evaluate_additional.py leave
all guarded core/main/SOTA files unchanged. Reuse existing fitting/evaluation
functions through scoped model-registry bindings in NEW PROCESSES only; record
and test those bindings, never mutate another process or original output.
Separate outputs under experiments/lacf_moe_ablations/additional/.

Discard source-only preflight weights; fresh fits, exact selected full-source
reload/canonical calibration/completion gate advancement. Evaluate only after
producer's verified queue-status entry, binding that checkpoint/completion;
repeat fresh source checks before verified cached target access. No target
feedback selects a model/recipe. Preserve all extraction/BF16/seed caveats.
Each queue1GiB GPUcap/2CPUthreads/featureworkers0; source-bank load requires
75GiB availableRAM and2GiB GPUfree. Wait rather than interfere with jobs.
Core evaluations/SSL F2 remain running; SSL F3 stays stopped/resumable.
No automatic retry/resume/overwrite, encoder execution/download, external
sets, further architecture, commit or push. Stop and preserve failures.

## Original LACF F4 and MoE-ablation target evaluation (14 November 2026)

Owner authorizes native P1 evaluation of completed original cached LACF F4
and MoE-A1–A6 on F1–F4, seed1234 (24 evaluations). Reuse the already completed
main three-expert A0 evaluations; additional B1–B6 remain held. This supersedes
the earlier target hold only for these fixed checkpoints. No target-guided tuning.
Preserve active SSL-last F2; SSL-last F3 is stopped at owner request with its
recovery checkpoint intact and must not restart automatically.

New evaluators stay in scripts/experiments/lacf_cached/ and
scripts/experiments/lacf_moe_ablations/. Reuse verified matching existing target
features, including the historical F1 format, without encoder execution/downloads.
Before each target read, bind launch/checkpoint/completion/cache identities,
verify canonical source exports and exact calibration, and freshly reproduce
ALL selected source-development scores bit-exact. Keep original F4 head batch1
and last WavLM layer/all three training losses; ablations retain their exact
heads/batch132/detection-only recipe and original3+5 concepts. Do not rescore
old models in a different precision. BF16-sensitiveFP32/source thresholds,
raw and separately transductive z-score ALL1/5/10 APCER/BPCER/ACER,EER/AUROC,
1000 conditional bootstraps(seed2026). Preserve historical extraction drift,
BF16 rounding, target exposure and original-F4 recipe disclosures.

Two guarded queues may share GPU with SSL F2, each capped1GiB with2CPUthreads,
featureworkers0. Retain source-dev arrays only; do not duplicate audio/features.
Stop on any failure; no retry/resume/overwrite, additional training/ablations,
other target/external evaluation or automatic commit/push. Evaluation artifacts
are new siblings, preserving every original training record. Originals under
outputs/lacf10s_optimized_20261009/evaluations/f4/1234/; ablations under
outputs/experiments/lacf_moe_ablations/evaluations/<variant>/<fold>/1234/.

## Parallel resumable SSL-last F3 (14 November 2026)

Owner requests fresh ssl_gating_last F3 alongside preserved F2, with recovery.
New train_last_f3.py and sota/ssl_gating_resume.py leave all guarded F2/shared
sources unchanged. Seed1234; ASV2019/CFAD/SpeechFake sources only, ASV5 target
unopened. Same Last model/block23+backend/15241090params,batch24/BF16-FP32,
Adam3e-6/wd1e-4/constant,CE+CKA1,20s random train/fullbatch1dev,max10/patience3,
strict source macro EER/ties earlier,train8/dev4workers/prefetch2/CPUthreads2.

Source GPU validation proves batch24/all24-layer stress fits a13GiB allocator
cap with max_split_size_mb:128 and>=15GiB livefree, alongside F2. Deterministic
algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 give exact actual-source
optimizer/RNG update replay and full129.75475s source reload. Expandable
segments are unsupported here; preserve that failed probe and the earlier
non-deterministic replay failure. No batch/precision/LR/architecture fallback.

Fresh final weights start from pinned official XLS-R/fresh backend, discarding
all source probes. Checkpoint model/optimizer/full selection/best state/loss
sums/epoch stage/sampler+crop seeds/CPU,CUDA,NumPy,Python RNG every15min and at
phase/epoch boundaries, atomic+fsync. Explicit --resume only, never automatic.
Replay only unprocessed optimizer draws; repeat an interrupted dev pass without
retraining a completed training phase. Per-draw crop RNG replaces legacy worker
RNG assignment for reproducible views, preserving the uniform crop and sampler
laws; disclose this execution difference. Current F2 is NOT retrofitted.
Stop on failure; preserve incomplete temp/checkpoint/log evidence and refuse
completed/foreign/config-mismatched recovery or simultaneous duplicate trainers.
No F3 target/other-fold evaluation, extra seeds, downloads or publication.
See docs/SSL_LAST_RECOVERY.md and F3 launch records for actual status.

## SSL-last F2 training and F1 evaluation (13 November 2026)

Owner explicitly requests fresh ssl_gating_last F2 seed1234 training and native
F1 evaluation. Preserve the completed F1 run (selected epoch10/source EER3.525555%)
and active cached MoE ablations. Do not edit guarded F1/shared sources. New
entry points train_last_fold.py (--fold f2 only) and evaluate_last.py reuse
the exact LastSSLFeatureGating/training/scoring math. F2 starts from pinned
official XLS-R/fresh backend after discarded source GPU validation, never F1
or partial/interrupted weights. Only block23/backend train (15241090params);
batch24/BF16-FP32,Adam3e-6/wd1e-4/constant,CE+CKA1,20s train/full batch1dev,
max10/patience3/strict macro source EER/ties earlier,8/4workers/prefetch2 remain.
Canonical train/dev membership/existence/bounded decode and pinned weight
hashes must pass; real20s/24batch all-layer stress, representative source updates
and full129.75475s source reload gate final fresh training. CUDA cap19GiB after
>=21GiB live free; preserve MoE and evaluation memory. No resume/retry/overwrite.

F1 evaluation binds original best/config/completion/source calibration/exports,
verifies canonical source labels/counts and exact frozen calibration, then
fresh first/tail and long60s full source score parity before SpeechFake test.
Full recordings/batch1/BF16-FP32 score subtraction,4workers/prefetch2/CPU2,
8GiB allocator cap after>=10GiB live free. Retain all raw/separate transductive
1/5/10% APCER/BPCER/ACER,EER/AUROC and1000conditionalbootstrap(seed2026).
Append evaluation artifacts without rewriting training records; stop on failure.
No F2 target, other fold/seed/model evaluation, extra training, target tuning,
downloads, changed recipe, publication or interference with existing jobs.

## Reduced LACF-MoE ablations authorized (13 November 2026)

Owner requests sequential core IDs, separate additional controls, and launch of
the reduced cached-feature study. Use only experiments/lacf_moe_ablations in
source/scripts/tests/outputs. Core A0=verified existing main three-expert MoE;
A1=CLAP embedding-only,A2=MoE WavLM+CLAP embedding concat,A3=WavLM anchors-only,
A4=uniform13-layer mean/full relations,A5=layer+relation-fusionMoE,
A6=same fusion experts/global learned fusion weights. New study IDs are scoped
as MoE-A0–A6, distinct from original-plan A0–A10 and Step4. Additional B1–B6
are definition-only and never queued. No old outputs are renamed or rewritten.

Fit A1→A6, each F1→F2→F3→F4, seed1234 only (24fresh fits), verifying/reusing
A0's four completed references. Reuse original3+5prototypes/frozen13-layer
WavLM+CLAP cache; no encoder execution. Keep batch132/accum1,AdamW3e-4/wd1e-4/
constant,BF16-sensitiveFP32/reference-rounded logits,detectionBCEonly,max30/
patience5/strict equal-domain macro source EER/ties earlier and exact source
reload/canonical calibration. A5fusion: three residual35→64→35 experts over
the full relation vector, LN35→Linear35,3/softmax router; A6global3logits.
Parameter counts A0–A6:967338,35041,1032612,965556,661479,981463,981288.

CUDA-hidden CPU checks precede bounded real source finite-update/full-source
reload checks for each fit. Discard preflight models; final models start fresh.
Check cache payloads/canonical source membership once, pass only the current
fold's source train/dev partitions into fitting. Source-selected fresh reload,
export/calibration/completion gate advancement; stop on failure without
retry/resume/overwrite. Feature workers0,CPUthreads2,1GiB CUDA allocator cap,
>=2GiB live free and>=75GiB RAM before initial cache load. Preserve SSL-last,
all other jobs, guards, historical artifacts and compatibility links.

Source training only: no target evaluation, extra seeds/additional ablations,
rescoring past results, encoder downloads, source feature regeneration or
publication. This is exploratory after prior target exposure; no target-guided
choices. Existing diagnostic seed plans are unchanged. Definitions/code/launch
contract: scripts/experiments/lacf_moe_ablations/README.md.

## Frozen TF2 F2–F4 evaluation authorized (13 November 2026)

Owner requests evaluation of completed fully frozen TF2 F2→F3→F4 seed1234.
The training queue completed with selected-checkpoint source reload/export/
exact calibration verification; reuse its completed models and preserve F1.
This overrides the preceding TF2 target hold for these three fixed runs only.
Use isolated scripts/experiments/token_fusion/evaluate_frozen.py (physical
scripts/token_fusion/ through the preserved compatibility folder). Reuse the
completed main MoE target feature caches; do not execute/download encoders or
duplicate caches. Verify the original source-cache identity, all training and
producer completions/hashes, canonical IDs/labels, centered10s/native-unpadded
crop policy, revisions, precision and each reused feature payload. Fresh full
TF2 source-dev score reload must be bit-exact before any target cache access.
Keep selected source thresholds/statistics, original head batch132/BF16-sensitive
FP32/scalar spoof logit and shared raw/separate transductive z-score1/5/10%
APCER/BPCER/ACER,EER/AUROC,1000conditional bootstrap(seed2026).

Output outputs/experiments/token_fusion/evaluations/tf2_frozen/{fold}/1234/;
launch/provenance/logs in its launch/. Bound all checkpoint/completion identities
at launch, stop on failure, no automatic retry/resume/overwrite. Head-only1GiB
allocator budget after>=2GiB live free;2CPUthreads/featureworkers0, one source
dev partition at a time. Preserve running SSL-last and all existing artifacts,
source guards and compatibility links. No training, variants, seed selection,
external-set access, target-guided changes or publication. Comparisons remain
single-seed exploratory after earlier target exposure.

## Fully frozen TF2 remaining folds (13 November 2026)

Owner requests all four full-source folds of the original fully frozen TF2
Transformer, whose F1 three-seed mean target EER was15.26%. This is NOT the
LoRA/adapted-CLAP run or the WavLM-LoRA/frozen-CLAP comparison. Preserve the
completed original F1 seed1234 (16.13% target EER); fresh F2→F3→F4 use seed1234
to match the current MoE comparison. No architecture or recipe selection from
target results. Keep the original TF2 model and fitting/export functions
unchanged; new sequential entry point scripts/experiments/token_fusion/
train_all_folds.py binds the canonical fold's sources within its own process.
No other running process or guarded source bytes change.

Both audio encoders remain frozen: use the existing verified cached13-layer
WavLM Base+ and CLAP projections, center+two fixed random10s train views and
centered native-unpadded source development. Only the original4,907,745-parameter
shared adapter/two-block Transformer/head trains, detectionBCE only, batch132/
accum1,AdamW3e-4/wd1e-4/constant,BF16-sensitiveFP32,max30/patience5/strict
equal-domain macro source EER/ties earlier. No prompts or auxiliary losses.
Per-fold discarded finite-update/full-source fresh reload GPU checks precede
fresh initialization. Verify final selected reload/exports/exact source
calibration before advancing. Preserve original F1 via a relative output link
and full-source bit-exact reload; no retrain/overwrite/resume/retry.

Keep new implementation/tests/outputs under experiments (the script compatibility
folder still physically resides under scripts/token_fusion/). Output family
outputs/experiments/token_fusion/tf2_frozen/{fold}/1234/; launch/status/logs in
its launch/. Load source cache once after live RAM >=62.6GiB arrays+12GiB reserve
and GPUfree>=5GiB, using a4GiB allocator cap,2CPUthreads/featureworkers0. Leave
all other jobs intact; wait for actual available memory, not arbitrary job exit.
No target evaluation, further seeds/variants, downloads or publication. These
full-fold runs remain exploratory after earlier target exposure; do not claim
inner-domain selection or independent confirmation.

## Adapted TF2 F1 native evaluation (owner request, 13 November 2026)

Owner authorizes evaluating the completed source-selected adapted full-F1 TF2
seed1234 on SpeechFake test. Queue behind its CURRENT final source reload/export
and actual producer exit_status.json; do not interrupt or repeat that full180194
source-audio verification. Require completion, canonical source IDs/labels/counts,
all score/checkpoint/config hashes, exact frozen calibration and pinned sparse
checkpoint reconstruction. A fresh evaluator first/tail source-batch bit-exact
check and actual22×10s source inference/memory check precede target access.
Bound the selected config/checkpoint identity at launch. If any gate fails,
stop without retries, changed precision/batch, skipping recordings or target access.

Use existing official local-only pinned bases plus ALL selected WavLM Q/V LoRA,
CLAP final-stage/projection and TF2 weights,20,049,185trainables in the trained
model. Both audio branches are recomputed; frozen-feature caches cannot stand
in for adapted encoders. Preserve native-unpadded WavLM grouping, centered common
up-to10s views/CLAP repeatpad,BF16-sensitiveFP32,batch22/workers4/prefetch2,source
thresholds and recovery provenance. Evaluation uses7GiB allocator cap only after
at least10GiB live free memory; preserve all other jobs. Report shared EER/AUROC,
raw and separate transductive z-score1/5/10% APCER/BPCER/ACER with1000conditional
bootstrap(seed2026); no target tuning. New evaluator through existing experiment
compatibility path scripts/experiments/token_fusion/evaluate_adapted.py; output
outputs/experiments/token_fusion/evaluations/tf2_adapted/f1/1234/. Sparse checkpoint
and original completion retain their actual names/format; do not fabricate a
standalone full-model checkpoint or training completion. No further training,
other folds/models, external evaluation, downloads or publication authorized.

## Four-expert live-memory scheduling correction (13 November 2026)

Owner points out free GPU memory while the four-expert queue is idle. Live
inspection confirms TF2 finished training at epoch17/patience5 and is verifying
its selected checkpoint, releasing optimizer memory: about8.6GiB is free while
SSL and the main MoE evaluation remain active. Replace ONLY the idle coordinator's
obsolete whole-encoder-process-exit wait with a live free-memory gate >=4GiB.
Preserve/terminate only that waiting coordinator (no children or started fits),
archive its launch records/source snapshot, then issue a fresh guarded F3→F4
continuation. No other process, source, model or recipe change. Existing trainer
retains1GiB allocator cap and source validation; evaluations retain their actual
completed-target-cache dependency and512MiB head cap. Preserve all F1/F2 runs,
calibration fix and failed history. New source/header fingerprints must explicitly
record this scheduler-only revision; do not weaken active job source guards or
create a fake producer completion. This supersedes the preceding whole-process
wait only. No resume/overwrite/retry of any fitted model, target tuning or push.

## Four-expert missing-fold completion (12 November 2026)

Owner explicitly requests fixing the stopped four-expert queue and completing
its missing folds without disturbing active processes. The failure was only
canonical CSV reduction order at F2 calibration; the reviewed sorted-ID fix
passes exact existing F1/F2 calibration and preserves all weights/results. Keep
F1/F2 and failed historical launch records unchanged. A fresh launch_f3_f4
runs the UNCHANGED trainer with --folds f3 f4, fresh seed1234 models, existing
four-expert cached-feature recipe and per-fold finite-update/full-source reload/
export/calibration gates. Do not resume/retrain F1/F2 or relax equality.

Only ~2GiB GPU headroom exists during active three-expert encoder evaluation.
The new continuation waits for existing four-expert F2 evaluation to succeed
and the heavy main evaluation process to exit before fresh cached F3→F4 fits.
No active job/source edit, stop, signal or priority change. Then evaluate F3→F4
using completed main-evaluation target caches, hash/crop/revision verification,
fresh full four-expert source reload and frozen source calibration. Preserve
shared raw/transductive1%,5%,10%,1000bootstrap(seed2026), no target tuning. Training
uses existing1GiB allocator cap; cached head evaluation512MiB. New coordinator
scripts/experiments/layer_fusion/complete_four_experts.py; records under training
and evaluation families' launch_f3_f4/. Missing/failed producer evidence stops
the dependent stage; no automatic retry, overwrite, recipe change or publication.
All work stays under experiments. This supersedes the previous F3/F4 training
hold for this explicit continuation only; external sets/other models stay held.

## Four-expert completed-fold evaluation continuation (12 November 2026)

Owner requests evaluation of the four-expert experiment across all completed
folds. Live inspection confirms ONLY F1/F2 trained: the old queue failed at F2
calibration verification before F3/F4. The reviewed sorted-ID verifier fix exists;
no F3/F4 four-expert training has subsequently run. Preserve F1's completed
metrics and failed historical queue records. Start F2 only in a fresh guarded
launch, using the main three-expert evaluation's completed CFAD frozen target
feature cache after its producer completion/crop/revision/hash checks. Run full
fresh four-expert source-score reload and exact calibration checks before target
scoring. Keep original four-expert model/thresholds/BF16-FP32/head132 and shared
raw/transductive1%,5%,10% metrics/1000bootstrap(seed2026). No encoder reload,
new training, F3/F4 placeholder results, automatic retries/overwrites, design
changes or publication. The cached head uses a512MiB allocator cap; preserve
active three-expert evaluation, SSL-last and TF2 source guards. New entry point:
scripts/experiments/layer_fusion/evaluate_four_experts_cached.py. Launch records:
outputs/experiments/layer_fusion/evaluations/lacf_layer_moe4/launch_cached_f2/.
F3/F4 evaluation remains blocked on absent training; report this explicitly.

## Main three-expert LACF-MoE evaluation (12 November 2026)

Owner explicitly authorizes native P1 evaluation of the completed main THREE-expert
LACF-MoE F1–F4, seed1234. This overrides older target holds for these four fixed
checkpoints only, including F4. Verify all source completions, checkpoint hashes,
canonical source calibration and execution guards. Preserve and link the completed
original F1 evaluation; fresh F2→F3→F4 evaluations require full bit-exact cached
source reload before target access. Reuse the existing source-validated batch32
WavLM/batch64 CLAP BF16 extractor, centered up-to10s/native-unpadded views and
head batch132. A source-only maximum-input encoder/memory check precedes each
new target extraction; stop on failure without precision/batch fallback, retry,
overwrite or any job interruption. Keep a7GiB allocator budget alongside preserved
SSL-last/TF2; verify live headroom. Retain target frozen features for reuse.
Report shared EER/AUROC, raw and separately transductive z-score APCER/BPCER/ACER
at ALL source1%,5%,10% settings with1000 conditional bootstrap resamples(seed2026).
No architecture/concept/loss/calibration changes, target tuning, external-set
access, ablations, further training or automatic publication. Main evaluator and
outputs: scripts/lacf_moe/evaluate.py and outputs/lacf_moe/evaluations/. Earlier
four-expert queues remain stopped and their artifacts unchanged. This remains
exploratory after prior target exposure, not independent method selection.

## SSL-gating last-block F1 (11 November 2026)

Owner requests a separate `ssl_gating_last` F1 seed1234 run alongside preserved
adaptedTF2 and main LACF-MoE. Train XLS-R block23 and the complete gating backend;
freeze frontend/blocks0–22. Reuse the approved partial recipe:batch24/accum1,
BF16 with FP32 pooling/losses/scores,Adam3e-6/wd1e-4/constantLR,weightedCE+CKA1,
training-only20s/full batch-one dev,max10/patience3/strict source macro EER/ties
earlier,workers8/4,prefetch2/CPUthreads2/19GiB cap. Start from pinned original
XLS-R and a fresh backend, never another run/probe checkpoint. New isolated
ssl_gating_last.py/train_last.py reuse shared partial math without editing live
sources. Minimal CPU tests, canonical source readiness and bounded real20s
memory/finite-gradient/native full-source-recording reload gates precede launch.
Preserve stochastic prefix execution from the earlier partial variant; only
trainable encoder scope changes. No new target evaluation, other folds/seeds,
resume/retry/overwrite, downloads, existing-job changes or publication.

## Main LACF-MoE all-fold study (11 November 2026)

Owner adopts the ORIGINAL THREE-expert LACF-MoE as the main working direction,
outside experiments: src/audio_deepfake_detection/lacf_moe/, scripts/lacf_moe/,
tests/lacf_moe/ and outputs/lacf_moe/. Preserve completed original F1 through a
link in the main output family; do not retrain or rewrite it. Run fresh F2→F3→F4,
seed1234, original groups R0–R3/R4–R7/R8–R12, original3+5 concepts, frozen cached
WavLM/CLAP, full35-D relation backend and detection BCE ONLY,967338trainables.
Retain batch132/AdamW3e-4/wd1e-4/constant/BF16-FP32/max30/patience5/strict
macro source EER/ties earlier. Validate exact reference implementation parity,
canonical source cache payloads once, full F1 source reload and each fresh fold's
finite-update/source-reload preflight. Pass only the current fold's source data
to fitting; fresh selection/reload/export/calibration must pass before advancing.
Preserve adaptedTF2 source guards and all older primary/experiment artifacts.
This overrides experiment-only placement and earlier three-expert holds for
these fresh fits. It does not restart four-expert queues or authorize target
evaluation, ablations, changed recipes/seeds, retry/resume/overwrite or publication.
All cached source views remain fixed10s center+two random/native-unpadded, as in
original MoE F1; do not silently substitute primary LACF's older live-crop recipe.
This is an exploratory main working method after prior target exposure; do not
claim independent method selection or new confirmatory generalization evidence.

## Four-expert LACF evaluation authorized (11 November 2026)

Owner requests evaluation of all four-expert folds if training is complete.
Evaluate seed1234 F1–F4 sequentially only after each selected checkpoint,
completion, canonical source scores/calibration and full fresh source reload
verify. F1 may reuse the preserved all-layer SpeechFake features; remaining
folds extract centered up-to10s native-unpadded WavLM13-layer and CLAP features
with the existing batch32/64 BF16 extractor after a source-only maximum-input
check. Preserve/cache the resulting features. Keep the four-expert architecture,
original3+5 prototypes,detectionBCE-only recipe and source thresholds fixed.
Use shared raw and separately transductive z-score metrics, all1/5/10% settings
and1000 conditional bootstrap resamples(seed2026). This overrides previous
four-expert target holds only; no redesign, target-guided selection, other
models/external evaluations, retries, overwrites, publication or training changes.
Implementation and outputs remain in experiments. Preserve active TF2/primary
LACFF4/MoE4 training and source guards. Exact archived F1 trainer plus reviewed
fold-only extension reconcile its historical trainer hash; all model/scorer
hashes must still match. Stop the evaluation queue on any failure.

## Four-expert LACF all-fold extension (11 November 2026)

Latest owner direction expands the four-expert study to ALL F1–F4, seed1234
only, with the already agreed architecture/recipe. F1 completed and must be
preserved, not rerun. Verify its completion and continue fresh F2→F3→F4 in the
same experiments/layer_fusion/lacf_layer_moe4 output family. Load the required
canonical source train/dev cache union once, pass only each fold's six source
partitions to its model, and start every fold fresh. Per-fold finite-update/
full-source reload preflight and selected-checkpoint reload/export/calibration
gate advancement; stop on failure with no overwrite/resume/retry. Preserve
cachedprimaryLACFF4/adaptedTF2 and active runtime guards. Preserve exact completed
F1 source before extending its isolated trainer. This overrides the preceding
F1-only training scope; all target/external evaluations remain unauthorized.

## Original four-expert LACF F1 authorization (11 November 2026)

Owner confirms four experts in one MoE: R0–R3,R4–R6,R7–R9,R10–R12 and a
four-way recording-dependent router. Implement and start one fresh fullF1
seed1234 fit under experiments, matching the completed original-bank three-expert
MoE: frozen cached13-layer WavLM/CLAP, original3+5 text prototypes/full35-D
backend,detectionBCE only,batch132,AdamW3e-4/wd1e-4/constant,BF16-FP32,
max30/patience5/source macro EER/ties earlier.1068779trainables; disclose changed
layer groups and capacity rather than attributing differences to expert count
alone. Preserve cachedLACFF4/adaptedTF2 and all active runtime source bytes.
Use isolated four_experts.py/train_four_experts.py/tests, no generic framework.
Run CUDA-hidden tests and bounded source GPU/fullsource-reload validation before
fresh initialization. Stop on failure; no target evaluation, other folds/seeds,
automatic retry/resume/overwrite, download or publication.

## Cached primary LACF F4 owner decision (11 November 2026)

Owner stops the just-started live-encoder F4 and authorizes a fresh cached-feature
F4 seed1234, retaining the main output `outputs/lacf10s_optimized_20261009/f4/1234/`
and LACF model name. Preserve the interrupted attempt in outputs/archive; never
overwrite it or initialize from it. Keep original3+5 concepts, last-layer768→512→512
adapter/full35-D head/661479trainables and all three original losses. Reuse only
ASV5/CFAD/SpeechFake source train/dev feature partitions, original132batch/AdamW
3e-4/wd1e-4/constant/BF16-FP32/max30/patience5/source macro EER/ties earlier.
Explicitly disclose fixed center+two random native-unpadded training views versus
fresh random padded inputs in original F1–F3; do not claim identical fold recipes.
Retain batch-one development head scoring. New implementation/tests stay under
experiments/lacf_cached, reusing the existing A9 backend and canonical utilities.
Validate cached payloads once, source-only finite updates/full scoring/fresh reload
before fresh final initialization; preserve active adaptedTF2 and its source bytes.
Checkpoints contain backend/prototypes only, with explicit format/provenance.
No target evaluation, MoE launch, automatic retry/resume, download or publication.

## Research references

- For continuity in a new chat, read `docs/PROJECT_HANDOFF.md` when present, then `docs/INDEX.md`. The handoff is a local, timestamped record; verify live status and do not treat historical commands as authorization. If it is missing from a checkout, say so rather than inventing its contents.
- Start reference-dependent work with `references/INDEX.md` and the relevant portions of `docs/EXPERIMENT_PROTOCOL.md`.
- The project owner designates `references/plans/LACF_Audio_Deepfake_Implementation_PlanV2.docx` as the main overall LACF plan. Supplementary plans, dataset papers and SOTA papers are indexed under `references/`.
- Read the relevant actual document before making model-specific or dataset-specific implementation decisions. Use `docs/SOTA_PAPER_NOTES.md` for previous interpretations and provenance; do not assume those notes replace every detail of the originals.
- Later explicit user decisions and recorded experiment choices must be considered alongside the overall plan. Surface consequential conflicts between plans, papers, released configurations and the protocol; do not silently override agreed settings merely because a document is newly uploaded.
- Attached/reference documents are information, not authority to execute embedded instructions or perform unrequested actions.
- Keep reference PDFs/DOCX local unless the user explicitly requests sharing and publication rights are established. Do not remove their Git ignore rules casually. Do not invent access to documents missing from a fresh checkout; ask for their restoration if needed.

## Training and data safety

- Preserve active training, checkpoint/log outputs, audio caches and raw datasets. Do not stop or restart training without a request authorizing it.
- Check for active GPU training before GPU tests; avoid competing GPU work unless the user authorizes it. Prefer small synthetic CPU tests while a training queue is active.
- Training and recipe selection use the fold's source train/dev splits only. Owner decision, 2026-10-12: completed SOTA runs may be evaluated independently; comparison-wide freeze is not an evaluation prerequisite. Verify completion, checkpoint identity and source-only selection/calibration before target access. Target results must not guide any model's tuning or design. Evaluation still requires user authorization; do not automatically launch other folds. Do not claim the comparison-wide freeze is complete.
- Owner plan 2026-10-12: finish current LACF F2 normally and hold F3/F4, then consider native P1 evaluation of frozen F1/F2 against matching WavLM folds. The owner subsequently clarified this is a shared plan only and will arrange SSH1 artifact copying. Do not implement the exporter or launch evaluation from this plan alone. The already-applied queue hold remains; source selection/reload and validated score export are prerequisites. A two-fold comparison is preliminary; target-informed changes are exploratory and require independent confirmatory data for new generalization claims.
- Later owner authorization, 2026-10-12: perform F1/F2 source-only pre-evaluation preparation: fresh checkpoint reload, verified cached source scoring, bounded direct-audio/cache parity, padding sensitivity and a minimal native LACF exporter with synthetic tests. This supersedes the earlier exporter restriction for preparation only. Do not launch target evaluation, retrain or resume F3/F4. Preserve the trained normalization and scoring recipe; a revised feature recipe requires a separate decision.
- Latest owner authorization, 2026-10-12: start native P1 LACF F1/F2 evaluation using the local /mnt/drive cache after the passed source preparation. Evaluate preserved seed-1234 checkpoints sequentially: F1 SpeechFake test, then F2 CFAD clean unseen test; stop on failure. This supersedes the prior target-evaluation deferral for these two folds only. Retain the original normalization/input/precision recipe, source calibration and shared metrics/CI; keep F3/F4 held. No retraining, new variants, other protocols or target-guided changes are authorized.
- Latest owner authorization, 2026-10-13: start fresh native 10 s LACF F3 only after source-only worker benchmarks and reviewed provenance reconciliation. Retain the scientific recipe; record the measured loading change from F1/F2. F4 stays held. This supersedes the earlier F3 hold only; no new target evaluation, retraining of F1/F2 or target-guided model changes are authorized.
- Latest owner authorization, 2026-10-14: evaluate completed LACF F3 on native ASVspoof 5 eval with up to all 32 available data-loader workers, after selected-checkpoint/source-calibration verification and bounded direct-audio/cache parity. Retain batch-one centered up-to-10 s scoring, BF16/FP32 policy and all shared metrics/CI. This supersedes the F3 target hold only; F4 training/evaluation stays held. No target-guided recipe changes are authorized.
- Latest owner direction, 2026-10-14: follow the staged information-loss investigation. Step 1 is authorized on preserved LACF F1/F2 source development and their already-evaluated targets, plus WavLM F1–F3 checkpoint layer weights; no retraining or scientific recipe changes. Keep F3 evaluation running. Hold WavLM F4 training, all F4/external target inspection and evaluation until a final model is frozen. Share each step before advancing; new design selection uses inner leave-one-source-out validation. Later trained diagnostics use seeds 1234/1235/1236; existing primary runs retain their recorded seed plans. Resolve the 3/8 wording by reporting bona-fide mass (3 concepts) and spoof mass (5 concepts), without changing anchors.
- Latest owner authorization, 2026-10-15: Step 1 results were shared; proceed with Step 2 source train/dev frozen-feature caching. Use the centred 10 s view plus two fixed random training crops, reusing identical short-recording views. Cache the projected WavLM frontend plus twelve Transformer outputs and the raw CLAP audio projection. Preserve native unpadded WavLM inputs; validate against the existing scorer before extraction. This cache is shared across subsequent seeds. For later inner-source probes/ablations the owner approves reusing AdamW LR 0.0003, weight decay 0.0001, batch 132, constant LR, maximum 30 epochs/patience 5 and the existing 64→16 MLP hidden layers, freshly initialized per inner split and seed 1234/1235/1236. Share each step's results before advancing; no automatic trained-stage queue, redesign or new target access.
- Owner speed request, 2026-10-16: accelerate Step 2 extraction while preserving completed features and the scientific/input recipe. Use the existing full encoder forward paths: equal-length unpadded WavLM batches and independently batched fixed-size CLAP processor features. Measure source-only throughput, memory and BF16 rounding changes; bit-exact batch-one output is not a prerequisite for this accelerated diagnostic cache. Retain original config/shards and record the new execution segment. Explicit feature-cache continuation is allowed; model-training resume remains unimplemented. No deeper encoder split, redesign, probe fitting or target access is authorized by this operational change.
- Latest owner authorization, 2026-10-18: Step 2 completion was shared; start Step 3 source-only probes for F1–F3. Four inputs (raw last-layer WavLM, learned uniform-initialized softmax mixture of 13 layers, normalized CLAP, weighted-WavLM/CLAP concatenation) each get a logistic head and the approved 64→16 MLP. Fresh heads/layer weights per inner split and seed 1234/1235/1236, two source train domains fit and third source dev selects by strict EER; preserve approved batch132/AdamW3e-4/wd1e-4/constantLR/30max/patience5/BF16-FP32. Pure detection BCE for these embedding probes; do not apply anchor-specific losses without anchors. Validate cache payloads/canonical identities once, run bounded timing/reload checks, then sequential fits with no resume/overwrite and stop on failure. Report inner-domain macro EER/AUROC then mean/sample SD across seeds. Stop after Step 3; Step 4, F4 and new target/external evaluation remain held.
- Later owner authorization, 2026-10-18: start fresh AASIST F4 only on SSH2 after the passed source-only execution checks. Preserve seed1234, batch24/accum1, FP32, Adam1e-4/wd1e-4, cosine floor5e-6, maximum100epochs/patience5 and strict macro source-dev EER selection. Use validated cached sinc filters, device-local class reversal, buffered batch-one source-dev score transfer, 16 train/8 dev workers, prefetch2 and CPUthreads2. Preserve the active LACF probes; refuse existing run directories and never initialize from timing or another fold's weights. This authorizes AASIST F4 training only; F4 targets, external evaluation, other F4 models and further diagnostic stages remain held.
- Later owner authorization, 2026-10-19: Step3 results have been shared; start Step4 source-only A4–A9 and auxiliary-loss controls while preparing a detailed Steps1–3 report. Preserve active AASIST F4. Fit fresh adapter/relation heads on the same cached last-layer WavLM/CLAP views, inner source rotations, seeds1234/1235/1236 and approved batch132/AdamW/BF16-FP32 recipe. Reuse A8 as the explicitly labelled A9_no_bf result because both remove the same BF loss; seven unique configurations mean189fits. Fixed foundation text prototypes may be read from identical preserved buffers; no old learned adapter/head/layer weights initialize fits. Verify cache identity, bounded source GPU/reload behavior and completion before advancing; stop on failure, with no resume or overwrite. No encoder reruns, layer-mixture redesign, FT4, target/external evaluation or automatic later stages.
- Later owner request, 2026-10-22: implement the frozen C0–C4 layer-fusion comparison in separate `layer_fusion/` source/script folders, retaining CLAP/anchors/full35-D backend with detection BCE only. Use the approved F1–F3 inner source rotations, seeds1234/1235/1236 and fixed diagnostic recipe; preserve primary LACF/SOTA. This supersedes the earlier redesign restriction for implementation only. Validate with CUDA-hidden synthetic CPU tests. No GPU work, experiment launch, fine-tuning, target access or automatic publication is authorized; preserve active AASIST F4. See `docs/LAYER_FUSION_EXPERIMENTS.md`.
- Later owner authorization, 2026-10-22: apply necessary small review documentation/test fixes and start the frozen C0–C4 source-only experiments after cache validation and GPU preflight pass. Sharing the small cached-feature queue with preserved AASIST F4 is authorized. Keep the agreed architecture/recipe/seeds and target holds; do not select a global winner, fine-tune or publish automatically. SSL gating remains held on SSH2 while required pinned weights/parity evidence are missing and AASIST occupies the GPU.
- Latest owner authorization, 2026-10-22: start native P1 evaluation of completed AASIST F4 seed1234 only. This explicitly supersedes the F4 target hold for this fixed baseline checkpoint; verify completion/checkpoint/source identity and reproduce full source-dev calibration before opening ASV2019 LA eval. Preserve FP32, batch-one first64600/repeat-short inputs, original labels/score direction and shared raw/transductive/bootstrap metrics. Preserve active C0–C4; no target-informed tuning, other F4 model evaluation, external evaluation, retraining or automatic publication.
- Preserve shared label/score conventions, canonical split membership and the common evaluator contract across models. Record model-specific adaptations and source-only reasons.
- Latest owner authorization, 2026-10-24: implement and start only D1/D2 multi-layer readout controls in the existing `layer_fusion/` folders after CUDA-hidden CPU checks and source-only GPU preflight. D1 uses the C2 learned13-layer mixture, original normalized adapter and original eight anchors, with only p_W feeding an8-D head; D2 uses the same normalized adapter embedding directly with a512-D head. Both use detection BCE only, no CLAP audio or auxiliary losses, fresh initialization, and the existing F1–F3 inner rotations/seeds1234/1235/1236/recipe:54fits. Preserve common C2 adapter/head-tail initialization and dropout RNG; disclose differing first classifier blocks/capacity and separately learned mixtures. Keep completed C0–C4 intact and active SSLF2 untouched. No automatic final architecture selection, fine-tuning, later variants, F4/external/target evaluation, resume, overwrite, retry, commit or push.
- Latest SSL-gating owner authorization, 2026-10-22: adopt version2 training-only random crops up to20seconds after16kHz resampling, batch5/FP32 and full-recording batch-one dev/test. After passed source-only maximum-input/reload checks, start fresh F1 seed1234 on SSH1 as `ssl_gating_20s`; preserve the failed uncapped run. Prepare SSH2 code/weights/data readiness for later folds only; do not automatically launch them or new target evaluation. Keep16train/4dev workers, prefetch2 and the existing optimizer/loss/sampler. The19GiB allocator guard leaves CUDA overhead within an approximately20GiB process budget. No smaller batch, precision change, training resume or pilot-weight initialization. Other SSL-gating seeds remain unscheduled and its documented family seed plan is unchanged.
- Later SSL-gating owner authorization, 2026-10-22: start fresh SSH2 F2 seed1234 only as `ssl_gating_20s`, reusing the prepared/passed source-readiness evidence and pinned original XLS-R. Preserve version2 training-only20s crops, batch5/FP32, full-recording batch-one dev/test, Adam3e-6/wd1e-4/constantLR/CE+CKA/max10/patience3, 16train/4dev workers/prefetch2 and19GiB allocator guard. Preserve active C0–C4/AASIST evaluation. No resume, automatic retry, pilot/fold-weight initialization, later folds/seeds or new target evaluation; stop and preserve evidence on failure.
- Latest SSL-gating owner authorization, 2026-10-29: start fresh F3 seed1234 on SSH1 as `ssl_gating_20s`, after source/cache identity and unchanged validated runtime checks. Retain version2 batch5/FP32, training-only random up-to20s crops, full-recording batch-one dev/test, 16train/4dev workers, prefetch2, CPUthreads2, 19GiB allocator guard and existing optimizer/loss/sampler/strict source-macro-EER selection. Initialize from the pinned original XLS-R plus a fresh backend, never another fold or pilot checkpoint. This authorizes F3 training only; no target evaluation, automatic other folds/seeds, resume or retry. Preserve other-host work and the existing family seed plan.
- Latest SSL-gating evaluation authorization, 2026-10-29: evaluate completed F1 seed1234 natively on SSH1 alongside preserved F3 training. Verify selected epoch3/checkpoint identity and reproduce full source-dev calibration before opening SpeechFake test. Retain full recordings, batch1/FP32, original score direction, raw and separately labelled transductive transfer and shared conditional bootstrap intervals. Use the local audio cache; 8workers/prefetch2/CPUthreads2 and a separate12GiB evaluation allocator guard protect concurrent training. Stop on OOM/failure without cropping, skipping, automatic retry or changes to F3. This authorizes F1 native P1 only, not F3 targets, other protocols, retraining or target-guided changes.
- Owner compute-budget decisions: WavLM family `wavlm_bs96` (2026-10-05), AASIST family `aasist` (2026-10-06), and the initial LACF families `lacf` (10 s) and `lacf4s` (2026-10-08/09) use seed 1234 only across F1–F4. Do not schedule seeds 2345/3456 for these families. Other models retain their existing seed plan unless explicitly changed. Single-seed results have no across-seed standard deviation; target-bootstrap intervals do not measure seed variability.

## TF2 adaptation owner authorization (2026-10-31)

Implement the separate TF2 all12-block Q/V LoRA comparison: rank8/alpha16/no
LoRA dropout, LR1e-4; optional CLAP final audio stage plus projection at1e-5;
TF2 LR3e-4, common AdamW wd1e-4/constantLR/effectivebatch132/BF16-FP32/max30/
patience5. Preserve frozen base weights/encoder eval execution and original TF2
BCE head. New adaptation selection uses F1's three inner source rotations and
seeds1234/1235/1236, with fresh official bases and heads. This is not originalA10.
Latest owner choice: finish implementation and **bounded source-only measured
GPU timing first; report runtime before launching the long queue**. Do not
start final fits automatically. Preserve active SSLF2; no new target/F4/external
access, resume, overwrite, retry, commit or push. See the dated protocol.

Later owner request, 31 October: publish the pending isolated token-fusion
implementation, evaluator, adaptation code, tests and public documentation to
GitHub main, then provide safe SSH1 synchronization instructions. This overrides
the publication hold only; long adaptation training and new evaluation remain held.
Keep outputs, local reports/handoffs and reference originals excluded.


Later owner launch decision, 31 October: start exactly one fresh TF2 F1 inner-source
pilot, seed1234, with all12-block WavLM Q/V LoRA plus CLAP final audio stage and
projection adaptation. Fit ASV5+CFAD train; ASV2019 dev alone selects by strict
EER. Use validated physical22/accum6/effective132, 8train/4dev workers/prefetch2,
10GiB allocator cap and the approved optimizer/precision/max30/patience5.
This supersedes the launch hold for this one fit only. Preserve SSLF2, source
runtime guards and prior outputs; no target evaluation, automatic later runs,
resume, retry, overwrite, commit or push. Stop on failure. See local handoff
for launch/status; a pilot launch does not establish completed validation.

## Full F1 adaptation supersedes the pilot (2026-11-01)

Owner explicitly stopped the two-source TF2 pilot and requested a fresh full F1
run, then reconfirmed launch. Train on ASV2019, ASV5 and CFAD train; select by
strict equal-weight macro EER over their three development sets, ties earlier.
Use seed1234, all12-block rank8 Q/V LoRA plus CLAP final audio stage/projection,
and the approved TF2/LR/BF16-FP32/max30/patience5 recipe. Retain validated
physical22/accum6/effective132, 8train/4dev workers/prefetch2 and10GiB cap.
Start from pinned official base weights and fresh TF2; never resume or initialize
from the stopped pilot. Preserve its artifacts with owner-stop provenance and
preserve SSLF2. This one full-fold run overrides the previous inner-pilot scope;
new target evaluation, other folds/seeds, retries and automatic publication are
not authorized. Score/reload all three source dev sets and save source calibration
before completion. Record live status in the local handoff.

## SSL F2 evaluation and F4 training authorized (2026-11-03)

The owner authorizes native P1 evaluation of completed SSL-gating20s F2
seed1234 and fresh SSL-gating20s F4 seed1234 training on SSH2. For F2,
verify the selected checkpoint and reproduce all three source-dev results and
calibration before opening CFAD unseen test. Retain full-recording batch1/FP32,
raw and separately labelled transductive transfer and the shared bootstrap.
For F4, use ASV5, CFAD and SpeechFake source train/dev only; retain the approved
version2 batch5/FP32, training-only20s crops, full-recording development,
Adam3e-6/wd1e-4/constantLR/CE+CKA/max10/patience3 recipe, workers16/4,
prefetch2, CPUthreads2 and19GiB allocator guard. Start from pinned official
XLS-R and a fresh backend. Preserve active full-F1 TF2 and its source guards;
use bounded source-only resource checks and sequence the SSL jobs if they
cannot safely share the GPU. Stop on failure without resume, retry, overwrite
or scientific recipe changes. F4 target evaluation, other folds/seeds and
external targets remain unauthorized. No automatic commit or push.
Later same-day owner clarification: start F4 training immediately and defer F2
evaluation. Do not queue F2 evaluation automatically after F4.

## Implementation style

- Owner recovery decision, 2026-11-05: after the confirmed server shutdown,
  restart only TF2 full F1 from its source-selected epoch4 adaptation weights
  in a new output directory. Use `--recover-from` with the preserved best
  checkpoint and identical source/base/model/input/optimizer settings. This
  narrowly overrides fresh-initialization/no-retry restrictions for this
  recovery run; optimizer/RNG/sampler restart fresh, patience restarts at0,
  logical epochs continue5–30 and the inherited epoch4 best remains eligible.
  Disclose weights-only recovery, never exact resume. Save latest completed
  epoch weights/optimizer as well as the selected best. Preserve all interrupted
  artifacts and source-only reload/calibration gates. No target evaluation,
  SSL restart/recipe changes, other fits or publication are authorized; SSL
  speed options were requested as suggestions only.

- Keep the layout simple: reusable code under `src/audio_deepfake_detection/`, model entry points under `scripts/<model>/`, synthetic tests under `tests/`, research notes under `docs/`, references under `references/`. Keep `README.md` and `AGENTS.md` at the project root for onboarding and instruction discovery.
- Preserve unrelated user changes. Use conservative cleanup; do not delete research notes, run records or data as incidental tidying.
- Update README, paper notes and protocol when a requested change materially affects instructions or the study recipe. Do not represent CPU synthetic feasibility as completed GPU/real-data validation.

## Later SSL partial-fine-tuning authorization (2026-11-05)

The owner approves the suggested faster SSL adaptation and requests F4 launch;
other folds may use this variant later but are not automatically queued. This
supersedes the earlier same-day SSL-suggestions-only hold. Use a separate family
`ssl_gating_20s_top4_bf16`: freeze the frontend and first20 XLS-R blocks, train
last4 and the unchanged complete gating backend. Retain original encoder
SpecAugment/dropout/layerdrop in train mode; disable HF's forced input gradients
for the frozen convolutional frontend. Keep all25 layer features, training20s,
full batch-one dev, domain-balanced natural classes, CE+CKA, Adam3e-6/wd1e-4,
constantLR, max10/patience3 and strict source macro EER/ties earlier. Use BF16
with FP32 pooling/CE/CKA/scores. Source-only timing selects batch24/accum1,
8train/4dev workers/prefetch2/CPUthreads2/19GiB cap, without LR scaling.
Start F4 seed1234 fresh from pinned official XLS-R and a fresh backend; do not
use interrupted F4 or timing weights. Preserve TF2 recovery and its guarded
original SSL files: implement only new variant files. Keep original full-FT
results separate; larger batches also change the batch-based CKA objective.
Save atomic15minute/end-of-training-phase snapshots plus epoch best/last;
snapshots are not completion markers or exact mid-epoch resume. No target
evaluation, other launch, automatic retry/overwrite, commit or push.

Later owner publication request, 5November: publish the reviewed SSL partial
variant, TF2 weights-recovery support, focused tests and public documentation,
then provide SSH1 synchronization instructions. This overrides publication
holds only; preserve active source bytes/jobs and exclude local records,
outputs, datasets, pretrained weights and reference originals. Do not infer
additional fold launches or evaluation permission from synchronization.

Later owner launch request, 5November: start fresh F3 seed1234 on SSH1 in
`ssl_gating_20s_top4_bf16`, using the same approved batch24/BF16/last-four-block
recipe and source-only gates as F4. Preserve interrupted full-FT F3 and active
SSH2 work. This authorizes this one fresh F3 run only; no training resume,
automatic retries, other folds/seeds, target evaluation or publication.


## Experiment organization (owner decision, 6 November 2026)

Inactive diagnostics, layer-fusion and token-fusion work is organized under
`experiments/` in source/scripts/tests/outputs. See docs/EXPERIMENTS_LAYOUT.md.
Preserve relative compatibility links: they retain historical imports, hashes
and output references without duplicate data. Active TF2 runtime files/shared
diagnostic dependencies and active run/launch directories remain physically at
their original paths; organized entries link there. Do not move or edit these
until jobs finish and guards/completion have been checked. The owner requests
a reminder to finish deferred relocation at the next post-training cleanup.
This organization does not authorize a stop, restart, new fit or evaluation.

## Partial SSL F3 evaluation authorized (2026-11-09)

The owner requests native P1 evaluation of completed `ssl_gating_20s_top4_bf16`
F3 seed1234 on SSH1. Verify selected epoch9/checkpoint identity, all completed
source exports/calibration and fresh bounded BF16 source parity before ASV5
eval access. Use the separate partial evaluator, full unpadded batch1, trained
BF16/FP32 pooling-score policy and shared raw/transductive/bootstrap metrics.
Preserve original full-FT runs, active other-host work and all source bytes.
No crop/skip, fallback precision, automatic retry/resume, other fold/seed,
training, target-guided changes or publication is authorized.

Later same-day owner reply explicitly authorizes one corrected F3 evaluation
restart after the configuration-JSON check failed before target access. Preserve
the first attempt and use the tested exporter with identical checkpoint/recipe.
This overrides the retry hold for this one restart only; stop on a new failure.

After the9November reboot, the owner requests restarting the work on SSH1.
This authorizes only a fresh partial SSL F3 evaluation with the same validated
exporter/checkpoint/recipe. Preserve the interrupted partial CSV and provenance
in the previous attempt's archive; partial score recovery is not implemented.
Restart full target scoring, repeat source gates and stop on a new failure.
Do not restart SSH2 jobs, training or another user's processes.


## SAM-AASIST F1 authorization (9 November 2026)

Owner requests implementing and starting fresh SAM-AASIST F1 seed1234 after
source-only feasibility/VRAM checks, and explicitly selects released RawBoost
algorithm0 (off). Use the pinned authors' pooling0.5 architecture,297866 random
initialized trainable parameters, batch24/FP32, two-pass nonadaptive SAM rho0.05
around Adam1e-4/wd1e-4, per-update cosine floor5e-6, max100/patience5 and the
shared source-only sampler/strict macro EER selection. See the dated protocol
for input, BatchNorm and checkpoint contracts. No pretrained/pilot/fold weights
initialize the final run. Other SAM seeds/folds, target evaluation, automatic
resume/retry, scientific fallback and publication remain unauthorized. Preserve
all existing outputs and other-host work; the family seed plan is unchanged.

## Cached frozen-CLAP preparation and naming (7 November 2026)

Owner requests preparation only of the TF2 comparison with WavLM Q/V LoRA
and completely frozen cached CLAP,5,202,657 trainables. New code/scripts/tests
must be under real experiments/tf2_frozen_clap/ folders, never edit active
TF2/SSL runtime files. Reuse the existing matched source-crop CLAP cache;
WavLM stays live. No launch, GPU probe, download, target evaluation or SSL
resume implementation is authorized by preparation. Source-only GPU memory/
throughput/cache-parity/reload checks and matching evidence remain pending.
Use concise descriptive output families from now on; fold/seed identifiers
remain explicit. Planned outputs/experiments/tf2_frozen_clap/f1/seed_1234/.
Do not rename existing historical or active output folders. See the experiment
README under scripts/experiments/tf2_frozen_clap/.

## Reboot recovery authorized (9 November 2026)

Owner requests restarting interrupted SSH2 jobs after the 14:52 shutdown and
15:34 boot. SSL partial F4 already completed with source verification: do not
restart its training. Restart only the interrupted original SSL F2 evaluation,
reusing hash-verified complete source CSVs and preserving incomplete files;
retain original numerical defaults and the unchanged source gate before target
access. Restart TF2 full F1 into a new descriptive output folder from the latest
completed epoch10 weights and optimizer, retaining selected epoch6 and patience4.
Repeat unsaved epoch11, retain maximum30/patience5 and all scientific settings.
RNG/sampler/worker streams restart; this is not exact uninterrupted resume.
Minimal tested --continue-from support is authorized for this epoch restart;
preserve old runs, stop on failure and do not retry automatically. Frozen-CLAP
preparation stays unlaunched. No other training/evaluation, downloads or automatic
commit/push. Check live evidence/local handoff rather than historical PIDs.

## Partial SSL F4 native evaluation authorized (9 November 2026)

Owner requests evaluation of completed `ssl_gating_20s_top4_bf16` F4 seed1234
only. This overrides the F4 target hold for this fixed baseline checkpoint.
Use the separate BF16 evaluator, strict checkpoint/config/source-export hashes,
canonical source membership, reproduced source macro EER/calibration and bounded
fresh-source bit-exact parity before opening ASV2019 LA eval. Preserve full
unpadded batch-one recordings, BF16/FP32 pooling and scores, original numerical
defaults, source thresholds, raw/separate transductive transfer and conditional
bootstrap intervals. Reuse the already full-reload-verified source CSVs; do not
rerun encoders for every source unnecessarily. Preserve active TF2/F2 evaluation
with a separate6GiB allocator guard,4workers/prefetch2/CPUthreads2. Stop on
failure without crop/skip/precision fallback/retry/overwrite. No other targets,
training, tuning on targets, downloads or automatic publication.


## SSL partial F2 training authorized (9 November 2026)

Owner requests fresh F2 seed1234 training in `ssl_gating_20s_top4_bf16`,
using the unchanged approved F3/F4 partial recipe: frontend/blocks0–19 frozen,
last4 XLS-R blocks and full gating backend train; BF16 with sensitive FP32;
batch24/accum1, Adam3e-6/wd1e-4/constantLR, CE+CKA, training-only20s, full
batch-one source dev, max10/patience3, strict equal-domain macro source EER,
ties earlier. Use8train/4dev workers/prefetch2/CPUthreads2/19GiB cap after
source readiness and bounded GPU/reload checks. Initialize from pinned official
XLS-R and a fresh backend, never full-FT F2 or another fold. F2 sources are
ASV2019,ASV5,SpeechFake; no CFAD target access/evaluation is authorized.
Preserve active TF2 and its guarded source bytes. Keep C4 prepared but held
at the owner's latest instruction. No automatic retry, resume, overwrite,
other folds/seeds, downloads, commit or push. Record measured launch status
in the ignored local handoff; CPU checks alone do not establish GPU readiness.


## Frozen cached-CLAP TF2 F1 launch authorized (9 November 2026)

Owner stops the newly started partial SSL F2 to prioritize the prepared full-F1
TF2 frozen-CLAP comparison. Preserve the SSL snapshot/logs with an owner-stop
record; do not resume or retry it automatically. Preserve active adapted TF2
and its source guards. After focused CPU and actual source-only GPU/cache-parity/
reload/headroom checks, start fresh seed1234 in
`outputs/experiments/tf2_frozen_clap/f1/seed_1234/`. Train on ASV2019,ASV5,CFAD;
their three dev sets select/calibrate source-only. Train all12-block Q/V rank8
LoRA alpha16/no dropout plus unchanged TF2,5202657 parameters. CLAP audio stays
an input through exact-view verified cached raw projections; its encoder is
not loaded/executed for final training. WavLM remains live because LoRA changes
it. Retain existing fixed10s views,domain/class sampling,detectionBCE only,
AdamW LoRALR1e-4/TF2LR3e-4/wd1e-4/constant,BF16-FP32,effective132,max30/
patience5,strict equal-domain macro source EER/ties earlier. All new experiment
implementation/tests/outputs stay under experiments. No target evaluation,
other folds/seeds,pilot initialization,resume,retry,overwrite or publication.
The prepared C4 run stays held. Discard every validation update; final source
full reload/export/calibration remains mandatory before completion.


## Both-host publication and synchronization (11 November 2026)

The owner requests publishing the reviewed uncommitted code/documentation from
both SSH1 and SSH2 and synchronizing their main checkouts. This overrides earlier
publication holds only. Preserve active-job source bytes, local records, outputs,
reference originals, datasets and caches. Reconcile overlapping changes before
publication, run CUDA-hidden offline CPU checks, then verify the shared commit,
clean Git status and all active guards on both hosts. No experiment launch,
training/evaluation restart, recipe change or output transfer follows from sync.

## SAM F2 training and F1 evaluation authorized (14 November 2026)

Owner requests fresh SAM-AASIST F2 seed1234 on SSH1 with source-only measured
loader tuning, and native P1 evaluation of completed F1 seed1234. Preserve all
scientific settings including batch24/FP32, pooling0.5,297866 fresh trainables,
RawBoost off and two-pass SAM. Use24train/8dev workers/prefetch4, pinned and
persistent memory loading/CPUthreads2. Owner's final clarification preserves
batch size and requests no GPU memory cap; it supersedes the intermediate25GiB
limit. Measure concurrent headroom before launch. F2 reads only ASV2019/ASV5/
SpeechFake train/dev. F1 SpeechFake test opens only after selected-epoch14 and
completion/config/runtime/source-export identities, exact frozen calibration
and bounded fresh-source parity pass. Preserve fixed first64600/repeat-short
batch-one FP32 scoring and shared P1 raw/transductive/bootstrap contracts.
Reuse already full-reload-verified source exports without unnecessary complete
source encoder reruns. Stop on failure; no automatic retry/resume, overwrite,
other folds/seeds, F2 targets, target-guided tuning or publication is authorized.
