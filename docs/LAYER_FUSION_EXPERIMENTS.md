# Frozen layer-fusion and readout comparisons

## Four-expert full-fold comparison (owner authorization, 11 November 2026)

Later same-day owner direction extends this study to allF1–F4, seed1234 only.
Retain completed F1 and its exact archived trainer source, then start fresh
F2→F3→F4 with `python -m scripts.experiments.layer_fusion.train_four_experts --folds f2 f3 f4`
through the recorded local launch gates. Cache required source train/dev arrays
once; each fit receives only its own six canonical source partitions. Per-fold
preflight and selected-checkpoint/source completion must pass before advancing.
No target evaluation follows. Results remain in the same family's fold/1234
directories; `source_summary.json` appears only after allfour completions verify.

The isolated `four_experts.py` model and `train_four_experts.py` entry point compare
four residual experts over R0–R3/R4–R6/R7–R9/R10–R12 with the completed original-bank
three-expert fullF1 seed1234 model. The router reads the mean of all13 raw layers,
uses LN→Linear768→4→softmax, starts uniformly, and returns a768-D weighted sum.
Original3+5 anchors, CLAP audio, adapter, full35-D relations and detectionBCE-only
objective remain unchanged.1068779trainables versus967338; changed layer boundaries
and capacity prevent attributing a difference to expert count alone. The common
backend initialization/dropout RNG matches the reference; all final weights are fresh.

Use the same validated frozen cache, domain/class sampling, batch132, AdamW3e-4,
wd1e-4, constantLR, BF16/FP32, max30/patience5 and strict source macro EER/ties earlier.
Results: `outputs/experiments/layer_fusion/lacf_layer_moe4/f1/1234/`; launch evidence
under its family's `launch/`. CPU and finite-source/full-source-reload GPU gates
precede fresh training. Record four-column routing summaries by source/class.
One fullF1 source fit is exploratory, not inner-held-domain validation or a new
generalization claim. No target evaluation, automatic later experiment or publication.

## D1/D2 authorization and design (24 October 2026)

The owner authorizes the two missing readout controls in the same source/script
folders, after CPU checks and source-only GPU preflight. The original C0–C4
comparison below remains intact; its completed output directory is never reused.
Keep active SSL F2 running. These diagnostics do not remove CLAP from primary
LACF, select a final method, change the F4 policy or authorize target evaluation.

Both controls use C2's uniform-initialized learned softmax mixture of all13 raw
WavLM representations, the original LN→768→512→GELU→Dropout(.2)→512 adapter,
and FP32 L2 normalization of the adapter output. Frozen encoders are not loaded.

| Condition | Readout after the normalized adapter | Trainable parameters |
|---|---|---:|
| D1 | Original8-prototype softmax at0.07 → LN(8)→64→GELU/dropout→LN(64)→16→GELU/dropout→1 | 659,710 |
| D2 | Normalized512-D embedding → same head ordering with512 input dimensions | 692,974 |

D1 retains language-defined text anchors but no CLAP audio or cross-view
relations. D2 has no anchor buffer or CLAP audio path. Detection BCE is the only
loss; both auxiliary flags and weights are false/zero. No prompts are changed.
All mixture logits, adapters and classifiers train freshly in each fit.

Within each seed, initialize the adapter and classifier64→16→1 tail identically
to a fresh C2, with equal zero layer logits and the same subsequent dropout RNG.
Construction uses a temporary randomly initialized C2 backend to match its RNG
consumption, discards the unused35-D head, and constructs the new first head
block inside an isolated RNG context. No old fitted model initializes these
controls. CPU tests and a source GPU check compare shared weights, normalized
adapter outputs and dropout RNG against a fresh C2.

These controls are not perfectly capacity-matched: the first classifier blocks
take35(C2),8(D1), or512(D2) inputs. D2 versus a direct Step3 probe also changes
depth, normalization and capacity. All arms learn their mixture coefficients
separately; they do not have identical fitted features. C2 versus D1 studies
the complete CLAP-relation pathway, not CLAP alone with an identical classifier.
D1 versus D2 studies the anchor readout with its necessary dimension change.
Report these limits rather than claiming either comparison isolates every cause.

Reuse the same cache, logical training views, balanced utterance draws, F1–F3
inner-source rotations and seeds1234/1235/1236. Batch132/accum1, AdamW3e-4,
weight decay1e-4, constant LR, BF16/sensitiveFP32, maximum30/patience5 and strict
EER selection are unchanged. Two conditions ×3source sets ×3rotations ×3seeds
give54fits. Each selected checkpoint must reload and reproduce all held-source
scores exactly before advancement. Summary means give equal weight to inner
domains, then report three-seed mean/sample SD. No target calibration or scores.

Only `model.py`, the existing runner and focused synthetic tests are extended.
The runner's default remains C0–C4; `--comparison readouts` selects D1/D2 only.
It validates canonical source identities and all feature payload hashes once,
keeps the existing feature bank in CPU RAM, and transfers only WavLM features
and labels for D1/D2 training. No waveform workers or new encoder passes.
Source GPU preflight performs8warmup+64measured updates per control, complete
ASV2019 source-dev scoring and exact fresh reload. Timing weights are discarded.
Log, configuration, provenance and failed checks are preserved; no resume,
overwrite or automatic retry. Stop after54fits; later experiments need a decision.

```bash
python -u -m scripts.layer_fusion.train --comparison readouts \
  --output outputs/layer_fusion/frozen_d1_d2_YYYYMMDD
```

The earlier7–8hour suggestion is an extrapolation from other heads, not measured
D1/D2 runtime. Use this run's source-only preflight and actual progress instead.

## Original C0–C4 comparison (22 October 2026)

Owner request, 22 October 2026: implement this comparison in separate folders.
Implementation and synthetic CPU validation alone do not authorize launching it.
Later owner instruction on 22 October authorizes the small review corrections,
source-only GPU preflight and fresh C0–C4 queue after it passes, sharing SSH2
with preserved AASIST F4. No SSL-gating launch is scheduled while its weights/
parity prerequisites are missing; no fine-tuning or target evaluation is added.
Primary LACF, completed diagnostics and SOTA recipes/results remain intact.
Partial fine-tuning is a later experiment and is not implemented here.

## Files and reuse

- `src/audio_deepfake_detection/experiments/layer_fusion/model.py`: fusion candidates wrapping
  the existing full LACF backend; no pretrained encoders are constructed.
- `src/audio_deepfake_detection/experiments/layer_fusion/diagnostics.py`: descriptive routing
  summaries on source development only.
- `scripts/experiments/layer_fusion/train.py`: bounded source GPU preflight followed by the
  fresh sequential comparison, if separately authorized to run.
- `tests/experiments/layer_fusion/test_layer_fusion.py`: synthetic CPU architecture, precision, initialization,
  sampling, gradients, reload, provenance and target-isolation checks.

The runner reuses canonical feature-cache validation and vectorized sampling
from `experiments/lacf_diagnostics/probes.py`, the existing optimizer/epoch loop and shared metrics.
The only existing training-helper extension is the explicit `layer_fusion`
experiment identity in `scripts/experiments/lacf_diagnostics/train_probes.py`; numeric Step3/Step4
identities and behavior remain unchanged. No SOTA or primary LACF model code
is modified. No extra dependencies or environment manifest are needed.

## Fixed comparison

Every row retains frozen WavLM Base+ and CLAP features, the original eight text
prototypes/three templates, temperature 0.07, 768→512→512 adapter, full 35-D
relation vector and original pre-LayerNorm 35→64→16→1 classifier. CLAP receives
its two original FP32 normalizations. JS and entropy remain input features.
The objective is **detection BCE only**, with semantic/BF losses both disabled
and both weights recorded as zero. This is an explicitly new comparison, not
an alteration or exact reproduction of the original primary three-loss model.

| Condition | WavLM fusion before the common adapter | Trainable parameters |
|---|---|---:|
| C0 | Last representation | 661,479 |
| C1 | Uniform mean of all 13 representations | 661,479 |
| C2 | Learned global softmax mixture of 13 representations | 661,492 |
| C3 | Three residual experts, learned global softmax mixture | 963,498 |
| C4 | Same experts, recording-dependent softmax router | 967,338 |

R0 is the projected convolutional frontend; R1–R12 are Transformer outputs.
Expert groups are R0–R3, R4–R7 and R8–R12. Each group is averaged, then processed
by `x + Linear(64,768)(GELU(Linear(768,64)(LN(x))))`. There is no additional
expert dropout. Each expert has 100,672 parameters. C3 is the capacity control
for C4: it has the same experts but a global three-logit mixture. Counts are
near-matched, not identical: C4 has 3,840 additional parameters.

C1/C2 average raw-scale layers without per-layer normalization. Uniform weights
do not imply equal contribution when layer norms differ; C1 versus C2 compares
fixed versus learned coefficients on those raw vectors. Three uniformly mixed
group means give residual input coefficients 1/12 for R0–R7 and 1/15 for R8–R12,
before adding the nonlinear expert outputs. C2 versus C3 changes grouping and
capacity together; C3 versus C4 isolates global versus input-dependent routing
with the same experts. No new normalization or regrouping is introduced.

C4 routes from the mean of all 13 raw layer vectors through LN(768), Linear(768,3)
and softmax. Its final linear starts at zero; C3 logits also start at zero.
Both therefore start with uniform expert weights and identical predictions.
C2 also starts uniformly. Common adapter/head initialization and subsequent
PyTorch dropout RNG state match across conditions within each seed; expert
initialization matches between C3/C4. Fusion initialization is isolated from
the common RNG stream. Each final fit starts fresh, never from an old trained
adapter/head, a timing checkpoint or an earlier fold.

Pooling, softmax mixtures, normalization, probabilities, logarithms, JS, entropy
and losses use FP32. Projection/expert/router linear work uses BF16 autocast.
Routing report moments accumulate in FP64 to avoid spurious standard deviations
from rounded squared probabilities; these descriptive moments do not train.

## Data and selection

Use the completed Step2 source cache: centered up-to-10 s development views,
center plus two fixed random training views with identical short views aliased.
For each sampled training utterance, choose one logical view uniformly. Preserve
native masked-mean pooling and the cache's recorded numerical/execution history.
Do not claim this reproduces fresh random waveform crops every epoch.

F1–F3 each have three inner rotations: fit two source **train** domains; select
on the third source **dev** domain. The original outer folds do not change.
Use seeds 1234/1235/1236, giving 27 fresh fits per condition and **135 total**.
Each fit uses batch132/accumulation1, inverse domain/class-bucket replacement
sampling, AdamW LR0.0003/weight-decay0.0001, constant LR, BF16/FP32, maximum30
and patience5. Strictly lower inner-held dev EER selects; ties retain the earlier
checkpoint. AUROC is reported, not used for selection. Labels remain 0=bona fide,
1=spoof; a higher scalar logit means spoof.

Canonical IDs/labels and every feature payload hash are verified once, then the
~62.6 GiB full feature bank remains in CPU RAM with vectorized batch gathers.
Use zero audio-loader workers and CPUthreads2: this path reads features, not
waveforms. No repeated audio decoding or full cache hashing inside epochs.
F4/external targets and F1–F3 target scores are not loaded. ASV2019 train/dev
remain valid sources in F1–F3. This queue never opens F4 eval recordings; this
does not mean ASV2019 was excluded from method development. The separate
22 October AASIST F4 evaluation authorization covers only that fixed baseline.
This queue reports all
conditions and does not automatically select a global winner or release F4.
Before choosing a design for later F4 evaluation, record its selection rule and
this dataset-level development exposure; the existing outer folds stay fixed.

Before final fits, GPU initialization checks verify FP32 features/loss and
bit-exact initial C3/C4 predictions plus CUDA dropout RNG states on one source
batch. Then each condition gets eight warmup and 64 measured updates on
F2 fit sources, complete ASV2019 source-dev scoring and an exact fresh reload
check. Timing-trained weights are discarded. Failed checks stop the queue;
there is no automatic precision fallback, resume, overwrite or retry. Real GPU
memory/throughput and time estimates remain unmeasured until this preflight is
authorized and run. Do not promise a wall-clock duration from synthetic tests.

## Outputs and diagnostics

Use a fresh directory under `outputs/layer_fusion/`; existing output directories
are refused. Each fit stores configuration, log, history, selected checkpoint,
canonical held-source scores, verified completion/score manifests and metrics.
Fresh reload must reproduce the selected score vector and metrics exactly.
Hashes bind cache, prototype source buffers, implementation and output artifacts.
Old F1/F2 checkpoint buffers supply only identical frozen prototypes; none of
those checkpoints' learned model parameters initializes a fit.

C2 reports learned layer weights. C3/C4 additionally report soft routing means,
standard deviations, mean entropy and top-expert counts, by source dev domain
and bona-fide/spoof class, after selecting/reloading the checkpoint. These are
not selection criteria. Argmax ties pick expert0; hard counts alone cannot
establish collapse. Differences across source domains do not prove a shortcut.
Only source dev is used for these reports; the outer target is never scored.

The queue advances only after verified completion and saved source diagnostics.
`summary.json` includes individual fit records, equal-three-inner-domain EER/AUROC
macros per seed, then three-seed means and sample SD (ddof=1). All 135 verified
fits and 15 condition/fold summary cells are required for the root completion
marker. No target bootstrap or target threshold calibration is performed.
Stop after this comparison and report results; there is no automatic fine-tuning,
model redesign, new loss variant or target evaluation.

## Commands for a later authorized launch

From the project root, with the existing environment activated:

```bash
python -u -m scripts.layer_fusion.train \
  --output outputs/layer_fusion/frozen_c0_c4_YYYYMMDD
```

Only cache/data/output paths are arguments; the agreed scientific recipe is
fixed and recorded. Default cache is `outputs/lacf_diagnostic_features_v1`,
default canonical audio metadata root is `/mnt/drive/audio-deepfake-cache`.
Preconditions: owner launch authorization (now given for C0–C4), available CPU RAM and CUDA memory,
unchanged completed source cache/prototype-source checkpoints, and preserved
other jobs. Launch outcome and measured status are recorded separately in the local handoff.

Synthetic verification (no GPU/weights/data required):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q tests/experiments/layer_fusion/test_layer_fusion.py
```

Implementation validation, 22 October 2026: 50 focused CPU tests and the
activated-environment canonical suite (406 passed, six existing warnings)
passed with CUDA hidden and offline flags. Launcher `--help` and
`git diff --check` passed. No real-source GPU preflight or fit was run.

Post-review validation, 22 October: 412 CUDA-hidden CPU tests passed (six
existing warnings), including the production queue on tiny synthetic source
features. Source GPU checks and experiment progress are recorded in the ignored
local handoff and run outputs; CPU tests do not establish GPU validation.


### Four-expert native evaluation (11 November 2026)

Owner-authorized evaluator: `python -m scripts.experiments.layer_fusion.evaluate_four_experts --folds f1 f2 f3 f4 --wait-for-training`.
Checks each source-complete run before target access; waits for the currently
running source-only queue without restarting it. Stops on failure and refuses
existing evaluation directories. New outputs:
`outputs/experiments/layer_fusion/evaluations/lacf_layer_moe4/<fold>/1234/`.
Includes copied selected checkpoint/source artifacts, fresh source-validation
record, target scores, metrics/conditional intervals and hashed completion.
F1 reuses the existing verified SpeechFake13-layer target cache; F2–F4 retain
new all-layer/CLAP target feature shards under their evaluation directories.
Fixed source thresholds and unchanged four-expert head; all raw/z-score
1%,5%,10% APCER/BPCER/ACER plus EER/AUROC are reported. Transductive target
normalization is separately labelled; these exploratory single-seed results
must not select further architecture/recipe changes.

Calibration verification rereads source scores in the same sorted recording-ID
order used when fitting/exporting the frozen calibration. File order can change
the last bits of floating-point standard deviations and normalized thresholds;
the verifier retains exact dictionary equality, with no tolerance or rounding.
Original source scores, calibration and checkpoints remain unchanged. After the
verifier-only correction, historical trainer hashes require an explicit archived
old/new hash record under the family's `launch/calibration_verification_fix/`.
F1 additionally retains its earlier reviewed fold-extension hash step. Model and
scorer guards remain strict; the correction does not restart a failed queue.


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
