# Frozen layer-fusion comparison: C0–C4

Owner request, 22 October 2026: implement this comparison in separate folders.
Implementation and synthetic CPU validation alone do not authorize launching it.
Later owner instruction on 22 October authorizes the small review corrections,
source-only GPU preflight and fresh C0–C4 queue after it passes, sharing SSH2
with preserved AASIST F4. No SSL-gating launch is scheduled while its weights/
parity prerequisites are missing; no fine-tuning or target evaluation is added.
Primary LACF, completed diagnostics and SOTA recipes/results remain intact.
Partial fine-tuning is a later experiment and is not implemented here.

## Files and reuse

- `src/audio_deepfake_detection/layer_fusion/model.py`: fusion candidates wrapping
  the existing full LACF backend; no pretrained encoders are constructed.
- `src/audio_deepfake_detection/layer_fusion/diagnostics.py`: descriptive routing
  summaries on source development only.
- `scripts/layer_fusion/train.py`: bounded source GPU preflight followed by the
  fresh sequential comparison, if separately authorized to run.
- `tests/test_layer_fusion.py`: synthetic CPU architecture, precision, initialization,
  sampling, gradients, reload, provenance and target-isolation checks.

The runner reuses canonical feature-cache validation and vectorized sampling
from `lacf/probes.py`, the existing optimizer/epoch loop and shared metrics.
The only existing training-helper extension is the explicit `layer_fusion`
experiment identity in `scripts/lacf/train_probes.py`; numeric Step3/Step4
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
  python -m pytest -q tests/test_layer_fusion.py
```

Implementation validation, 22 October 2026: 50 focused CPU tests and the
activated-environment canonical suite (406 passed, six existing warnings)
passed with CUDA hidden and offline flags. Launcher `--help` and
`git diff --check` passed. No real-source GPU preflight or fit was run.

Post-review validation, 22 October: 412 CUDA-hidden CPU tests passed (six
existing warnings), including the production queue on tiny synthetic source
features. Source GPU checks and experiment progress are recorded in the ignored
local handoff and run outputs; CPU tests do not establish GPU validation.
