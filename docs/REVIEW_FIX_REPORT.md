# Evaluation review and fixes — 2 October 2026

Historical audit: the findings below describe that review's state, not the
latest live run. Later batch/seed decisions, independent-fold authorization
and score recovery are recorded in `EXPERIMENT_PROTOCOL.md` and the local
`PROJECT_HANDOFF.md`. Preserve this audit; do not read its original three-seed
defaults or pre-evaluation status as overriding later owner decisions.

## 1. EXECUTIVE VERDICT

**CONSISTENT after fixes, within static and synthetic verification.** The metric
formulas and main normalization/bootstrap algorithms were already correct. The
changes fix gaps in score membership, provenance, seed enforcement, duplicate
consistency, and publication summary validation. No training or real-data
evaluation was launched for this review.

## 2. FILES CHANGED

- `src/audio_deepfake_detection/metrics.py`
- `src/audio_deepfake_detection/protocol.py`
- `src/audio_deepfake_detection/data.py`
- `scripts/wavlm/evaluate.py`
- `scripts/evaluate_scores.py`
- `scripts/summarize_results.py`
- `scripts/prepare_cache.py`
- `tests/test_metrics.py`
- `tests/test_evaluation_pipeline.py`
- `tests/test_data_cache.py` (new synthetic reader/cache tests)
- `README.md`
- `REVIEW_FIX_REPORT.md` (this report)

The two new files contain tests and the requested audit report. No application
module, framework, dependency, registry, or model abstraction was introduced.

## 3. CRITICAL FIXES

| File | Old behavior and risk | New behavior |
| --- | --- | --- |
| `metrics.py`, `protocol.py`, both evaluators | Existing scores could be accepted by dataset and count without proving canonical membership. Equal-sized wrong splits or incorrect labels could pass. | Exact canonical ID set and per-ID labels are checked through the official reader. Duplicate, missing, extra, and wrong IDs fail. Accepted rows are sorted by ID, making bootstrap results independent of CSV row order. |
| `protocol.py`, `wavlm/evaluate.py` | Manifest bound fold/seed/checkpoint, but not the declared split or CSV bytes. A replaced CSV with the same count could be reused. | Manifest binds model, fold, seed, checkpoint SHA-256, completion SHA-256, dataset/split per file, and CSV SHA-256. Old or unregistered score files fail instead of being adopted. Newly generated files are validated before digest registration. |
| Both evaluators | Completion/checkpoint guards were uneven; a NaN source macro EER could evade an absolute-difference comparison. | Require completion and best checkpoint, validate run identity and epochs, require finite source EERs in [0,1], verify source macro EER before target access. WavLM additionally checks loaded checkpoint identity, best epoch, source membership, target, and saved macro EER. |
| `evaluate_scores.py` | Generic metrics omitted seed and trusted score filenames. | Require seed, canonical split validation, completion/checkpoint manifest, source-selection consistency, and source calibration before target membership is opened. Record model/run identity, roots, checkpoint/score hashes, epoch, and evaluator hash. |
| `metrics.py` | CI metadata did not explicitly describe all conditional assumptions. | Record target-only resampling, class stratification/count preservation, fixed source thresholds/statistics, fixed target statistics, percentile endpoints, and exclusion of training/calibration uncertainty. Record metric implementation hash and schema version. |
| `summarize_results.py` | Missing seed provenance passed; arbitrary seed lists could look final; incomplete operating metrics or CI structures could be accepted. | Publication default requires 1234/2345/3456 for every fold. Missing/wrong seed/model/fold/target/split/source identity, absent or malformed CIs, mismatched metric structure, and non-finite metrics fail. Explicit `--non-final` writes an exploratory summary. |
| `data.py` | SpeechFake duplicate signatures included only five selected metadata fields; path aliases were not fully normalized. | Compare every supplied metadata column, trim whitespace, canonicalize label spelling, and normalize relative paths. Conflicting train/dev overlaps raise; matching overlaps are removed from train. No count requirement was weakened. |
| `prepare_cache.py` | Zero/negative workers failed late; an unset DATA_ROOT caused a KeyError even with CLI arguments. | Reject `--workers < 1` at parsing; use the existing canonical raw root as the default when DATA_ROOT is absent. Copy/cache architecture is unchanged. |
| Evaluators and summary command | An existing report could be silently replaced. | Refuse existing `metrics.json` and summary output. This review did not overwrite real-run outputs. |

## 4. METRICS VERDICT

- EER: correctly interpolates the bona-fide/spoof error crossing of the ROC.
  Perfect separation gives 0, fully inverted separation gives 1, and all ties
  give 0.5. No target EER crossing threshold is exported for deployment.
- AUROC: positive class is spoof (label 1); orientation is unchanged.
- APCER: `P(score < threshold | spoof)`.
- BPCER: `P(score >= threshold | bona fide)`.
- ACER: `(APCER + BPCER) / 2`.
- Threshold direction: equality predicts spoof.
- Calibration: sort pooled source-development spoof scores and select the
  zero-based element `floor(rate * n_spoof)`. With the equality rule, source APCER
  is no greater than the nominal rate, including ties. Rates remain 1%, 5%, 10%.
- Non-finite scores and invalid labels fail. EER/AUROC require both classes;
  each class-conditional metric requires its corresponding class.

These formulas and score direction were preserved.

## 5. NORMALIZATION VERDICT

Each source-development domain uses its own mean and population standard
deviation. The normalized source scores are pooled for source-only calibration.
The full target's raw-score distribution provides its own unlabeled mean/std;
no target labels enter that calculation. There is no per-class target
normalization. `ddof=0` and the minimum std of `1e-12` are unchanged; constant,
near-constant, and non-finite distributions fail clearly. Positive-scale
normalization preserves EER/AUROC ranking, so they are reported once from raw
scores. This branch remains explicitly transductive. Normalization is absent
from training and checkpoint selection.

## 6. BOOTSTRAP VERDICT

The CI is **TARGET-ONLY, CLASS-STRATIFIED, PERCENTILE BOOTSTRAP WITH FIXED SOURCE
THRESHOLDS AND FIXED TARGET Z-SCORE STATISTICS**. Source normalization statistics
also stay fixed. Within each target class, observations are sampled with
replacement while retaining the original class counts. Defaults remain 1,000
resamples, seed 2026, and 2.5th/97.5th percentiles for 95% intervals.

Intervals cover target EER/AUROC and both transfer branches' APCER/BPCER/ACER at
all three operating points. Full-pipeline bootstrap was not introduced. These
intervals exclude training-seed and calibration uncertainty; training-seed
variation is summarized separately.

## 7. SCORE VALIDATION VERDICT

Both evaluation paths now reject wrong split membership, wrong IDs, missing
IDs, extra IDs, wrong labels, duplicate IDs, wrong dataset columns, and changed
or unregistered checkpoint-associated score files. Correct IDs/labels in a
different order pass; deterministic canonical order is used for metrics.
Internal WavLM scoring still verifies loader labels/order against its records.

Legacy manifests without exact split and score-checksum provenance fail.
This review did not migrate or overwrite any existing real scores.

## 8. LODO / TARGET ISOLATION VERDICT

| Fold | Sources | Target split | Verdict |
| --- | --- | --- | --- |
| F1 | asv2019, asv5, cfad | speechfake/test | Preserved; target gate tested synthetically. |
| F2 | asv2019, asv5, speechfake | cfad/test_unseen | Exact definition tested; same evaluation guards. |
| F3 | asv2019, cfad, speechfake | asv5/eval | Explicit test confirms ASV5 is absent from sources, including source development. |
| F4 | asv5, cfad, speechfake | asv2019/eval | Exact definition tested; same evaluation guards. |

WavLM training statically loads only source train/dev. Its architecture,
dataset-balanced replacement sampler, natural within-domain class ratios,
runtime random crops, BF16, optimizer groups, scheduler timing, source macro
checkpoint criterion, early stopping, best/last saves, and existing-run refusal
match the current contract. Its completion marker is written after the training
loop finishes normally. Training and model files were not edited.

Both evaluators verify source scores and source macro EER and calibrate source
thresholds before reading target membership. Synthetic WavLM main-path tests
prove checkpoint/completion/provenance/source-EER failures block target scoring.

## 9. SEED / SUMMARY VERDICT

Final defaults are exactly **1234, 2345, 3456**, with all four folds required.
Every final report must contain its seed. Per-fold summaries calculate mean and
sample standard deviation (`ddof=1`) across seeds. The headline is an
equal-weight mean of the four fold means. `target_datasets_pooled=false` remains
explicit. Non-final exploratory seed lists require an explicit override and
produce a separately named output.

## 10. TESTS ADDED OR UPDATED

Verification: **90 synthetic CPU tests passed**. `git diff --check` passed.

Coverage includes exact F1–F4 definitions and F3 isolation; perfect/inverted/tied
EER and AUROC; non-finite and one-class inputs; threshold equality and tied
order statistics; per-domain normalization and target-label independence;
near-constant std rejection; deterministic bootstrap with fixed class sizes,
thresholds/statistics and percentile endpoints; exact score membership and
order independence; changed score/checkpoint/completion provenance; missing
completion/checkpoint/manifest; wrong checkpoint epoch/seed/membership; source
macro mismatch blocking target access; all twelve run reports and summary
validation; SpeechFake duplicate/overlap conflicts in formerly ignored fields;
and invalid cache worker arguments.

WavLM main-path tests replace model loading and score generation with CPU-only
fakes. No pretrained checkpoint, audio, real dataset reader invocation, model
forward pass, or CUDA kernel is used by those tests.

## 11. ANY UNRESOLVED QUESTIONS

**CANNOT VERIFY WITHOUT REAL DATA:** actual canonical counts after stricter
SpeechFake metadata validation, completeness of the real cache, and a completed
best checkpoint's real source-EER reload equivalence. Real-data access was
excluded by this task, so these were inspected statically and tested with
temporary fixtures only. No canonical count target has been relaxed.

The current WavLM baseline's previously documented reproduction omissions
(published augmentation/regularization details) remain outside this review.

## 12. DO NOT CHANGE THESE

No changes were made to the WavLM recipe or architecture, active checkpoints,
current F1 process/output files, raw datasets, target isolation rules, cache
architecture, score direction, or fold definitions. No training was launched,
no real target was scored or opened, and no long evaluation job was run.
