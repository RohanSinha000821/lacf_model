# Python code study guide

This guide describes the current implementation. WavLM-WA and AASIST have training/scoring paths; LACF now has its primary frozen model and source-only trainer, with synthetic CPU checks and recorded source-only GPU validation; the 2026-10-09 optimization requires a new duration-specific freeze. The 2026-10-07 AASIST pilot passed; full source-dev/reload validation remains a separate gate. Other SOTA models remain planned. Start with the WavLM runtime flow below, then follow the reading order. Sections 13 and 15 cover AASIST and LACF. Studying these files does not require starting training or evaluation.

## 1. The overall flow

There are three separate jobs: prepare a reusable audio cache, train on source datasets, and evaluate a completed, frozen model on the held-out dataset.

```text
PREPARATION (run separately, not automatically by training)
prepare_cache.py → official train/dev records from data.py
                 → copies of full audio files + supporting metadata in cache

TRAINING
train_folds.sh → train.py for F2, then F3, then F4
                  │
                  ├─ protocol.py: choose the three source domains
                  ├─ data.py: source train/dev paths and labels
                  ├─ wavlm_wa.py: audio → dataset → sampler → batches → model
                  ├─ train.py: loss → backward → optimizer update
                  └─ source dev scores → metrics.py: EER per domain
                       → macro EER → best/last checkpoint → early stopping

EVALUATION (a separate, later operation)
wavlm/evaluate.py → verify completed run and selected checkpoint
                 → source-dev scores → source-only threshold calibration
                 → verify reproduced source macro EER
                 → held-out target scores → metrics.py → metrics.json

AGGREGATION
summarize_results.py → metrics.json from four folds × declared model seeds
                    → per-fold mean/sample standard deviation
                    → equally weighted four-fold mean → summary.json
```

Training does not load the held-out target. Evaluation does not update model weights. Cache preparation does not create features, embeddings or training crops: it copies full audio files so crops can be selected during training.

There is also a generic route: `evaluate_scores.py` reads already-exported, validated score CSVs and produces the same metrics without running a detector. It is reusable across WavLM, AASIST and future models.

## 2. Recommended reading order

| Order | File | What to understand first |
|---|---|---|
| 1 | [protocol.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/protocol.py:18) | Fold definitions and `read_dataset()`. Leave the checksum/manifest helpers for the evaluation chapter. |
| 2 | [data.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/data.py:38) | A reader returns paths and labels; audio loading happens later. |
| 3 | [wavlm_wa.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/sota/wavlm_wa.py:37) | Dataset preprocessing, batching, sampling, and the neural network. |
| 4 | [train.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/wavlm/train.py:205) | How the preceding components are connected into an epoch. |
| 5 | [metrics.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/metrics.py:47) | EER first, then threshold transfer, normalization and confidence intervals. |
| 6 | [WavLM evaluate.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/wavlm/evaluate.py:94) | Frozen checkpoint → source calibration → target testing. |
| 7 | [evaluate_scores.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/evaluate_scores.py:20) | Model-independent evaluation from score files. |
| 8 | [summarize_results.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/summarize_results.py:34) | Combining complete results without pooling target datasets. |
| 9 | [prepare_cache.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/prepare_cache.py:180) | How source audio is copied and checked before experiments. |
| 10 | [tests](/mnt/drive/rohan/audio-deepfake-detection/tests/test_metrics.py), [queue script](/mnt/drive/rohan/audio-deepfake-detection/scripts/wavlm/train_folds.sh) | Examples of expected behavior and operational safeguards. |

`src/audio_deepfake_detection/` contains reusable components. `scripts/` contains executable entry points that assemble those components. A function named `main()` orchestrates a job; functions starting with `_` are internal helpers. The `if __name__ == "__main__"` block starts a script when executed directly, rather than when imported.

## 3. Shared protocol: which data belongs to a run?

File: [protocol.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/protocol.py).

| Fold | Source datasets: train and dev | Held-out target split |
|---|---|---|
| F1 | ASVspoof2019, ASVspoof5, CFAD | SpeechFake / test |
| F2 | ASVspoof2019, ASVspoof5, SpeechFake | CFAD / test_unseen |
| F3 | ASVspoof2019, CFAD, SpeechFake | ASVspoof5 / eval |
| F4 | ASVspoof5, CFAD, SpeechFake | ASVspoof2019 / eval |

Shared labels are `0 = bona fide` and `1 = spoof`. Owner decisions set families `wavlm_bs96`, `aasist` and initial `lacf`/`lacf4s` to seed `(1234,)` only; remaining models retain their declared plans. `final_seeds_for(model)` returns the applicable summary seed policy.

| Function | Input | Output / responsibility |
|---|---|---|
| `validate_run_name()` | Output-family string | Validated safe name; rejects invalid characters. |
| `dataset_root()` | Base directory, dataset name | Dataset's `extracted/` path. |
| `read_dataset()` | Dataset name, split, base directory | List of `(audio_path, label)` records from the appropriate reader. |
| `utterance_id()` | Audio path, dataset, base directory | Relative POSIX path: a stable ID shared by raw and cached copies. |
| `canonical_score_labels()` | Dataset/split/root; optionally already-read records | Dictionary `{utterance_id: label}` specifying exact score-file membership. |
| `checkpoint_digest()` | File path | SHA-256 checksum; also used for score CSVs and provenance files. |
| `verify_completed_run()` | Run directory, fold, seed | Completion metadata and checksum of `best.pt`; rejects incomplete or mismatched runs. |
| `verify_source_macro()` | Observed and recorded macro EER | No return value; rejects nonfinite, invalid or mismatched EERs. |
| `load_score_manifest()` | Run identity and checkpoint checksum | Validated manifest dictionary; optionally creates a fresh one. |
| `save_score_manifest()` | Run directory and manifest | Writes manifest atomically through a temporary file. |
| `verify_score_digest()` | CSV path and manifest | Rejects unregistered or modified score files. |

The manifest binds scores to the selected checkpoint, completion record, fold, seed, model family and intended splits. Checksums establish file identity; canonical IDs/labels establish split membership.

## 4. Dataset readers and audio loading

File: [data.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/data.py).

A `Record` is a tuple `(path_string, label_integer)`. Reading records is different from decoding audio: readers identify examples, while `load_audio_segment()` opens their waveforms.

| Function | Input | Output / processing |
|---|---|---|
| `read_asvspoof2019()` | Dataset's extracted root and train/dev/eval split | Records defined by the official LA protocol and `.flac` paths. |
| `read_asvspoof5()` | Extracted root and split | Records defined by official Track 1 protocol entries. |
| `read_cfad()` | Extracted root and train/dev/test_seen/test_unseen split | Sorted `.wav` records from the appropriate clean real/fake directories. |
| `read_speechfake()` | Extracted root and train/dev/test split | Records from official baseline CSV metadata, with duplicate/overlap safeguards. |
| `_speechfake_metadata_path()` | Root and split | Metadata CSV path. |
| `_read_speechfake_metadata()` | Root and split | CSV rows as dictionaries. |
| `_index_speechfake_rows()` | Metadata rows and split name | Canonical relative-path index; compares normalized metadata fields and rejects conflicts/invalid paths. |
| `load_audio_segment()` | Audio path; optional seconds and random-crop flag | Mono float32 waveform `[samples]` and native sample rate. Reads the selected physical segment rather than decoding a whole long file for a short crop. |
| `resample_audio()` | Waveform, native rate, desired rate | Waveform resampled to the desired rate; unchanged when rates match. |

SpeechFake train/dev overlap handling is explicit: consistent shared entries are removed from train; conflicting metadata raises an error. It does not silently choose one conflicting label.

`load_audio_segment(seconds=None)` reads the full utterance. For a requested segment, long files use a random crop when requested, otherwise a centered crop; short files retain their available samples. Channels are averaged to mono. Padding and WavLM's 16 kHz requirement belong to the next file, not the generic readers.

Study question: can you follow one `(path, label)` from a reader into the waveform returned by `load_audio_segment()`?

## 5. WavLM data pipeline and neural network

File: [wavlm_wa.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/sota/wavlm_wa.py).

### Dataset and loaders

| Class / function | Input | Output / responsibility |
|---|---|---|
| `WavLMDataset.__init__()` | Records and `training` flag | Dataset storing records and choosing train versus evaluation preprocessing. |
| `WavLMDataset.__len__()` | Dataset instance | Number of records. |
| `WavLMDataset.__getitem__()` | Record index | `(waveform, attention_mask, label)` for one example. |
| `collate_wavlm_batch()` | List of dataset samples | Padded waveforms `[B, T_max]`, Boolean masks `[B, T_max]`, integer labels `[B]`. |
| `make_wavlm_sampler()` | One dataset per source domain, seed, optional number of draws | Replacement sampler giving approximately equal probability to each domain. |
| `make_wavlm_train_loader()` | Source datasets, batch size, seed, workers, prefetch and pin-memory settings | Combined DataLoader using the domain-balanced sampler. |
| `make_wavlm_eval_loader()` | One dataset and loader settings | Sequential, non-shuffled DataLoader for development or testing. |

Training example: random continuous 4-second segment at the native rate → resample to 16 kHz → truncate/pad to exactly 64,000 samples. The attention mask marks valid samples and excludes padding.

Development/testing example: full utterance → resample to 16 kHz → variable-length waveform. Current evaluation batch size is one. There is no training-style random 4-second crop in the main evaluation path.

The training sampler assigns each example weight `1 / size_of_its_domain`. Consequently, each domain has equal total sampling weight, but within-domain class proportions remain natural. This is domain balancing, not class balancing. One epoch uses `sum(source_training_sizes)` draws with replacement; it does not mean every utterance is visited exactly once.

Worker processes decode/prepare audio on CPU; they are not additional GPU training processes. Prefetch factor is batches queued per worker. Persistent workers and prefetch settings are enabled only when worker count is positive. Pinned memory helps transfer CPU tensors to CUDA.

### Model

| Method | Input | Output / responsibility |
|---|---|---|
| `WavLMWA.__init__()` | Pretrained model name, normally `microsoft/wavlm-base` | WavLM encoder, 13 learned layer weights, and linear `768 → 2` classifier. Internal SpecAugment is disabled. |
| `WavLMWA.forward()` | Waveforms `[B, audio_samples]` and sample masks of the same shape | Two unnormalized class logits per example, `[B, 2]`. |
| Nested `capture_frontend()` | Feature-projection module's forward output | Captures the projected frontend representation for the weighted mixture. |
| `WavLMWA.spoof_score()` | Logits `[B, 2]` | One raw score `[B]`: `spoof_logit - bona_fide_logit`. Higher means more spoof-like. |

The model mixes the projected frontend and all 12 Transformer-layer outputs. Softmax makes the 13 learned mixing weights sum to one. The frontend is captured with a temporary forward hook; `hidden_states[0]` is deliberately not substituted for that representation. Masked temporal averaging excludes padded frames before classification.

| Stage for a full training batch | Tensor shape |
|---|---|
| Input audio / input mask | `[96, 64000]` |
| Labels | `[96]` |
| Frontend plus 12 Transformer outputs stacked | `[13, 96, T_feature, 768]` |
| Learned weighted mixture | `[96, T_feature, 768]` |
| Masked temporal mean | `[96, 768]` |
| Classifier logits | `[96, 2]` |
| Spoof scores | `[96]` |
| Cross-entropy loss | Scalar |

`T_feature` is the shorter frame sequence produced by WavLM's convolutional frontend, not 64,000 audio samples. The final partial training batch can have fewer than 96 examples.

Study question: why must the audio-sample mask be converted to a feature-frame mask before temporal pooling?

## 6. Training orchestration

File: [train.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/wavlm/train.py).

| Function | Input | Output / responsibility |
|---|---|---|
| `parse_args()` | Command-line arguments | Validated Namespace: fold, seed, roots, batch size, output family, workers and prefetch. |
| `set_seed()` | Integer seed | Seeds Python, NumPy, PyTorch and CUDA generators. |
| `evaluate_eer()` | Model, source-dev loader, CUDA device | One domain's EER fraction, using inference mode and full utterances. |
| `main()` | Parsed arguments and source data | Trains a fold, selects checkpoints, writes configuration/completion files and prints progress. |

Read `main()` in this order:

1. Resolve fold and output directory; require CUDA; reject existing checkpoints rather than overwriting them.
2. Read only the three sources' train/dev records.
3. Build training and development datasets/loaders.
4. Load WavLM and construct the two-group Adam optimizer, scheduler and weighted loss.
5. For every batch: move tensors to CUDA → zero gradients → forward/loss under BF16 autocast → backward → Adam step.
6. After every epoch, score each source-dev domain separately and average its EER with equal domain weight.
7. Save `last.pt`; save `best.pt` only for a strictly lower macro source-dev EER. Ties count as non-improvement.
8. Stop after five consecutive non-improving epochs, or 100 epochs. On normal completion, write `config.json` and `training_complete.json`.

Current defaults: training batch 96; development batch 1; training workers 16; development workers 4 per domain; prefetch 2; encoder LR `2e-5`; layer-weight/classifier LR `5e-3`; Adam; exponential LR multiplier `0.95` after each epoch; weighted cross-entropy with bona-fide/spoof weights `(9, 1)`; BF16 autocast.

Both encoder and backend are trained. Cross-entropy receives logits directly, not already-softmaxed probabilities. The accumulation setting is one; the loop currently updates the optimizer every batch and does not implement a general multi-batch accumulation loop. Loader-helper defaults differ from entry-point defaults: the arguments passed by `train.py` determine the actual run.

Scheduler stepping occurs before checkpoints are written, so stored scheduler/optimizer state contains the learning rates for the next epoch. Checkpoints hold model, optimizer and scheduler state, epoch statistics, patience state and recipe configuration. The script has no resume entry point despite saving those states.

Displayed loss is the running average of batch losses. EER is printed as a percentage, but internally stored as a fraction. Target metrics are never used for early stopping.

## 7. Metrics and normalization

File: [metrics.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/metrics.py).

| Function | Input | Output |
|---|---|---|
| `_scores()`, `_pair()`, `_both_classes()` | Scores and/or labels | Validated arrays or validation only; reject malformed/nonfinite inputs and missing classes where required. |
| `compute_eer()` | Binary labels, higher-is-spoof scores | Interpolated equal-error rate, a fraction in `[0, 1]`. |
| `compute_auroc()` | Labels and scores | Area under ROC, with spoof as the positive class. |
| `compute_apcer()` | Labels, scores, fixed threshold | Fraction of spoofs with `score < threshold`. |
| `compute_bpcer()` | Labels, scores, fixed threshold | Fraction of bona-fide examples with `score >= threshold`. |
| `compute_acer()` | Labels, scores, threshold | Mean of APCER and BPCER. |
| `calibrate_apcer_thresholds()` | Source-dev labels/scores, requested rates | Threshold dictionary for source APCER operating rates, normally 1%, 5%, 10%. |
| `zscore_stats()` | A domain's scores | Mean and population standard deviation (`ddof=0`). |
| `normalize_scores()` | Scores and those statistics | Z-scores `(score - mean) / std`. |
| `_operating_metrics()` | Labels, scores, threshold dictionary | APCER/BPCER/ACER for each operating rate. |
| `calibrate_source_scores()` | `{domain: (labels, scores)}` | Per-domain EER/statistics, macro EER, frozen raw and normalized thresholds. |
| `_bootstrap_intervals()` | Target arrays, fixed thresholds, resampling count/seed | Class-stratified 95% percentile intervals with explicit assumptions. |
| `evaluate_transfer()` | Source-dev dictionary and target labels/scores | Complete report containing ranking metrics, both transfer branches, normalization and optional intervals. |
| `write_score_csv()` | Destination and iterable of `(ID, dataset, label, score)` rows | Atomic score CSV; refuses an existing destination. |
| `read_score_csv()` | CSV, expected dataset, optional canonical ID→label mapping | Label/score arrays; rejects invalid rows and, when specified, nonexact split membership. |

There are two operating-threshold branches:

- **Strict raw transfer:** pool source-dev raw scores, derive thresholds from source spoof scores, freeze them, then apply them to raw target scores.
- **Unlabeled z-score transfer:** normalize each source-dev domain separately, pool its normalized scores and derive thresholds. Normalize target scores using the target score distribution's own mean/std, without using target labels for those statistics. Apply the frozen normalized thresholds. This branch is transductive, not strict zero-shot.

Threshold rule: sort pooled source spoof scores ascending and select index `floor(rate × number_of_spoofs)`. With `score >= threshold` predicting spoof, source APCER is at most the nominal rate, including ties. The target APCER need not equal the nominal source rate.

Normalization standardizes the **scores**, not the audio waveform or model weights. Z-scores are not probabilities and are not bounded to `[0, 1]`; raw logit differences are also unbounded. Error-rate metrics are fractions: `0.012` means `1.2%`. A positive-scale z-score preserves ranking, so target EER/AUROC are reported once. Constant or nearly constant score distributions are rejected rather than dividing by a tiny standard deviation.

Bootstrap defaults: 1,000 resamples, seed 2026, resampling target observations within each class. Source thresholds and all normalization statistics remain fixed. These intervals measure target-sampling uncertainty conditional on the fitted model/calibration, not training or calibration uncertainty. Across-seed variation is summarized separately.

Study question: where are target labels used for reporting, and why must they not select the transferred thresholds?

## 8. Frozen-model evaluation

File: [WavLM evaluate.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/wavlm/evaluate.py).

| Function | Input | Output / responsibility |
|---|---|---|
| `verify_checkpoint()` | Loaded checkpoint, completion metadata, fold, seed, run family | Validates selected epoch, identity, fold membership and recorded source macro EER. |
| `score_rows()` | Model, sequential loader, records, dataset/root/device | Generator of CSV rows containing stable ID, dataset, label and raw score; checks output count and record-label order. |
| `score_split()` | Model, dataset/split, roots, output path, loader settings, manifest | Label/score arrays; either verifies an existing registered CSV or scores the complete split and registers its checksum. |
| `main()` | CLI fold/seed/family/roots/settings | Loads `best.pt`, scores source dev first, verifies source macro EER, then scores target and writes `metrics.json`. |

Important detail: the evaluation CLI defaults to the legacy family `wavlm`. Batch-96 runs require explicit `--run-name wavlm_bs96` when evaluation is eventually authorized. Evaluation is full-utterance and batch one.

The evaluator refuses to replace existing metrics. A valid completion marker and checkpoint/score provenance are required; merely having a `best.pt` file from an interrupted run is insufficient.

The [experiment protocol](/mnt/drive/rohan/audio-deepfake-detection/docs/EXPERIMENT_PROTOCOL.md) records the comparison-wide design/recipe freeze and the owner's 2026-10-06 exception permitting independent completed WavLM fold evaluation. The scripts verify per-run integrity and source-before-target checks; an operator acknowledgment is not an automated scientific audit. The exception does not permit target-guided tuning or claim that the wider comparison is frozen.

### Generic evaluator

File: [evaluate_scores.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/evaluate_scores.py).

`evaluate_run(run_dir, fold_name, seed=..., source_data_root=..., target_data_root=..., bootstrap_resamples=...)` reads validated score CSVs, checks completion/manifest/checksums/canonical membership, calibrates source thresholds before opening target membership, then returns a report and writes `metrics.json`. Its `main()` parses CLI arguments and prints key results.

Unlike the WavLM evaluator, it does not construct a model, decode audio or generate scores. It still needs official metadata/directory membership to validate the supplied CSVs, plus the common completion/checkpoint/manifest contract. Future models need compatible score exporters; those are not automatically provided by this file.

## 9. Aggregating results for the paper

File: [summarize_results.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/summarize_results.py).

| Function | Input | Output / responsibility |
|---|---|---|
| `flat_metrics()` | One metrics report | Flat dictionary of 20 target metrics: EER, AUROC and 18 operating-point metrics. |
| `summarize()` | Output root, model family, seeds, optional non-final flag | Per-fold mean/sample standard deviation and equal-weight mean across four folds; validates report identities and required uncertainty metadata. |
| `main()` | CLI arguments | Writes `summary.json`, or `exploratory_summary.json` in non-final mode; refuses overwrites. |

Final mode requires all four folds. Families `wavlm_bs96`, `aasist` and initial `lacf`/`lacf4s` default to seed 1234 only: four evaluated runs, `training_seed_count=1`, `across_seed_variability_estimated=false`, and `sample_std=null`. Other models still require seeds 1234, 2345, 3456: twelve evaluated runs. When multiple seeds exist, standard deviation is across training seeds, using `ddof=1`. A single-seed standard deviation is unavailable, not zero. Dataset sizes do not determine the headline four-fold weighting; target observations are not pooled across folds. Target-bootstrap intervals remain in each run's report and do not substitute for seed replication.

The batch-96 family is separate from historical batch-64 WavLM runs. A completed batch-64 F1 must not silently fill the batch-96 family's F1 slot; a consistent final batch-96 table requires its own F1 run. The F2→F3→F4 queue alone does not provide F1, and an interrupted F4 cannot count as a completed fold. A three-fold report remains partial.

## 10. Cache preparation

File: [prepare_cache.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/prepare_cache.py).

| Function | Input | Output / responsibility |
|---|---|---|
| `parse_args()` | Dataset names, source/cache roots, worker count | Validated Namespace; requires at least one copying worker. |
| `dataset_root()` | Base directory and dataset name | Dataset extracted-root path used by the cache script. |
| `read_records()` | Dataset name, extracted root, split | Official source train/dev records. |
| `copy_support_files()` | Dataset, source/cache extracted roots | Copies train/dev protocol or metadata files needed by cached readers. |
| `copy_audio()` | Source path and source/cache extracted roots | Copies audio preserving relative path; Boolean indicates whether a copy was made. Existing same-size copies are skipped. |
| Nested `worker()` | Source audio path | Delegates one copy to `copy_audio()`. |
| `cache_dataset()` | Dataset name, source/cache base roots, workers | Copies unique train/dev audio concurrently; verifies cached reader counts and file presence. |
| `main()` | Parsed arguments | Validates root safety and prepares each requested dataset. |

Raw source files are not modified. Source/cache paths are distinct, and unsafe cache locations are rejected. Existing-copy checks are size-based, not cryptographic content verification. Target test/eval audio is not cached by this train/dev job.

The separate [prepare_test_cache.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/prepare_test_cache.py)
copies the four `FOLDS` target partitions with the same readers. `copy_verified()`
checks source/cache containment, SHA-256 verifies full file bytes, refuses any
existing-cache mismatch, and publishes new files atomically without replacement.
`cache_target()` copies target metadata, records path/label/size/hash journals,
and compares cached canonical IDs and labels with the original split before
writing a partition completion marker. `main()` takes the cache lock, applies
bounded copying and disk-space guards, and writes `all_complete.json` only
when all requested partitions pass. This CPU-only cache job performs no model
inference or audio transformation. Require the complete markers before future
authorized evaluations use the target cache; retain valid existing scores.
Synthetic preservation/membership checks live in `tests/test_test_cache.py`.


## 11. Output files: what creates and consumes them?

Current batch-96 layout: `outputs/wavlm_bs96/<fold>/<seed>/`.

| Artifact | Created by | Consumed by / meaning |
|---|---|---|
| `training.log` | Queue's `tee` logging | Human progress inspection; `train.py` itself prints to stdout. |
| `last.pt` | Trainer after each epoch | Most recently finished epoch; not necessarily the selected model. |
| `best.pt` | Trainer on source macro-EER improvement | Selected checkpoint used for evaluation. |
| `config.json` | Trainer on normal completion | Human-readable final run recipe; checkpoint configuration exists earlier inside `.pt` files. |
| `training_complete.json` | Trainer on normal completion | Completion/selected-epoch record required by evaluators. |
| `source_dev_<domain>.csv` | Model-specific evaluator | Source calibration and common evaluator. |
| `target_scores.csv` | Model-specific evaluator after source checks | Held-out target metrics. |
| `score_manifest.json` | Evaluation helpers | Checkpoint/completion/split/CSV identity and integrity checks. |
| `metrics.json` | WavLM or generic evaluator | One fold/seed report, including uncertainty and provenance. |
| `summary.json` at model-family root | Summary script | Publication aggregation across four folds using the model's declared seed policy. |

A saved best checkpoint alone does not mean training is complete. A completion marker does not mean held-out evaluation has occurred. A metrics report is separate from both.

## 12. Remaining files and tests

- [train_folds.sh](/mnt/drive/rohan/audio-deepfake-detection/scripts/wavlm/train_folds.sh) is Bash, not Python. It activates the environment, queues F2→F3→F4 at batch 96, logs output and uses a family-wide lock to prevent a second queue. Failure or a missing completion marker stops subsequent folds. It does not launch target testing or resume interrupted runs.
- [activate.sh](/mnt/drive/rohan/audio-deepfake-detection/activate.sh) configures the shared environment and project paths. [pyproject.toml](/mnt/drive/rohan/audio-deepfake-detection/pyproject.toml) declares package/dependency settings; `uv.lock` records dependency resolution.
- [package __init__.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/__init__.py) is a package initializer. Training, scoring and cache commands use the documented entry points under `scripts/`; there is no package-level greeting command. `sota/__init__.py` is an empty package initializer.
- [test_metrics.py](/mnt/drive/rohan/audio-deepfake-detection/tests/test_metrics.py): small numerical examples for ranking, thresholds, normalization, bootstrap and score validation.
- [test_evaluation_pipeline.py](/mnt/drive/rohan/audio-deepfake-detection/tests/test_evaluation_pipeline.py): completed-run gates, canonical membership, manifests/checksums, source-before-target behavior and summary checks.
- [test_data_cache.py](/mnt/drive/rohan/audio-deepfake-detection/tests/test_data_cache.py): SpeechFake duplicates/overlaps/path handling and cache argument checks.
- [test_training_queue.py](/mnt/drive/rohan/audio-deepfake-detection/tests/test_training_queue.py): batch/family settings, queue order, failure handling and concurrency locking using temporary/fake runs.
- [README.md](/mnt/drive/rohan/audio-deepfake-detection/README.md) is the operational starting point; [EXPERIMENT_PROTOCOL.md](/mnt/drive/rohan/audio-deepfake-detection/docs/EXPERIMENT_PROTOCOL.md) records research rules; [SOTA_PAPER_NOTES.md](/mnt/drive/rohan/audio-deepfake-detection/docs/SOTA_PAPER_NOTES.md) records paper interpretations. Documentation describes intended policy; executable code determines what is actually implemented.

For a first study session, trace just one batch: `train.main()` → `read_dataset()` → `WavLMDataset.__getitem__()` → `collate_wavlm_batch()` → `WavLMWA.forward()` → cross-entropy → `backward()` → optimizer step. Then trace one epoch's source-dev evaluation and best-checkpoint decision. Leave target evaluation and bootstrap internals until that path is clear.

## 13. AASIST: the second baseline

The shared `data.py`, `protocol.py`, `metrics.py` and summary contract remain unchanged by the AASIST addition. Its code is deliberately limited to one adapter, the licensed upstream architecture, two entry points and a queue:

```text
scripts/aasist/train_folds.sh
  → scripts/aasist/train.py
    → protocol.read_dataset() → data readers
    → sota/aasist.py: AASISTDataset + loaders
    → sota/aasist_arch.py: official AASIST network
    → CE loss, Adam, per-update cosine
    → separate source-dev EERs → best/last/completion

scripts/aasist/evaluate.py
  → completed best checkpoint + explicit freeze acknowledgement
  → source score CSVs/calibration/verification → target score CSV
  → shared metrics.py → metrics.json → summarize_results.py
```

| File / function | Input | Output / purpose |
|---|---|---|
| [aasist.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/sota/aasist.py): `fixed_length_audio()` | Mono 16 kHz waveform and train/eval flag | Exactly 64,600 samples: random train crop, first eval crop, repeated short signal. |
| `AASISTDataset.__getitem__()` | Record index | Fixed waveform `[64600]` and project label; full audio is loaded and resampled before length handling. |
| `make_train_loader()` / `make_eval_loader()` | Source datasets / one dev-test dataset and worker settings | Domain-balanced training batches `[24,64600]` / sequential batch-one evaluation. No padding mask is needed for repeated fixed-length input. |
| `AASIST.forward()` / `spoof_score()` | Fixed waveform batch / two-logit batch | Project-order logits `[B,2]` / higher-is-spoof raw scores `[B]`. |
| [aasist_arch.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/sota/aasist_arch.py): `Model.forward()` | Fixed waveform batch | Original `(embedding, logits)` tuple; adapter reorders its original class columns. Architecture is unchanged. |
| Architecture helpers | Sinc convolution features and graph nodes | Residual CNN → spectral/temporal graph attention → heterogeneous graph branches/pooling → max-graph operation → readout/classifier. |
| [AASIST train.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/aasist/train.py): `parse_args()` / `set_seed()` | CLI / integer seed | Validated settings / reproducible RNG initialization. |
| `make_optimizer()` | Model and steps per epoch | Adam + cosine scheduler spanning `100 × steps_per_epoch` updates. |
| `train_epoch()` | Model, loader, optimizer, scheduler, device | Updates parameters in FP32 and returns example-weighted average batch loss. |
| `evaluate_eer()` | Model and one dev loader | Source-domain EER; inference only, no weight updates. |
| `code_provenance()` / `write_json()` | Project root / destination and dictionary | Code hashes and Git identity / atomic metadata write. |
| `main()` | Source-only run arguments | Complete recipe, epoch checkpoints and completion metadata including runtime/memory information. |
| [AASIST evaluate.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/aasist/evaluate.py): `verify_checkpoint()` | Checkpoint, completion and requested identity | Validation only; rejects incompatible identity, architecture, precision, crop or epoch. |
| `score_rows()` / `score_split()` | Model, records, loader / split and manifest | CSV row generator / validated label-score arrays, registering score hashes after export. |
| `parse_args()` / `main()` | CLI with freeze acknowledgement | Completed-run validation → source calibration → target scoring → shared report. |

AASIST is randomly initialized, not an SSL encoder. Its train/eval duration policy differs from WavLM's full-utterance evaluation. Its class weights are `[0.9,0.1]`; the encoder/head use one Adam group at `1e-4`, weight decay `1e-4`, with cosine minimum `5e-6`. No BF16, RawBoost, frequency masking or SWA is silently added. See README and the protocol for disclosed deviations from the released training workflow.

The AASIST queue covers all four folds for one seed and refuses to start while the existing batch-96 WavLM queue is active. It is not an automatic handoff. [test_aasist.py](/mnt/drive/rohan/audio-deepfake-detection/tests/test_aasist.py) checks audio handling, exact upstream logits after class reordering, CPU forward/backward/reload, checkpoint writing, target gates and metric export. [test_aasist_queue.py](/mnt/drive/rohan/audio-deepfake-detection/tests/test_aasist_queue.py) checks queue order and locking without real training.

## 14. P2–P4 evaluation flow (2026-10-05)

Update 2026-10-06: the owner authorizes native P1 evaluation of completed
WavLM folds independently, while retaining source-only checkpoint/threshold
selection and no target feedback into any model's development. In
`scripts/wavlm/evaluate.py`, `--confirm-run-frozen` records that policy without
claiming the wider comparison is already frozen. `scripts/wavlm/evaluate_folds.sh`
queues F1→F2→F3 sequentially and stops on failure. Each fold saves separate
metrics/scores/logs; `summarize_results.py --non-final --folds f1 f2 f3` makes
an explicitly partial summary. Later, `evaluate_folds.sh f4` evaluates only F4
and builds the four-fold summary from all existing reports. Synthetic tests
cover queue ordering, failure/lock behavior, truthful freeze flags and adding
F4 without changing earlier results. The P2–P4 supplementary gates below are
separate and unchanged by this P1 launch.

Start with [evaluate_extended.py](/mnt/drive/rohan/audio-deepfake-detection/scripts/evaluate_extended.py).
Input is a completed run directory, evaluation protocol/dataset, raw/source roots,
freeze acknowledgment and, for P2, the source-only single-checkpoint selection
record. Output is a separate `analyses/<protocol>/<dataset>/metrics.json`, with
registered additional score CSVs only if inference is explicitly requested.

Flow: completed checkpoint/hash → registered canonical source dev scores →
source-only calibration and macro-EER verification → official target metadata →
verified P1 scores or explicitly requested native inference → common reports.
Default reporting never constructs a GPU model. Record-writing mode exits
before opening external targets. WavLM/AASIST source scorers and checkpoint
verifiers are reused; their training files and native preprocessing are unchanged.

| File/function | Input | Output |
| --- | --- | --- |
| [evaluation_data.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/evaluation_data.py), `native_examples()` | Dataset, official split, raw root | Exact P1 membership enriched with generator, attack or codec metadata |
| `asv2021_df()` / `partialspoof()` / `mlaad_mailabs()` | Raw root plus official DF keys or genuine M-AILABS root where required | External eval examples with stable ID, waveform path, binary label and annotations |
| `cfad_conditions()` | Raw root, unseen or seen test partition | Clean/noisy/codec examples, noise/SNR/codec groups and class/source-qualified original IDs |
| [analysis.py](/mnt/drive/rohan/audio-deepfake-detection/src/audio_deepfake_detection/analysis.py), `evaluate_frozen()` | Source calibration, target labels/scores, optional fixed parent statistics | EER/AUROC, fixed raw/transductive operating metrics and conditional bootstrap intervals |
| `group_reports()` | Examples/scores, calibration, group dimensions | Generator/attack rows sharing parent genuine reference, or codec/noise rows with native group membership; all retain parent statistics |
| `paired_raw_deltas()` | Declared original IDs, condition scores, source thresholds | Descriptive matched-original APCER/BPCER differences; variant-balanced per original |
| `source_scores()` / `verify_selection()` | Completed run/registered source files or P2 selection identity | Verified calibration or a clear refusal before target access |
| `export_rows()` | Explicit native model adapter, examples, unchanged batch-one loader | Shared `utterance_id,dataset,label,raw_score` rows, preserving order and spoof orientation |

All rates are fractions. Single-class groups have `null` ranking/ACER, not
fabricated metrics. Paired differences are signed fractions. Do not interpret
these many subgroup intervals as corrected significance tests or training-seed
uncertainty. See README for commands and protocol for scientific assumptions.

Run synthetic tests with the GPU hidden from **that test process** while other
jobs train (pinned-memory loaders can otherwise initialize CUDA even in a CPU test):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider
```

Tests in [test_extended_evaluation.py](/mnt/drive/rohan/audio-deepfake-detection/tests/test_extended_evaluation.py)
cover schemas, safe paths, membership, source-first gates, fixed normalization,
single-class metrics, genuine-reference deduplication, paired IDs, mock CPU
export, immutable selection records, supplementary provenance and score-tamper
rejection. This does not establish real-data/GPU inference validation.

## Local reporting artifacts

The [supervisor report folder](wavlm_f1_f3_report/README.md) holds ignored report
sources and exports. Its numerical extraction reads completed native P1 score
CSVs and run records, verifies provenance and point metrics, and reuses saved
confidence intervals. Report generation does not train, run inference, repeat
bootstrap experiments or modify original outputs. The report README explains
the standalone LaTeX source, plotting files and separate PDF reading-copy export.
These helpers are reporting utilities rather than model entry points.

The [concise companion report](wavlm_f1_f3_short_report/README.md) reuses those
verified aggregate results for a shorter results note with an explicit bootstrap
explanation. Its local builder preserves the original detailed report and performs
no model inference or bootstrap rerun.

## 15. LACF: primary frozen model and training

LACF code is separate from `sota/`. Its flow is:

```text
scripts/lacf/train.py
  → lacf/pretrained.py: cached frozen encoders/processors and 24 text prompts → 8 prototypes
  → protocol.read_dataset(): three official source train/dev partitions only
  → lacf/data.py: one native-rate segment → 16/48 kHz views; domain/class-balanced sampler
  → lacf/pretrained.py: processor collation, valid sample masks, CLAP features
  → lacf/model.py: masked speech pool → adapter → anchor distributions → relation head
  → lacf/training.py: configured losses/accumulation → trainable-parameter AdamW
  → lacf/cache.py: fixed source-dev frozen features (encode once, current backend every epoch)
  → shared metrics.compute_eer(): separate source domains → equal-domain macro selection
  → best/last checkpoints → selected source-dev reload → completion record
```

| File / component | Responsibility |
|---|---|
| `lacf/data.py`: `LACFDataset` | Reuses shared audio loading/resampling. Selects a random train or centered dev segment once at the original rate. Short audio remains available signal before processor padding. |
| `balanced_sampler()` / `make_loader()` | Equal domain and class probability, uniform within each bucket, N replacement draws, no dropped final batch; development is sequential batch one. |
| `lacf/pretrained.py`: `load_local_model()` | Local-only WavLM Base+/unfused CLAP loads in FP32; processor revisions follow cached model revisions. Builds exact prompts and fixed normalized prototype buffer without gradients. |
| `DualViewCollator` | WavLM right padding with validity masks and no waveform normalization; CLAP 48 kHz repeat-padding. Rejects invalid/too-short input and prevents secondary CLAP cropping. |
| `lacf/model.py`: `ComponentConfig` | Explicit feature groups and loss switches/coefficients. Defaults are all 35 primary features and all three losses. |
| `LACF` | Frozen encoders remain eval/no-grad; only adapter/head train by default. Converts sample masks to WavLM feature-frame masks, then masked mean pools. Returns views/features and one spoof logit. |
| `RelationFeatures` / `relation_classifier()` | Ordered probability/disagreement/agreement/JS/entropy concatenation and dimension-derived 64→16→1 head. Explicit replacement fusion declares its dimension and configuration. |
| `LACFLoss` | Unweighted detection BCE, spoof-group semantic BCE and bona-fide-only JS; undefined/disabled auxiliary losses are omitted explicitly, and an all-spoof batch has zero consistency contribution. |
| `lacf/precision.py` | Explicit FP32 LayerNorm/GroupNorm/BatchNorm boundaries preserve encoder state names and values. A CLAP projection output hook casts before its internal L2 normalization. |
| `lacf/training.py`: `train_epoch()` | BF16 autocast with FP32 objective, actual-microbatch-count accumulation and finite-loss/gradient checks; loop consumes the configured objective without ablation-specific branching. |
| `lacf/cache.py`: `frozen_development()` | Source-only centered crops; persistent FP32 WavLM/CLAP features bound to records, duration, processors, encoder/code revisions and environment. Rejects trainable encoders and corrupt/incomplete caches. No random train crops or backend outputs cached. |
| `source_eer()` / `make_optimizer()` | Shared EER on source scores / AdamW on enabled trainable parameters only. |
| `scripts/lacf/train.py` | Guards existing outputs/cache acknowledgment/CUDA; the current verified-cache acknowledgment records its evidence source from the passed validation record; the expired owner-waiver flag is removed. Records components, processor policy, provenance, seed plan and recipe. Strict macro source-EER improvement, patience five, selected-checkpoint reload before completion. |
| `scripts/lacf/train_folds.sh` / `verify_fold_completion()` | Fresh F1→F2→F3→F4 at seed 1234; shared completion/hash checks plus LACF recipe and source-reload validation. Any failure stops the queue. No skip/resume. |
| `tests/test_lacf.py` | Synthetic audio, fake encoders and a random tiny WavLM configuration. Tests equations, actual feature-mask conversion, freezing, configuration, accumulation, target isolation and preservation; never loads downloaded weights or real data. |

The owner-approved initial study uses seed 1234 only, BF16 mixed precision and
661,479 trainable adapter/head parameters. Normalization, masked pooling,
anchor probabilities, logarithms, divergences, entropies and losses stay FP32.
Parameters and optimizer state stay FP32; BF16 needs no gradient scaler.
Both training and source development use the same precision policy. Physical
batch, accumulation, workers and prefetch are configurable; defaults remain
4×8 until source-only feasibility chooses the final setting. AdamW stays
3e-4/weight decay 1e-4, constant LR, 30 epochs/patience 5 with no LR scaling.
No training resume is implemented. Synthetic CPU BF16 checks do not establish
actual checkpoint/CUDA integration, memory feasibility or source recipe freeze.
When `LACF_SOURCE_VALIDATION_RECORD` is supplied by the validation supervisor,
the trainer requires a passed record matching the configured recipe, source-code
hashes and encoder revisions, and records its path/hash before creating a run.
No FT4, ablation execution, target scoring or new threshold/CI logic is added.

The subsequent owner-relayed SSH1 checksum-success report clears the transfer
wait; future final runs use `--confirm-cache-verified`. Its original log is not
available locally, so preserve the reported evidence source rather than claim
an independent SSH2 audit. The source-validation freeze gate remains mandatory
for the current launch; no deferred-preflight start path is used.
After separately authorized source/GPU validation, a native LACF exporter still
needs the common completed-checkpoint, score-manifest and source-first gates.

Owner update, 2026-10-09: `--segment-seconds {4,10}` selects separate `lacf4s`
and `lacf` seed-1234 families; `--output-root` protects stopped runs by allowing a
fresh family directory. The queue routes output/completion checks to that root.
The 4 s arm precedes the fresh 10 s arm. `encode_audio()` returns frozen pooled
WavLM/normalized CLAP features; `classify_audio()` recomputes the current adapter,
relations and classifier. Source-dev encoder/backend batches stay one. Reload
checks frozen states before reuse; caches are never a training resume mechanism.

Training loss/gradient checks remain at update boundaries. Loss summaries and
cached scores accumulate on device; CPU transfer occurs at reporting/metric
boundaries. Shared CPU resampling filters are cached with the exact original
dtype/kernel arithmetic. Logs now report phase, counts, percentage, loss,
throughput and ETA. NNPACK is disabled in CPU parent/workers on the unsupported
virtual GPU host. Synthetic tests and bounded official-checkpoint CUDA probes
are distinct from full source-dev/reload and final training completion.

Latest owner decision, 2026-10-09: schedule only fresh10 s F1→F4; the4 s-first
sequence is stopped and deferred. The source freeze explicitly records reused
full native-10 s validation plus bounded optimized-code regression against saved
pilot scores. It does not claim full current-code preflight rescoring. The trainer
records this validation scope. Fixed source-dev features are built after the first
training epoch, during its development phase, then reused. Pilot weights are
validation-only and do not initialize final training. Existing outputs remain
preserved and no resume is introduced.
