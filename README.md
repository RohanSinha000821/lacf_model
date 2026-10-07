# Audio deepfake detection: LODO runs

The project owner's main LACF plan and all supplied supplementary plans, dataset
papers and SOTA papers are organized locally under `references/`. Start with
[references/INDEX.md](references/INDEX.md) for the document map and authority
notes. `AGENTS.md` directs future project work to these references. Full PDF/DOCX
copies are excluded from Git; back up the reference folder separately.

Project notes now live in [docs/](docs/INDEX.md): the experiment protocol,
paper findings, code study guide and historical review. `README.md` remains
at the root for GitHub/package onboarding, and `AGENTS.md` remains here for
instruction discovery. A detailed local-only `docs/PROJECT_HANDOFF.md`
records chat continuity, machine-specific status and unpublished run results;
it is ignored by Git. New chats should read it when available and verify the
live state. A fresh clone needs separate restoration of that handoff, reference
originals and experiment artifacts; it must not pretend those are included.

The fixed four-fold membership, paper-informed recipes, and four fairness
safeguards are in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) (v4). On 2026-10-06 the owner approved
independent evaluation of completed WavLM batch-96 folds before the wider model
comparison is frozen. Each run still uses its source-only selected checkpoint
and thresholds; target feedback must not guide any model's tuning or design.
This is not a claim that LACF or the other methods are already frozen. Other
target-access gates remain as recorded in the protocol. SpeechFake source
train/dev caching runs separately from training.
Do not start F2–F4 until `outputs/speechfake-cache.log` ends with
`verification: PASS` and `Cache preparation complete`.

## Native P1 test-audio cache

`scripts/prepare_cache.py` remains the source train/dev copier.
`scripts/prepare_test_cache.py` adds the four native P1 held-out partitions:
ASV2019 LA eval, CFAD clean unseen test, ASV5 Track 1 eval and SpeechFake test.
It copies full original files and the required protocol/metadata into the same
relative layout under `/mnt/drive/audio-deepfake-cache`; no decoding,
resampling, cropping, feature extraction or model inference is performed.

New files are SHA-256 checked before exclusive atomic publication. Existing
files are reused only after size and SHA-256 agreement; a mismatch fails without
overwriting. A destination lock prevents duplicate cache jobs. Work is bounded
and stops before consuming the configured free-space reserve. Each completed
partition must match canonical IDs and labels exactly. The source train/dev
cache and raw files are preserved. The original train/dev copier's older
same-size reuse policy is unchanged.

From the project root after activation, a CPU-only low-priority copy is:

```bash
export PYTHONPATH=src
CUDA_VISIBLE_DEVICES='' nice -n 15 ionice -c 3 python -u scripts/prepare_test_cache.py --datasets asv2019 cfad asv5 speechfake --workers 2 --reserve-gib 20 --state-dir outputs/test_cache_20261007
```

On 2026-10-07 this job was launched in tmux `test-cache` while the acquisition
session `p2-data-download` was preserved. Inspect its `cache.log`, `status.json`,
per-file hash journals and per-partition completion records under the state
directory. **Do not evaluate a partially populated target cache.** For this
four-partition job, require `all_complete.json` and all four partition records
before choosing the cache as `--target-data-root` for a future authorized
native evaluation. Existing valid fold scores must not be rerun merely because
storage changed. These completion records establish copy/membership integrity,
not an all-file decoding check or a measured speedup. P2 external corpora are
outside this cache job; acquisition and scientific evaluation gates still apply.

## WavLM-WA, one fold at a time

The latest WavLM batch is **96**. New runs use `outputs/wavlm_bs96/<fold>/<seed>/`
so the completed batch-64 F1 under `outputs/wavlm/f1/1234/` is preserved. It must
be retrained at batch 96 before the new family's four-fold publication summary.
Training defaults to `--batch-size 96` and an output family derived from batch;
evaluation uses `--run-name` to select the intended family.

From this project directory, activate the existing Python environment and make
the source package importable:

```bash
source activate.sh
export PYTHONPATH=src
set -o pipefail
```

Run one fold and seed, then evaluate its **completed best checkpoint** under the
owner-approved independent-run freeze or an actually completed wider freeze. The
training command reads source train/dev only; the evaluation command scores
source dev first, calibrates both transfer branches, then opens the held-out
target from the original read-only dataset root.

```bash
mkdir -p outputs/wavlm_bs96/f1/1234
python -u scripts/wavlm/train.py --fold f1 --seed 1234 --batch-size 96 --run-name wavlm_bs96 --data-root /mnt/drive/audio-deepfake-cache --num-workers 16 --eval-workers 4 --prefetch-factor 2 2>&1 | tee -a outputs/wavlm_bs96/f1/1234/training.log
# Owner-approved completed WavLM fold; no target-guided tuning:
python -u scripts/wavlm/evaluate.py --fold f1 --seed 1234 --run-name wavlm_bs96 --source-data-root /mnt/drive/audio-deepfake-cache --target-data-root /mnt/salt/datasets/audio-deepfake --confirm-run-frozen
```

Use seed **1234 only** for WavLM across `f1`, `f2`, `f3`, `f4`, per the owner's
2026-10-05 compute-budget decision. Do not schedule WavLM seeds 2345 or 3456.
The trainer refuses to overwrite existing checkpoints. An
interrupted run lacks `training_complete.json` and cannot be evaluated as a
finished run. Earlier interrupted WavLM pilots were removed; the completed
batch-64 F1 is preserved separately. New batch-96 runs live in
`outputs/wavlm_bs96/<fold>/<seed>/`.

### P2 external tests, P3 generators and P4 robustness

Shared implementation: `scripts/evaluate_extended.py`, `evaluation_data.py`
(official test metadata) and `analysis.py` (frozen thresholds/subgroup reports).
No additional training is needed. Native inference adapters currently exist for
WavLM and AASIST; later models use the same report contract. The correct raw
root is `/mnt/salt/datasets/audio-deepfake`, not `datsets`.

**Do not execute these examples until the comparison-wide freeze is complete.**
They do not resume an interrupted fold or complete missing models' pilots.
Report-only commands use CPU and reuse verified scores, without loading a GPU
model. All rates are fractions; source thresholds stay fixed. Generator rows
share the complete bona-fide reference and parent normalization statistics.

```bash
# After P1 scoring, no extra inference for P3 or ASV5 P4:
python scripts/evaluate_extended.py --protocol p3 --dataset speechfake --run-dir outputs/wavlm_bs96/f1/1234 --confirm-protocol-frozen
python scripts/evaluate_extended.py --protocol p3 --dataset asv5 --run-dir outputs/wavlm_bs96/f3/1234 --confirm-protocol-frozen
python scripts/evaluate_extended.py --protocol p3 --dataset asv2019 --run-dir outputs/wavlm_bs96/f4/1234 --confirm-protocol-frozen
python scripts/evaluate_extended.py --protocol p4 --dataset asv5 --run-dir outputs/wavlm_bs96/f3/1234 --confirm-protocol-frozen

# Additional CFAD clean/noisy/codec test inference, ONLY with GPU availability:
python -u scripts/evaluate_extended.py --protocol p4 --dataset cfad --run-dir outputs/wavlm_bs96/f2/1234 --export-scores --model wavlm --confirm-protocol-frozen
# Secondary seen partition: add --cfad-split test_seen (kept separate).
```

For AASIST, replace the run directory with `outputs/aasist/<fold>/<seed>` and
use `--model aasist` on export commands. Keep its declared seed plan.

P2 first needs a documented **single** source-only deployment-checkpoint choice
per model/seed, before external scores. The following F1 path is an example,
**not a decision that F1 must be chosen**. Replace the reason with the actual
prespecified/source-only selection rule; do not choose a fold on target scores.
Record creation checks source scores and exits without opening external data.
If source CSVs are not yet exported, record creation can add
`--export-scores --model wavlm` to score **source dev only**.

```bash
python scripts/evaluate_extended.py --protocol p2 --dataset asv2021_df --run-dir outputs/wavlm_bs96/f1/1234 --write-selection-record outputs/wavlm_bs96/p2_selection_1234.json --selection-reason 'REPLACE with the documented source-only selection rule' --confirm-protocol-frozen

# Supply the full official DF keys, NOT the unlabeled trial-ID list:
python -u scripts/evaluate_extended.py --protocol p2 --dataset asv2021_df --run-dir outputs/wavlm_bs96/f1/1234 --selection-record outputs/wavlm_bs96/p2_selection_1234.json --df-keys /path/to/keys/DF/CM/trial_metadata.txt --export-scores --model wavlm --confirm-protocol-frozen

# SAME checkpoint/selection record for both stress tests:
python -u scripts/evaluate_extended.py --protocol p2 --dataset partialspoof --run-dir outputs/wavlm_bs96/f1/1234 --selection-record outputs/wavlm_bs96/p2_selection_1234.json --export-scores --model wavlm --confirm-protocol-frozen
python -u scripts/evaluate_extended.py --protocol p2 --dataset mlaad_mailabs --run-dir outputs/wavlm_bs96/f1/1234 --selection-record outputs/wavlm_bs96/p2_selection_1234.json --mailabs-root /path/to/genuine/M-AILABS --export-scores --model wavlm --confirm-protocol-frozen
```

The local ASV2021 trial list lacks labels; its VCTK/VCC mapping is not the CM
key. Official [DF full evaluation keys](https://github.com/asvspoof-challenge/2021/tree/main/eval-package)
were downloaded and checksum-verified on **2026-10-06**, then installed at
`/mnt/salt/datasets/audio-deepfake/asvspoof2021/extracted/keys/DF/CM/trial_metadata.txt`.
That is the reader's default location for the existing dataset root.

MLAAD's installed README requires its referenced genuine M-AILABS files;
unrelated datasets cannot substitute for them. Owner-authorized acquisition
started in tmux session `p2-data-download` on **2026-10-06**. On **2026-10-07**,
the owner requested recovery after an accidental interruption: completed US/UK
English locales are reused, and German extraction resumes by verifying existing
files against the retained archive before filling missing files. The remaining
locales stay queued; this is not completed availability. Required locale
archives are identified from the installed metadata's `original_file` paths,
saved under `/mnt/salt/datasets/audio-deepfake/mailabs/raw/`, and staged before
publication under `mailabs/extracted/<locale>/`. After reference validation,
use `--mailabs-root /mnt/salt/datasets/audio-deepfake/mailabs/extracted`.
Download progress and completion records remain local-only under
`outputs/dataset_downloads/20261006_p2/`; inspect `status.json` and
`mailabs_reference_validation.json` before claiming availability. Acquisition
does not run P2 or establish the comparison-wide freeze.

**MLAAD evaluator prerequisite:** the installed Amharic Edge-TTS metadata has
literal transcript quotes that the current default CSV parser misinterprets,
merging records. The acquisition helper reads only the reference-column prefix;
`evaluation_data.py` is unchanged and still needs a metadata-parser correction
and validation before MLAAD evaluation. PartialSpoof reports utterance detection
only. Generator-family novelty still needs verified cross-dataset annotations;
attack-ID reports do not invent that claim.

Outputs: `outputs/<family>/<fold>/<seed>/analyses/<protocol>/<dataset>/metrics.json`
(CFAD paths include `cfad_test_unseen` or `cfad_test_seen`), plus separately
registered supplementary scores when additional inference is needed. Existing
reports are never overwritten. See `docs/EXPERIMENT_PROTOCOL.md` for normalization,
bootstrap, paired robustness and single-class rules.

### Independent completed-fold P1 evaluation

Owner-approved queue, F1→F2→F3, seed 1234:

```bash
bash scripts/wavlm/evaluate_folds.sh
```

The standard queue uses batch 1, BF16, 8 workers, prefetch 2 and 1,000 bootstrap
resamples, with **no GPU allocator limit**, as requested after the initial 8 GiB
cap rejected a long full recording despite free physical memory. The short
source-only benchmark showed under 0.4% loader waiting; it did not establish a
benefit from increasing workers.

On 2026-10-07 the owner separately approved restarting the active F3 evaluation
with **32 workers**, retaining batch 1 and prefetch 2. Its verified 466,016-row
target prefix was preserved for recovery. This is an operational override for
that F3 run, not a measured speedup or a change to the standard queue defaults.
Native full-utterance input is unchanged. Each run's source-dev reload/calibration
is verified before its target is opened. The queue stops on failure, refuses
existing reports, and takes a family-wide evaluation lock. It records
`comparison_wide_freeze_confirmed_by_operator=false` and the owner-approved
independent-run policy, not a fictional completed global freeze.

Interrupted scores resume from a checksum-verified, checkpoint-bound
`*.csv.progress.json` journal; completed source score files are reused only
after manifest verification. No recording is skipped or shortened. A genuine
OOM still stops the queue and commits the completed prefix. A stale journal
(for example after a forced kill) fails closed instead of trusting new bytes.
Pre-journal `.tmp` files require explicit operator adoption after checking their
origin: `WAVLM_ADOPT_LEGACY_PARTIAL_FOLD=f1 bash scripts/wavlm/evaluate_folds.sh`.
That one-time setting is not required once F1's recovery journal exists.

Reports/scores are kept separately in `outputs/wavlm_bs96/<fold>/1234/`;
each fold has `evaluation.log`. The three-fold aggregate is
`outputs/wavlm_bs96/partial_summary_f1_f2_f3.json`, explicitly non-final and
missing F4. This queue runs native P1 metrics, not all P2–P4 supplementary tests.

After F4 is actually trained, evaluate only the new fold:

```bash
bash scripts/wavlm/evaluate_folds.sh f4
```

That command preserves F1/F2/F3 and automatically creates `summary.json` from
all four existing reports. It does not resume or retrain F4. The complete
summary retains equal-weight per-fold aggregation and the single-seed limitation.
Neither partial nor complete summary pools the target datasets. Existing
summary artifacts are never overwritten.

To queue **F2 -> F3 -> F4**, source-only, for one seed:

```bash
bash scripts/wavlm/train_folds.sh 1234
```

The queue uses batch 96, training workers 16, development workers 4, and prefetch
2. It stops on a failed training/logging command or a missing completion record.
A shared lock prevents concurrent batch-96 queues. It does not resume interrupted
runs, overwrite checkpoints, or start target scoring. Status is recorded in
`outputs/wavlm_bs96/queue_1234.log` and each fold has its own `training.log`.
No additional WavLM seed queues are planned. The stopped F4 run remains
incomplete; this queue cannot resume it or replace its existing checkpoints.

Once all four seed-1234 run reports exist:

```bash
python scripts/summarize_results.py --model wavlm_bs96
```

The WavLM summary defaults to seed 1234 and requires all four completed/evaluated
folds. It reports single-seed point estimates and an equal-weight four-fold mean;
`sample_std` is `null`, not zero. Target-bootstrap intervals remain in each run's
metrics report, but they do not measure variability across training seeds. A
three-fold report is partial, not a complete four-fold result. AASIST also uses
seed 1234 only under the owner's 2026-10-06 decision; remaining models retain
their existing seed plans unless explicitly changed separately.

## AASIST: fixed baseline and source-only validation

The authors' full AASIST architecture (297,866 parameters) is vendored without
changes at `src/audio_deepfake_detection/sota/aasist_arch.py`; its MIT notice is
preserved in `AASIST_LICENSE`. The small `aasist.py` adapter handles resampling,
fixed-length inputs, domain-balanced sampling and the project's label order.
No released ASVspoof-trained checkpoint is loaded, including in F4.

The primary recipe is **batch 24**, FP32, Adam `1e-4`, weight decay `1e-4`,
weighted CE `[0.9, 0.1]`, and per-update cosine decay to `5e-6` over a nominal
100 epochs. Patience is 5 using the same macro source-dev EER as WavLM.
Train inputs are random 64,600-sample segments after resampling to 16 kHz;
dev/test inputs take the first 64,600 samples. Short signals repeat rather
than zero-pad. Frequency masking is off, matching the released config default;
RawBoost is not added to this ordinary AASIST baseline. Upstream SWA and
target-during-training evaluation are omitted under our single-best-checkpoint,
source-only selection policy. This is an official-architecture LODO adaptation,
not a claim of identical published training or published scores.

Synthetic CPU batch-24 forward/backward/Adam checks passed with CUDA hidden.
On 2026-10-07, the owner authorized AASIST F1 training alongside test-audio
caching and ongoing acquisition. A real source-only GPU feasibility check
passed three batch-24 optimizer steps and a checkpoint-reload score check on
six source-dev examples. Fresh F1 training then started in tmux `aasist-f1`,
seed 1234; the pilot checkpoint is not used for final training. Complete
source-development scoring and selected-checkpoint reload validation remain
pending; this short check is not full validation or a completed experiment.
Use 8 training workers, 4 dev workers and prefetch 2. These are initial loader
settings, not a benchmarked optimum. Inspect actual logs under
`outputs/aasist/f1/1234/` and local pilot records under
`outputs/aasist_launch_20261007/`; dated launch observations are not live status.
Preserve other GPU users and obtain authorization for competing work.

After activation and `export PYTHONPATH=src`, a single source-only fold is:

```bash
mkdir -p outputs/aasist/f1/1234
set -o pipefail
python -u scripts/aasist/train.py --fold f1 --seed 1234 --num-workers 8 --eval-workers 4 --prefetch-factor 2 2>&1 | tee -a outputs/aasist/f1/1234/training.log
```

Alternatively, queue F1 → F2 → F3 → F4 for one seed:

```bash
bash scripts/aasist/train_folds.sh 1234
```

The queue refuses to launch while the existing batch-96 WavLM queue lock is held,
prevents duplicate AASIST queues, and stops on any training/logging failure or
missing completion marker. It never launches target testing. Launch it manually
after WavLM. The implementation did not itself authorize a launch; the owner
separately authorized the F1 launch recorded above. No additional folds were queued.
**Owner decision, 2026-10-06: AASIST uses seed 1234 only across F1–F4.**
Do not schedule seeds 2345/3456. This is a disclosed compute-budget adaptation
to the paper's three-run reporting, not a claim of seed robustness. Existing
checkpoints cannot be overwritten or resumed by this trainer.

Only after the comparison-wide freeze is complete and the selected run has
completed, evaluate its best checkpoint:

```bash
python -u scripts/aasist/evaluate.py --fold f1 --seed 1234 --confirm-protocol-frozen
```

The flag is an explicit operator acknowledgement of the research gate, not an
automatic audit of every model's freeze record. Source calibration/checkpoint
verification still precedes target access. Outputs and metrics use the same
contract as WavLM under `outputs/aasist/<fold>/1234/`; after all four seed-1234 reports:

```bash
python scripts/summarize_results.py --model aasist
```

## Shared metrics for every SOTA model

Each model writes `source_dev_<dataset>.csv` for the three active source domains
and `target_scores.csv` in its run directory. CSV columns are
`utterance_id,dataset,label,raw_score`, with label 0=bona fide, 1=spoof, and a
larger raw score meaning more spoof-like. The same evaluator then works for any
model:

```bash
python scripts/evaluate_scores.py --run-dir outputs/<model>/<fold>/<seed> --fold <fold> --seed <seed> --source-data-root /mnt/drive/audio-deepfake-cache --target-data-root /mnt/salt/datasets/audio-deepfake
```

`metrics.json` contains target EER/AUROC, raw-score transfer, unlabeled
z-score transfer, source-derived APCER thresholds at 1%, 5%, and 10%, and 95%
class-stratified bootstrap intervals. Metric values are fractions in `[0,1]`;
multiply by 100 to display percentages. Raw scores are never clipped or
normalized before ranking metrics. For normalized transfer, each source dev
domain uses its own mean/std and the target uses its own **unlabeled** score
mean/std. This branch is transductive, not strict zero-shot.

The common metrics are ready for all models. WavLM-WA and AASIST now have
training and score-export implementations. AASIST still needs
complete source-only real-data development/reload validation; its short GPU
feasibility check passed on 2026-10-07. Other SOTA detectors need model-specific code.

Both evaluators validate every score ID and label against the canonical split
reader. Reordered rows are accepted and sorted by ID for reproducible bootstrap
sampling. Generic evaluation requires `best.pt`, `training_complete.json`, and a
`score_manifest.json` that binds fold, seed, splits, checkpoint/completion hashes,
and each score CSV's SHA-256. Future score exporters can use the small
`load_score_manifest`, `checkpoint_digest`, and `save_score_manifest` helpers in
`protocol.py`: create the manifest after training completion and before export,
then register each completely written, validated CSV's digest. Existing files
without this provenance fail; they are never silently adopted or overwritten.

Bootstrap intervals resample target observations within each class only. Source
thresholds, source normalization statistics, and full-target normalization
statistics stay fixed. These intervals exclude training and calibration
uncertainty. Final summaries require all four folds: families `wavlm_bs96` and
`aasist` use seed 1234 only; other families retain their declared seed plans.
Single-seed summaries use `sample_std=null`, not zero.
`--non-final --seeds ...` writes an explicitly exploratory summary instead.
Evaluators and summary commands refuse to overwrite existing metric reports.

SpeechFake duplicate checks compare every supplied metadata column after trimming
whitespace, normalizing relative paths, and canonicalizing label spelling. A
conflicting train/dev overlap raises before deduplication; identical overlaps
are removed from train. No canonical count expectation has been relaxed. Changes
to the actual metadata schema or counts still require a source-only data check.

## Local results report

The owner-requested [WavLM F1–F3 supervisor report](docs/wavlm_f1_f3_report/README.md)
is stored in a Git-ignored folder under `docs/`. It includes editable LaTeX,
a PDF reading copy, complete recorded P1 metrics and scientific plots. See its
README for provenance, regeneration commands and the LaTeX compiler limitation.
F4 remains necessary for the complete four-fold report.

A separate [concise WavLM F1–F3 results report](docs/wavlm_f1_f3_short_report/README.md) presents the same
completed folds with simpler tables and an explicit bootstrap explanation.
It preserves the detailed companion report; both numerical reports remain local-only.

## Entry points

Run the documented model and data commands under `scripts/`. The generated
package-level greeting command has been removed; package initializers remain
for imports. Training, evaluation, recovery and synthetic-test files are retained
because they support the current study and future detectors.
