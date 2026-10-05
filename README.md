# Audio deepfake detection: LODO runs

The project owner's main LACF plan and all supplied supplementary plans, dataset
papers and SOTA papers are organized locally under `references/`. Start with
[references/INDEX.md](references/INDEX.md) for the document map and authority
notes. `AGENTS.md` directs future project work to these references. Full PDF/DOCX
copies are excluded from Git; back up the reference folder separately.

The fixed four-fold membership, paper-informed recipes, and four fairness
safeguards are in `EXPERIMENT_PROTOCOL.md` (v3, updated 2026-10-02). Before
inspecting any held-out target result from any method, complete its
comparison-wide design/recipe freeze. Source-only pilots and training may
continue while remaining methods are implemented; target evaluation waits for
that gate. SpeechFake source train/dev caching runs separately from training.
Do not start F2–F4 until `outputs/speechfake-cache.log` ends with
`verification: PASS` and `Cache preparation complete`.

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

Run one fold and seed, then evaluate its **completed best checkpoint** once the
comparison-wide freeze gate is complete. The
training command reads source train/dev only; the evaluation command scores
source dev first, calibrates both transfer branches, then opens the held-out
target from the original read-only dataset root.

```bash
mkdir -p outputs/wavlm_bs96/f1/1234
python -u scripts/wavlm/train.py --fold f1 --seed 1234 --batch-size 96 --run-name wavlm_bs96 --data-root /mnt/drive/audio-deepfake-cache --num-workers 16 --eval-workers 4 --prefetch-factor 2 2>&1 | tee -a outputs/wavlm_bs96/f1/1234/training.log
# Run only after the comparison-wide freeze is complete:
python -u scripts/wavlm/evaluate.py --fold f1 --seed 1234 --run-name wavlm_bs96 --source-data-root /mnt/drive/audio-deepfake-cache --target-data-root /mnt/salt/datasets/audio-deepfake
```

Use seed **1234 only** for WavLM across `f1`, `f2`, `f3`, `f4`, per the owner's
2026-10-05 compute-budget decision. Do not schedule WavLM seeds 2345 or 3456.
The trainer refuses to overwrite existing checkpoints. An
interrupted run lacks `training_complete.json` and cannot be evaluated as a
finished run. Earlier interrupted WavLM pilots were removed; the completed
batch-64 F1 is preserved separately. New batch-96 runs live in
`outputs/wavlm_bs96/<fold>/<seed>/`.

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
three-fold report is partial, not a complete four-fold result. Other models retain
their existing three-seed plan unless explicitly changed separately.

## AASIST: ready for a source-only GPU pilot after WavLM

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

Only synthetic CPU tests have run so far. A full batch-24 forward/backward/Adam
step passed with finite loss/gradients and no CUDA initialization. Real-data
source-only throughput, CUDA-memory and complete-dev/reload checks remain to be
performed after the WavLM queue releases the GPU. Start with 8 training workers,
4 dev workers and prefetch 2; these are initial loader settings, not a benchmarked
throughput optimum. Do not start a second GPU trainer during WavLM.

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
after WavLM; no AASIST job or automatic handoff has been started by implementation.
Repeat the fixed recipe later for seeds 2345 and 3456. Existing checkpoints cannot
be overwritten or resumed by this trainer.

Only after the comparison-wide freeze is complete and the selected run has
completed, evaluate its best checkpoint:

```bash
python -u scripts/aasist/evaluate.py --fold f1 --seed 1234 --confirm-protocol-frozen
```

The flag is an explicit operator acknowledgement of the research gate, not an
automatic audit of every model's freeze record. Source calibration/checkpoint
verification still precedes target access. Outputs and metrics use the same
contract as WavLM under `outputs/aasist/<fold>/<seed>/`; after all twelve reports:

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
training and score-export implementations. AASIST still needs its source-only
GPU/real-data pilot. The other SOTA detectors need their model-specific code.

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
uncertainty. Final summaries require all four folds: WavLM family `wavlm_bs96`
uses seed 1234 only; other families retain seeds 1234/2345/3456.
`--non-final --seeds ...` writes an explicitly exploratory summary instead.
Evaluators and summary commands refuse to overwrite existing metric reports.

SpeechFake duplicate checks compare every supplied metadata column after trimming
whitespace, normalizing relative paths, and canonicalizing label spelling. A
conflicting train/dev overlap raises before deduplication; identical overlaps
are removed from train. No canonical count expectation has been relaxed. Changes
to the actual metadata schema or counts still require a source-only data check.
