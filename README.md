# Audio deepfake detection: LODO runs

The fixed four-fold membership and paper-informed recipes are in
`EXPERIMENT_PROTOCOL.md`. SpeechFake source train/dev caching runs separately
from training. Do not start F2–F4 until `outputs/speechfake-cache.log` ends with
`verification: PASS` and `Cache preparation complete`.

## WavLM-WA, one fold at a time

From this project directory, activate the existing Python environment and make
the source package importable:

```bash
source activate.sh
export PYTHONPATH=src
set -o pipefail
```

Run one fold and seed, then evaluate its **completed best checkpoint**. The
training command reads source train/dev only; the evaluation command scores
source dev first, calibrates both transfer branches, then opens the held-out
target from the original read-only dataset root.

```bash
mkdir -p outputs/wavlm/f1/1234
python -u scripts/wavlm/train.py --fold f1 --seed 1234 --data-root /mnt/drive/audio-deepfake-cache 2>&1 | tee -a outputs/wavlm/f1/1234/training.log
python -u scripts/wavlm/evaluate.py --fold f1 --seed 1234 --source-data-root /mnt/drive/audio-deepfake-cache --target-data-root /mnt/salt/datasets/audio-deepfake
```

Repeat the same two commands for `f2`, `f3`, `f4`, one at a time, then for seeds
`2345` and `3456`. The trainer refuses to overwrite existing checkpoints. An
interrupted run lacks `training_complete.json` and cannot be evaluated as a
finished run. The old WavLM F1 pilot has been removed; new runs live in
`outputs/wavlm/<fold>/<seed>/`.

Once all twelve run reports exist:

```bash
python scripts/summarize_results.py --model wavlm
```

## Shared metrics for every SOTA model

Each model writes `source_dev_<dataset>.csv` for the three active source domains
and `target_scores.csv` in its run directory. CSV columns are
`utterance_id,dataset,label,raw_score`, with label 0=bona fide, 1=spoof, and a
larger raw score meaning more spoof-like. The same evaluator then works for any
model:

```bash
python scripts/evaluate_scores.py --run-dir outputs/<model>/<fold>/<seed> --fold <fold>
```

`metrics.json` contains target EER/AUROC, raw-score transfer, unlabeled
z-score transfer, source-derived APCER thresholds at 1%, 5%, and 10%, and 95%
class-stratified bootstrap intervals. Metric values are fractions in `[0,1]`;
multiply by 100 to display percentages. Raw scores are never clipped or
normalized before ranking metrics. For normalized transfer, each source dev
domain uses its own mean/std and the target uses its own **unlabeled** score
mean/std. This branch is transductive, not strict zero-shot.

The common metrics are ready for all models. Only WavLM-WA currently has a
training and score-export implementation in this repository; the other SOTA
detectors still need their model-specific code before they can be run.
