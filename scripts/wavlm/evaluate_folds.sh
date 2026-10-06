#!/usr/bin/env bash
# Owner-approved independent completed folds; source-only calibration precedes targets.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/activate.sh"
export PYTHONPATH="$project_root/src:$project_root"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
family_root="$project_root/outputs/wavlm_bs96"
folds=("$@")
if (( ${#folds[@]} == 0 )); then folds=(f1 f2 f3); fi
declare -A requested=()
for fold in "${folds[@]}"; do
    case "$fold" in f1|f2|f3|f4) ;; *) echo "Invalid fold: $fold" >&2; exit 2 ;; esac
    if [[ -v requested[$fold] ]]; then echo "Duplicate fold: $fold" >&2; exit 2; fi
    requested[$fold]=1
    if [[ ! -f "$family_root/$fold/1234/training_complete.json" ]]; then
        echo "Training is incomplete: $fold" >&2; exit 1
    fi
    if [[ -e "$family_root/$fold/1234/metrics.json" ]]; then
        echo "Metrics already exist for $fold; refusing to overwrite" >&2; exit 1
    fi
done
exec 9>"$family_root/.evaluation.lock"
flock -n 9 || { echo "Another WavLM evaluation queue is active" >&2; exit 1; }
for fold in "${folds[@]}"; do
    echo "[$(date -Is)] Starting native P1 evaluation: $fold seed 1234"
    recovery_args=()
    if [[ "${WAVLM_ADOPT_LEGACY_PARTIAL_FOLD:-}" == "$fold" ]]; then
        recovery_args+=(--adopt-legacy-partial)
    fi
    python -u "$project_root/scripts/wavlm/evaluate.py" \
        --fold "$fold" --seed 1234 --run-name wavlm_bs96 \
        --source-data-root /mnt/drive/audio-deepfake-cache \
        --target-data-root /mnt/salt/datasets/audio-deepfake \
        --num-workers 8 --prefetch-factor 2 --bootstrap-resamples 1000 \
        --confirm-run-frozen "${recovery_args[@]}" \
        2>&1 | tee -a "$family_root/$fold/1234/evaluation.log"
    [[ -f "$family_root/$fold/1234/metrics.json" ]] || { echo "Missing final report: $fold" >&2; exit 1; }
    echo "[$(date -Is)] Completed: $fold"
done
available=()
for fold in f1 f2 f3 f4; do
    if [[ -f "$family_root/$fold/1234/metrics.json" ]]; then available+=("$fold"); fi
done
if (( ${#available[@]} == 4 )); then
    python "$project_root/scripts/summarize_results.py" --model wavlm_bs96 --root "$project_root/outputs"
else
    python "$project_root/scripts/summarize_results.py" --model wavlm_bs96 --root "$project_root/outputs" --non-final --folds "${available[@]}"
fi
echo "[$(date -Is)] Evaluation queue finished; existing fold results preserved"
