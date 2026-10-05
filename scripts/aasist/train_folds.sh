#!/usr/bin/env bash
# Source-only AASIST F1 -> F2 -> F3 -> F4. Launch AFTER the WavLM queue finishes.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_dir/activate.sh"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
seed="${1:-1234}"
if [[ $# -gt 1 || ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'Usage: bash scripts/aasist/train_folds.sh [non-negative seed]\n' >&2
    exit 2
fi
queue_root="$project_dir/outputs/aasist"
mkdir -p "$queue_root"
exec 9>"$queue_root/.queue.lock"
if ! flock -n 9; then
    printf 'Another AASIST queue is active; refusing concurrent launch.\n' >&2
    exit 1
fi
# Do not interrupt or compete with the currently running WavLM queue.
# Open its existing lock without truncating it; retain this lock for our queue.
if [[ -f "$project_dir/outputs/wavlm_bs96/.queue.lock" ]]; then
    exec 8<"$project_dir/outputs/wavlm_bs96/.queue.lock"
    if ! flock -n 8; then
        printf 'WavLM queue is still active; start AASIST after it finishes.\n' >&2
        exit 1
    fi
fi
log_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$queue_root/queue_${seed}.log"
}
log_status "Queue started: F1 -> F2 -> F3 -> F4; seed=$seed; batch=24; float32"
for fold in f1 f2 f3 f4; do
    run_dir="$queue_root/$fold/$seed"
    mkdir -p "$run_dir"
    log_status "Starting $fold"
    if python -u "$project_dir/scripts/aasist/train.py" \
        --fold "$fold" --seed "$seed" --run-name aasist \
        --data-root /mnt/drive/audio-deepfake-cache \
        --num-workers 8 --eval-workers 4 --prefetch-factor 2 \
        2>&1 | tee -a "$run_dir/training.log"; then
        if [[ ! -f "$run_dir/training_complete.json" ]]; then
            log_status "FAILED $fold: no completion record; queue stopped"
            exit 1
        fi
        log_status "Completed $fold"
    else
        log_status "FAILED $fold: training or logging failed; queue stopped"
        exit 1
    fi
done
log_status "All four folds completed; no target evaluation launched."
