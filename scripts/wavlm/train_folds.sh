#!/usr/bin/env bash
# Source-only F2 -> F3 -> F4. Any failure stops the queue; no target scoring.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_dir/activate.sh"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"

seed="${1:-1234}"
if [[ $# -gt 1 || ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'Usage: bash scripts/wavlm/train_folds.sh [non-negative seed]\n' >&2
    exit 2
fi

run_name="wavlm_bs96"
queue_root="$project_dir/outputs/$run_name"
mkdir -p "$queue_root"
# All seeds in this output family share a lock: one GPU training queue at a time.
exec 9>"$queue_root/.queue.lock"
if ! flock -n 9; then
    printf 'Another batch-96 queue is active; refusing a concurrent launch.\n' >&2
    exit 1
fi

log_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$queue_root/queue_${seed}.log"
}

log_status "Queue started: F2 -> F3 -> F4; seed=$seed; batch=96; train workers=16; dev workers=4; prefetch=2"
for fold in f2 f3 f4; do
    run_dir="$queue_root/$fold/$seed"
    mkdir -p "$run_dir"
    log_status "Starting $fold"
    if python -u "$project_dir/scripts/wavlm/train.py" \
        --fold "$fold" --seed "$seed" --batch-size 96 --run-name "$run_name" \
        --data-root /mnt/drive/audio-deepfake-cache \
        --num-workers 16 --eval-workers 4 --prefetch-factor 2 \
        2>&1 | tee -a "$run_dir/training.log"; then
        if [[ ! -f "$run_dir/training_complete.json" ]]; then
            log_status "FAILED $fold: no training completion record; queue stopped"
            exit 1
        fi
        log_status "Completed $fold"
    else
        log_status "FAILED $fold: training or logging failed; queue stopped"
        exit 1
    fi
done
log_status "All three folds completed. No target evaluation was launched."
