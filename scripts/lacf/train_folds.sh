#!/usr/bin/env bash
# Fresh source-only F1 -> F2 -> F3 -> F4, seed 1234. No resume or target scoring.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_dir/activate.sh"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
arguments=("$@")
segment_seconds=10
queue_root=""
for ((i=0; i<${#arguments[@]}; i++)); do
    argument="${arguments[$i]}"
    case "$argument" in
        --fold|--fold=*) printf 'The queue fixes fold order; do not pass --fold.\n' >&2; exit 2 ;;
        --segment-seconds|--output-root)
            value="${arguments[$((i+1))]:-}"
            [[ -n "$value" && "$value" != --* ]] || { printf 'Missing value for %s\n' "$argument" >&2; exit 2; }
            if [[ "$argument" == --segment-seconds ]]; then segment_seconds="$value"; else queue_root="$value"; fi
            ;;
        --segment-seconds=*) segment_seconds="${argument#*=}" ;;
        --output-root=*) queue_root="${argument#*=}" ;;
    esac
done
case "$segment_seconds" in
    4) queue_root="${queue_root:-$project_dir/outputs/lacf4s}" ;;
    10) queue_root="${queue_root:-$project_dir/outputs/lacf}" ;;
    *) printf 'Expected --segment-seconds 4 or 10\n' >&2; exit 2 ;;
esac
for fold in f1 f2 f3 f4; do
    if [[ -e "$queue_root/$fold/1234" ]]; then
        printf 'Existing LACF run preserved; no resume: %s\n' "$queue_root/$fold/1234" >&2
        exit 1
    fi
done
mkdir -p "$queue_root"
exec 9>"$queue_root/.queue.lock"
flock -n 9 || { printf 'Another LACF queue is active.\n' >&2; exit 1; }
for fold in f1 f2 f3 f4; do
    printf 'Starting %s, seed 1234\n' "$fold"
    python -u "$project_dir/scripts/lacf/train.py" --fold "$fold" "$@"
    python - "$queue_root/$fold/1234" "$fold" <<'PY'
import sys
from pathlib import Path
from audio_deepfake_detection.lacf.training import verify_fold_completion
verify_fold_completion(Path(sys.argv[1]), sys.argv[2])
PY
    printf 'Verified completion: %s\n' "$fold"
done
printf 'All four source-only LACF folds completed.\n'
