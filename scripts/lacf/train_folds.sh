#!/usr/bin/env bash
# Fresh source-only F1 -> F2 -> F3 -> F4, seed 1234. No resume or target scoring.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_dir/activate.sh"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
for argument in "$@"; do
    case "$argument" in
        --fold|--fold=*) printf 'The queue fixes fold order; do not pass --fold.\n' >&2; exit 2 ;;
    esac
done
queue_root="$project_dir/outputs/lacf"
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
