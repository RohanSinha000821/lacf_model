#!/usr/bin/env bash
# Prepared command only: run manually after owner authorizes F4 training.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_dir/activate.sh"
export PYTHONPATH="$project_dir/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run="$project_dir/outputs/wavlm_bs96/f4/1234"
proof="$project_dir/outputs/wavlm_bs96/f4/preparation/source_validation.json"
exec 9>"$project_dir/outputs/wavlm_bs96/.queue.lock"
flock -n 9
test ! -e "$run"
test -f "$proof"
mkdir -p "$run"
set +e
python -u -m scripts.wavlm.train --fold f4 --seed 1234 --batch-size 96 --run-name wavlm_bs96 \
    --num-workers 16 --eval-workers 4 --prefetch-factor 2 \
    --data-root /mnt/drive/audio-deepfake-cache --last-block-only --source-validation-record "$proof" \
    2>&1 | tee "$run/training.log"
codes=("${PIPESTATUS[@]}")
python - "$run" "${codes[0]}" "${codes[1]}" <<'PYEXIT'
import json,sys
from pathlib import Path
from audio_deepfake_detection.protocol import checkpoint_digest
run=Path(sys.argv[1]);code=int(sys.argv[2]) or int(sys.argv[3])
result={'trainer_exit':int(sys.argv[2]),'logger_exit':int(sys.argv[3]),'verified_complete':False,'target_access':False}
if not code:
    try:
        done=json.loads((run/'training_complete.json').read_text())
        assert done['selected_checkpoint_reloaded_and_source_verified'] and done['full_source_scores_bit_exact']
        assert done['best_checkpoint_sha256']==checkpoint_digest(run/'best.pt')
        for name,digest in done['source_scores_sha256'].items():assert checkpoint_digest(run/name)==digest
        assert (run/'score_manifest.json').exists()
        result['verified_complete']=True
    except Exception as error:
        result['error']=str(error);code=1
with (run/'exit.json').open('x') as f:json.dump(result,f,indent=2)
sys.exit(code)
PYEXIT
