# SSL-last F3: parallel training and recovery

The owner authorizes fresh F3 seed1234 alongside F2, with resumability.
The original F1/F2 trainers and their running sources remain unchanged.

Implementation: `scripts/ssl_gating/train_last_f3.py` and
`src/audio_deepfake_detection/sota/ssl_gating_resume.py`.
Outputs: `outputs/ssl_gating_last/f3/1234/`.
Launch/logs: `outputs/ssl_gating_last/f3/launch/`.

Keep the same model and scientific recipe as SSL-last F1/F2: XLS-R block23
plus gating backend train (15,241,090 parameters); other encoder blocks stay
frozen but retain training stochasticity. Batch24/BF16-sensitiveFP32,
Adam3e-6/wd1e-4/constant, weightedCE+CKA1,20s random train/full batch1 dev,
maximum10/patience3, strict equal-domain source macro EER, earlier ties,
seed1234, domain-only balancing/natural classes, workers8/4/prefetch2.
Fit ASV2019/CFAD/SpeechFake sources; ASV5 target is not opened or evaluated.

## Measured parallel execution

Two19GiB budgets would not safely fit the35GiB GPU. F3 uses a validated13GiB
PyTorch allocator cap after at least15GiB live free memory, standard CUDA
allocator with `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`, and CPUthreads2.
F2's existing19GiB budget and process remain untouched. These are execution
settings, not new architecture/loss/batch/precision settings.

Actual maximum20s×24/all24-layer source stress passed. Peak allocated
12,516,036,096 bytes; reserved13GiB. Deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` reproduced an actual source optimizer update
with zero loss/parameter/buffer differences and matching CPU/CUDA RNG state.
Fresh full129.75475s source score reload was bit-exact. All probe weights are
discarded before final initialization. CPU synthetic recovery tests also pass;
they are distinct from these actual source GPU results.

Two earlier probes remain preserved under the launch directory:
`memory_probe_expandable/` (driver does not support expandable segments) and
`memory_probe_standard/` (memory fits, strict recovered GPU update differed
without deterministic algorithms). No final model was trained from either.

## What recovery saves

`recovery.pt` is written every15minutes after a complete optimizer update,
after the training phase, and after each source-development epoch. It contains:

- Full model and Adam optimizer state, plus all RNG state (Torch CPU/CUDA,
  NumPy and Python).
- Current epoch/phase, next processed-draw offset and accumulated loss totals.
- Complete sampled utterance order, each draw's crop seed, and generators for
  future epochs.
- Best model/source scores/metrics and early-stopping patience, together with
  configuration identity. Best selection does not restart on recovery.

Files are written to a temporary path, fsynced, then atomically renamed. A
partial temporary write cannot replace the preceding complete checkpoint.
Explicit recovery preserves leftover temporary files for inspection. The
recovery file itself is not a training-completion marker.

Crop randomness is assigned per sampled draw rather than per worker. This
retains the original uniform crop distribution, replacement/domain sampler,
epoch length and batch size, but changes RNG assignment relative to legacy
F1/F2. It allows remaining crops to match after worker restart/prefetch loss.
Source-development loader RNG is isolated from encoder/model RNG. CPU tests
verify continuation after changing worker count, including actual long-audio
cropping; production keeps the approved eight training workers.

If interrupted, updates since the latest completed checkpoint (roughly up to
15minutes) are repeated. An interrupted development pass is rescored from its
beginning, with no extra training updates. Verified exact replay applies to
the tested software/device/settings; it is not a guarantee across hardware,
driver, dependency or kernel changes. Those identities and source guards must
remain compatible. F2's old snapshots have not been converted to this format.

## Operator commands

Watch the running job:

```bash
tmux attach -t ssl-last-f3
```

After an actual interruption, recovery requires a new explicit owner request
and compatible source/software evidence. The original launcher entry was:

```bash
bash outputs/ssl_gating_last/f3/launch/resume.sh
```

For the later16November recovery, `resume_last_f3.py` and the separately
recorded `launch/resume_20261116.sh` handle only an absent-versus-null
`attn_implementation` serialization difference. They preserve original guarded
sources and checkpoint/config bytes. Other attention/config changes still fail
validation. Use the owner-approved launcher recorded in the local handoff; this
document does not authorize replaying either historical launch. Diagnose an
unexplained native crash before requesting another recovery.

Recovery is explicit; no automatic restart is scheduled. The launcher preserves
and appends the existing console log, writes a separate recovery exit record,
and uses a lock to refuse a second trainer while the first is active. The
trainer rejects completed or mismatched runs. Source guards, canonical source
readiness and passed GPU evidence remain mandatory.

Checkpoint/source selection and final fresh full-source reload/export/calibration
remain unchanged. No target evaluation, new folds/seeds, downloads or automatic
commit/push follow from recovery support.
