# TF2 with frozen cached CLAP

Owner authorization, 9 November: source-only validation and a fresh full-F1
launch are authorized, replacing the owner-stopped partial SSL F2 run. This is an
exploratory full-F1, seed-1234 comparison against the existing adapted TF2 run,
not inner-source model selection or a new target evaluation. The current
reference underwent weights-only recovery; disclose that trajectory difference.

## Model and data

Train rank8/alpha16 Q/V LoRA in all12 WavLM Base+ blocks and the original TF2
adapter/tokens/two fusion blocks/classifier: 294,912 + 4,907,745 = 5,202,657
trainable parameters. Final training loads no CLAP weights, processor or
48kHz waveform; bounded source validation loads the frozen encoder only to
verify the cache. Its audio branch remains present as cached512-D raw CLAP
projections; original TF2 normalization is unchanged. WavLM features must be
recomputed because LoRA updates them, including source-development features.

`src/audio_deepfake_detection/experiments/tf2_frozen_clap/model_data.py` reuses
the guarded adaptation model, fixed-crop definition and domain/class sampler.
It checks canonical source membership, shard metadata/completion hashes and
all CLAP payload hashes, shapes and finite values once at loading. It does not
read/hash the unused WavLM arrays. Each sampled recording/view is matched to
its exact cached CLAP row. Source audio length/rate is checked on access; this
is not a new full audio-content checksum audit. No target partition is allowed.

F1 sources: ASV2019/ASV5/CFAD train and their three development sets. Training
chooses among the preserved centered and two fixed random physical10s views;
short-recording aliases stay intact. Development uses centered views. WavLM
encoding groups valid native lengths before convolution/normalization.

Keep detection BCE only, BF16/FP32 boundaries, AdamW LoRALR1e-4/fusionLR3e-4,
wd1e-4, constantLR, effectivebatch132, max30/patience5, strict equal-domain
macro source-dev EER, ties earlier. Fresh pinned WavLM and seeded TF2; no
pilot/adapted-CLAP weights, resume, automatic retry, overwrite or target tuning.

## Naming and files

New implementation/test files are exclusively under `experiments/`:

- `src/audio_deepfake_detection/experiments/tf2_frozen_clap/model_data.py`
- `scripts/experiments/tf2_frozen_clap/train.py`
- `tests/experiments/tf2_frozen_clap/test_cached_clap.py`

Planned run: `outputs/experiments/tf2_frozen_clap/f1/seed_1234/`.
Use a meaningful manual suffix for a separately authorized repeat, never reuse
an existing run. Fold/seed numbers are scientific identifiers; timestamps do
not replace the method name. Existing historical folders are not renamed.

Outputs after launch: `config.json`, `training.log`, `history.jsonl`,
`best_adaptation.pt`, `last_adaptation.pt`, source CSVs,
`source_calibration.json`, source-only `metrics.json`, and `complete.json`.
Completion requires selected-checkpoint reload from the pinned base,
bit-exact full source scores, verified CSV export and calibration agreement.
Cached CLAP tensors are external frozen inputs; checkpoint provenance binds
their original source cache/revisions. Periodic/resumable training is not added.

## Preparation and remaining launch gates

Activate the shared environment at the repository root. CPU-only preparation
prints the inherited candidate recipe and its hash, without loading models,
reading feature arrays, decoding audio, creating a run or accessing CUDA:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.experiments.tf2_frozen_clap.train --mode prepare
```

Batch22/workers8/memory cap10GiB retain the adapted reference execution
settings. The owner stopped partial SSL F2 before the comparison launch.
Matching source GPU validation must measure real longest10s forward/backward/optimizer memory, finite enabled
gradients, actual-source train/dev throughput, cached-vs-direct frozen CLAP
agreement (report numerical rounding), fresh sparse checkpoint reload and
headroom for both existing jobs. Discard all timing weights. No runtime estimate
is established by CPU tests or by this preparation command.

The new trainer deliberately does not manufacture a passed validation record.
The future validation operator must save actual evidence with `passed: true`,
`target_access: false`, the exact preparation `recipe_sha256`, and all checks:
`longest_ten_second_forward_backward`, `finite_gradients`,
`direct_cached_clap_parity`, `fresh_checkpoint_reload`,
`concurrent_memory_headroom`. Its `measurements` must contain finite positive
`train_examples_per_second`, `source_dev_examples_per_second` and
`peak_allocated_bytes` below the configured cap. Preserve detailed logs and
failed probes; do not mark parity passed without reviewing actual results.
New settings/code/cache identity require a newly matching validation record.
Actual measured evidence is stored in the launch folder, separately from
CPU preparation. Only a passed matching record opens the training gate.

After explicit launch authorization and matching passed checks, the command is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m scripts.experiments.tf2_frozen_clap.train --mode train \
  --validation-record /absolute/path/to/actual_source_validation.json
```

The path above names future measured evidence; no such passed record is created
by preparation. The CLI refuses a missing/stale/failed record before model/GPU
work. Use the same measured batch/worker/memory flags as that record. Later
folds/repeats require separate authorization. Guarded adapted TF2 files stay untouched.

Synthetic checks:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/experiments/tf2_frozen_clap -p no:cacheprovider
```

## Source validation and live records

The focused validator is `scripts/experiments/tf2_frozen_clap/validate.py`. It
checks all six F1 source CLAP payloads/canonical identities, bounded direct
CLAP parity for partition boundaries, short/long recordings and all fixed views,
actual10s LoRA updates/finite gradients, sampled real-source throughput, and
fresh pinned-base sparse-state reload. It writes matching launch evidence only
after passing. Accelerated BF16 cache rounding is reported with the declared
cosine>=0.999/relativeL2<=0.02 bound; this is not bit-exact direct/cache parity.
No target data are used. Full-source verification remains part of final fitting.

Launch/evidence/logs: `outputs/experiments/tf2_frozen_clap/launch/`.
The guarded launch wrapper rejects existing runs or unmatched evidence.
Check `source_validation.json` and the local handoff for actual validation and
launch status; the authorization is not a completion claim. Probe weights are
discarded before the separate fresh training process.
