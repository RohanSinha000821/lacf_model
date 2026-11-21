# LHCC frozen detector adaptation

Preparation requested 20 November 2026; the owner explicitly accepts labelled
adaptations with recorded assumptions. This is our implementation from the
[LHCC paper](https://doi.org/10.1016/j.inffus.2026.104358), not released author
code or a reproduction of its published numbers. The local original is
`references/sota/LHCC.pdf`. No official detector checkpoint was confirmed.

The official XLS-R300M revision
`1a640f32ac3e39899438a2931f9924c02f080a54` stays frozen in eval/no-grad mode.
Only AFM, DCM and classifier train, freshly initialized per fold/seed. Use all
24 Transformer outputs, retaining time and feature axes; the convolutional
state is excluded. Existing pooled WavLM/CLAP caches cannot substitute for them.

Paper components retained: layers1–21 versus22–24,32-channel projections,
3:2 Bottle2neck streams, gated fusion/residual/LayerNorm, three SE/Transformer
blocks with MHA parameters shared by blocks1 and3, and a binary classifier.
We do not replace this with a small MLP or only train the output linear layer.

Explicit architectural resolutions where author code/specification is missing:

- Stack layers as 2-D convolution channels; preserve time and1024 feature
  coordinates. Initial1x1 projection and3x3 stride1 padded maxpool; Res2Net
  scale4 with32 channels and BatchNorm/ReLU. Kernel/stride/scale/BN are study
  choices. Low/high stacks contain3/2 residual blocks, respectively.
- Pooled channel gates use64→32→64, sigmoid; a64→64 residual1x1 convolution
  adds to the gated streams. A learned64→1 channel bridge yields[time,1024].
  The paper does not specify this bridge; it is an implementation assumption.
- DCM width1024,8 attention heads, FFN hidden1024, SE reduction16, residual
  post-LayerNorm, no added positional encoding/dropout, then an additional FFN
  matching Eq.3. Only first/third MHA share; FFN/SE/norm weights are separate.
- Fig.1 shows3x3/stride3 averaging,67x341 flatten,22847→1024→2 classifier,
  conflicting with the prose GAP/sigmoid classifier. We follow the figure,
  use ReLU between classifier layers, and record actual parameter counts.
  The count need not equal the paper's rounded40.3M; it cannot be certified
  without author code. No learned layer/temporal pooling replaces native inputs.

Input is mono16kHz, first64600 samples/repeat-short for train and dev/test,
following the literal truncate/repeat description (crop origin is unreported).
No extra waveform normalization. RawBoost is off: the paper does not identify
its algorithm; omission is a declared augmentation adaptation for this frozen
comparison, not a claim that the paper omitted augmentation.

For already16kHz audio, seek/read only the exact native segment rather than
decode the unused tail. Other sample rates retain full resampling before crop
to preserve resampler boundary behavior. Synthetic stereo16k/44.1k tests prove
exact loading parity; no real-data speedup is claimed from those tests.

Adam1e-6/wd1e-4, batch8, constantLR (schedule unreported), max10/patience2.
Weighted binary BCE uses[BF,spoof]=[0.9,0.1], a study choice because numeric
weights are unreported. Public logits are[BF,spoof]; their difference is the
spoof logit, equivalent to negating a bona-fide logit for binary BCE. Scores
increase with spoof probability. Original paper's encoder-training contradiction
is retained in the notes; the owner's frozen-detector request fixes our scope.

Use the existing environment; no new dependencies. Safe CPU inspection:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. \
  python scripts/lhcc/train.py --fold f1 --inspect
```

Future training command, **not launched by preparation**:

```bash
PYTHONPATH=src:. python scripts/lhcc/train.py --fold f1 --seed 1234 \
  --data-root /mnt/drive/audio-deepfake-cache --num-workers 16 \
  --eval-workers 4 --prefetch-factor 2 --cpu-threads 2
```

The trainer first checks local pinned weight files/canonical source membership
and discards a real-source full-batch finite-update/first-tail reload probe.
It resets detector/sampler/crop RNG before the fresh fit. This preflight has
NOT run on real data/GPU during implementation. Sharing GPU memory or optimal
batch/precision/throughput has not been established; inspect live headroom before
authorizing a launch. No allocator cap or automatic fallback is configured.

Each fold reads only its three official source train/dev partitions and uses
the shared equal-domain/natural-class replacement sampler. Select strict macro
source-dev EER, ties earlier. This differs from LACF-MoE's class-balanced sampler
and fixed10s cached views; report those differences. Native input durations
remain separate from matched-duration controls. Target results cannot select
architecture/weights/recipe, and earlier target exposure makes the new comparison
exploratory. A fair comparison need not use identical batches or trainable counts.

Outputs: `outputs/lhcc_frozen_v1/<fold>/<seed>/`; detector-only best/last/snapshot
checkpoints explicitly bind the pinned encoder files, config/source identities
and runtime hashes. Snapshots are not resumable/completed runs. Completion
requires full selected-source bit-exact reload, canonical score exports and exact
CSV-round-tripped source calibration. No overwrite/retry, training resume,
target evaluation, queues, downloads, automatic commit/push are implemented.
Seed options retain the project's default1234/2345/3456; no seeds are queued.
Source exports support the existing1/5/10% operating points; target metrics/CIs
require a later separately implemented/authorized evaluator.
