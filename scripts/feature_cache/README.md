# Shared four-second frozen features

Cache: `outputs/frozen_features/4s/`.

The same physical interval feeds frozen `microsoft/wavlm-base-plus` at16kHz and
`laion/clap-htsat-unfused` at48kHz. Each source training recording has **one**
fixed random interval up to4seconds, keyed by canonical ID and crop seed1234.
Development has one centered interval. Shorter recordings retain their full
waveform. There are no extra crop banks or three-view aliases.

Each row stores FP32 arrays `wavlm_layers.npy [N,13,768]` (projected frontend
and12Transformer outputs, validity-mask time means) and `clap_projected.npy
[N,512]` (raw projected audio embedding). Encoders execute with BF16 and
FP32 normalization/pooling. WavLM batches contain equal native lengths and no
waveform padding. CLAP still repeat-pads its up-to4s physical crop internally
to its10s input. Cached fixed features cannot fine-tune either encoder or
introduce a new temporal pooling method.

Eight canonical source train/dev partitions are extracted once and shared
across all folds:945745training +297657development recordings. Payload is
52.203GB, approximately55GBwith metadata. Existing10s caches are preserved.
Targets and external datasets are never opened by this producer.

Entrypoint: `python -m scripts.feature_cache.cache_4s`. The launch folder must
first contain the reviewed `provenance.json` and guarded launcher. It waits
without initializing CUDA until12GiBfree, then uses a10GiBallocator limit.
It checks local pinned weights, canonical source membership, direct/native
feature parity, source-only encoder-batch/worker throughput, measured BF16
batch drift, repeated output equality and disk serialization before extraction.
Shards have payload hashes and atomic completion directories. No automatic
retry, resume, overwrite, final head fitting or target evaluation occurs.

Consumers use `audio_deepfake_detection.features.frozen_4s.load_fold` and
`SingleViewBatches`, which verify the4s/single-view schema, membership, labels,
crop offsets, row mapping, dtype and payload hashes. Each fold loads only its
three source training/development domains.

Prepared main consumers:

- `FrozenWavLMBasePlusWA`: train13softmax layer weights plus a768→2linear
  classifier,1551parameters. The pooled-cache computation is an adapted
  frozen Base+ baseline, distinct from original WavLM Base/full fine-tuning.
- Existing main three-expert `LACFMoE`:967338parameters, original3+5concepts,
  35-Drelations and detection BCE only.

Both use exactly the same feature rows. Model-training seeds affect head
initialization, sampling and trainable dropout; they do not create fresh audio
crops. Future model runs and target evaluations need separate launch requests.


Source-only training entrypoint (prepared; not launched):

```bash
python -m scripts.feature_cache.train_heads_4s --model wavlm_wa --prepare
python -m scripts.feature_cache.train_heads_4s --model lacf_moe --prepare
# After the complete cache and a separate training request:
python -m scripts.feature_cache.train_heads_4s --model wavlm_wa --fold f1 --seed 1234
python -m scripts.feature_cache.train_heads_4s --model lacf_moe --fold f1 --seed 1234
```

Run these with the project environment. Preparation reads no feature bank and
initializes no GPU. Final fitting requires the completed four-second bank,
checks all six canonical source partitions, discards a source-only update/reload
check, then starts fresh. Selected scores are reloaded bit-exactly, exported and
source-calibrated before completion. Existing run folders are never overwritten;
training resume and target evaluation are not implemented here.

Outputs are `outputs/wavlm_wa_frozen_base_plus/4s/<fold>/<seed>/` and
`outputs/lacf_moe/4s/<fold>/<seed>/`. Preparation records are already present.
WA preserves domain-only sampling, weighted CE [9,1], batch96, Adam0.005,
ExponentialLR0.95, maximum100/patience5. MoE preserves domain/class sampling,
detection BCE, batch132, AdamW0.0003/weight decay0.0001, constant LR,
maximum30/patience5. These are each method's original head settings; they are
not identical optimization recipes. Both use the same frozen features and
actual BF16 final scores with sensitive calculations in FP32.
