# Cached LACF Layer-MoE: two explicit 4+4 text banks

Owner scope: two fresh full-F1 fits, seed1234 only. Train ASV2019, ASV5 and
CFAD source train; strictly lowest equal-domain macro source-dev EER selects,
ties retain earlier. This is a full-source comparison, not the diagnostic
inner-held-source protocol. Prior target exposure remains disclosed; no target
result selects a prompt, setting or checkpoint. No target evaluation is queued.

The two banks share seven exact phrases/prototypes. Their final spoof phrase
is either `speech synthesized by a neural vocoder` (`vocoder`) or
`speech synthesized artificially by a neural network` (`neural_synthesis`).
See `src/audio_deepfake_detection/experiments/concept_bank/model.py` for the
complete eight phrases. Bona-fide indices0–3; spoof indices4–7; uniform spoof
mass0.5. Retain the three original templates, individual normalization then
prototype averaging/normalization, temperature0.07 and35-D relation vector.

Unchanged cached C4 architecture: three layer-group residual experts,
input-dependent router,768→512→512 adapter and35→64→16→1 relation classifier;
967,338 trainables. Both frozen audio encoders remain unloaded. Reuse the
validated centered+two-fixed-random10s source cache, including native unpadded
WavLM extraction and raw CLAP projections. Different text prototypes do not
invalidate these raw audio features. The new subclass changes metadata only;
same-prototype scores equal the original C4 math in CPU tests.

Recipe: detection BCE only, auxiliary losses disabled; AdamW3e-4/wd1e-4,
constant LR,batch132/accum1,BF16 with sensitive FP32,max30/patience5,
equal-domain/class replacement sampling,0feature workers,2CPUthreads,
1GiB CUDA allocator ceiling. Cached features load/verify once for both fits.
Pinned original CLAP text weights are used on CPU only to encode the two banks;
the first seven prototypes are shared bit-for-bit. No downloads.

The queue checks current guarded TF2 sources, frozen-cache completion/config/
execution hashes, canonical source IDs/labels and every cache payload. Each
bank gets32 source preflight updates and full-source scoring/fresh reload.
Those weights are discarded. Final fits reset seed1234 and start fresh.
Final completion requires full selected-checkpoint source reload, canonical
score export, calibration reproduction and bound checkpoint/config/score hashes.
The second bank advances only after the first completion verifies. No resume,
automatic retry, existing-directory overwrite, target access or publication.

New model/script/tests live only under their respective `experiments/concept_bank/`
directories. Existing production and historical guarded experiment files remain
unchanged. The entry point reuses shared loading, sampling, epoch training,
scoring, atomic checkpoint, preflight, metrics and routing utilities; its local
fit orchestration records the new bank identities without rewriting historical
guarded trainers.

Authorized command (project environment/PYTHONPATH; local HF cache only):

```bash
python -u -m scripts.experiments.concept_bank.train_moe
```

Outputs under `outputs/experiments/concept_bank/moe_f1/`:

- `banks/`: exact definitions, pinned text-encoding evidence and prototypes.
- `vocoder/f1/1234/`, `neural_synthesis/f1/1234/`: independent fresh runs.
- `launch/console.log`: cache checks, both preflights and live queue progress.
- Each run: `training.log`, `history.jsonl`, `best.pt`, `last.pt`, source scores,
  calibration, routing summary and verified `training_complete.json`.
- `complete.json`: written only after both source completions verify.

Source calibration is saved for later authorized raw/transductive evaluation.
No unseen-domain normalized ACER claim follows from these source-trained fits.

## F1 target evaluation authorized on 11 November 2026

The owner subsequently requested evaluation of both completed banks on
SpeechFake test. This overrides the training-stage target hold above for these
two fixed F1 checkpoints only. No fitting, prompt selection or recipe changes
use these targets; the results remain exploratory after prior target exposure.

Run `python -u -m scripts.experiments.concept_bank.evaluate_moe` in the project
environment. The small isolated wrapper reuses the existing MoE full-source
reload and target-cache verification, canonical score writers and shared
metrics. It verifies bank/prototype/checkpoint/source hashes and source
calibration before opening target metadata, then scores the existing centered
10-second WavLM/CLAP audio-feature cache. No encoders or weights are downloaded
or executed. Both audio branches remain inputs.

The queue evaluates `vocoder`, then `neural_synthesis`, with head batch132,
the trained BF16/FP32 policy, two CPU threads and a1GiB CUDA allocator ceiling.
It reports EER/AUROC and raw plus separately labelled transductive z-score
APCER/BPCER/ACER at all1%,5%,10% source APCER thresholds. Shared conditional
95% bootstrap intervals use1000 class-stratified resamples, seed2026.

Outputs: `outputs/experiments/concept_bank/moe_f1/evaluations/`, with
`<bank>/f1/1234/metrics.json`, score/provenance files and
`evaluation_complete.json` per bank; queue `complete.json` only after both.
Launch evidence and console log live in sibling `evaluation_launch/`.
Existing training/evaluation outputs are never overwritten. Stop on failure;
no automatic retries, other folds, new training or publication.
