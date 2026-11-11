# LACF Layer-MoE: full F1 comparison

Owner authorization,10 November2026: train a fresh full-F1 model at seed1234,
then evaluate its source-selected checkpoint on SpeechFake test with the same
metric contract as frozen TF2. Preserve both active TF2 jobs. All new code,
tests,logs and outputs stay under experiments. No further folds,seeds,retries,
resume,downloads or publication are queued.

## Model and recipe

**LACF Layer-MoE** is the unchanged diagnostic implementation previously called
C4; that identifier remains only in internal configuration for provenance.
Use all13 frozen WavLM Base+ pooled layer representations. Average earlyR0–R3,
middleR4–R7 and lateR8–R12 within each group, pass each through its existing
768→64→768 residual expert, and combine the three outputs using recording-
dependent softmax weights. All three experts execute; this is a dense mixture.
Retain the original768→512→512 adapter,L2 normalization,frozen CLAP audio,
eight concepts/three templates,T0.07,full35-D relations and35→64→16→1 head.
Train **967,338 parameters** with detection BCE only. Semantic/BF losses are
disabled, while JS/entropy remain features. No architecture or prompt change.

Both encoders stay frozen and are not loaded/executed. Reuse the approximately
27GiB source feature bank: common up-to10s physical intervals,centered plus two
fixed random training views,centered development,short-recording aliases.
Train on ASV2019,ASV5,CFAD:246137 logical training recordings per replacement
sampled epoch; score all180194 source-dev recordings. Equal source-domain/
class mass. RAM gathers use0feature workers,CPUthreads2. Keep batch132/accum1,
AdamW LR0.0003/wd0.0001/constant,BF16 with sensitiveFP32,max30/patience5,
strict equal-domain macro source-dev EER,ties earlier. No training resume.

Before fitting,32 source training batches and complete source-dev scoring/
fresh model reload measure finite updates,memory,throughput and exact parity.
Discard all preflight updates,then reinitialize seed1234. The GPU allocator
ceiling is1GiB,leaving room for both TF2 processes. Final training completion
requires a fresh selected-checkpoint full-source reload,canonical CSV export
and calibration reproduction; router statistics are descriptive only.

## Training followed by fixed-checkpoint evaluation

Training and evaluation code are separate so source training never reads target
metadata or features. The evaluator verifies source completion,checkpoint/config/
score hashes,canonical source labels and frozen calibration,then freshly scores
all three source-dev caches bit-for-bit. Only after those gates may it read the
existing SpeechFake frozen-feature cache. Verify its source-cache/revisions,
canonical membership,crops,chunk hashes,finite shapes and every exported score.
Neither encoder is rerun for target evaluation. Stop if cache identity differs.

Use the unchanged common evaluator for EER,AUROC,APCER/BPCER/ACER at source
APCER1/5/10%,strict raw transfer,separately labelled transductive zscore,and
1000class-stratified bootstrap resamples with seed2026. The confidence intervals
condition on the fixed checkpoint/calibration; one training seed has no
across-seed SD. Target results do not select any setting or later run.

## Files and outputs

- Trainer: `scripts/experiments/layer_fusion/train_full_f1.py`.
- Evaluator: `scripts/experiments/layer_fusion/evaluate_lacf_layer_moe.py`.
- Reused model: `src/audio_deepfake_detection/experiments/layer_fusion/`.
- Focused tests: `tests/experiments/layer_fusion/test_full_f1.py`.
- Training: `outputs/experiments/layer_fusion/lacf_layer_moe/f1/1234/`.
- Source cache/GPU preflight and live log: family `launch/console.log`.
- Evaluation: `outputs/experiments/layer_fusion/evaluations/lacf_layer_moe/f1/1234/`.
- Queue/auth/CPU evidence: `outputs/experiments/layer_fusion/lacf_layer_moe/`.
- Existing target features: frozen TF2's `eval_f1_speechfake_20261030T131151+0100/target_features/`.

The owner-authorized queue wrapper is preserved locally as the family `run.sh`;
it trains then evaluates only after successful source completion,stops on
failure,and rejects existing run folders. tmux session:`lacf-layer-moe-f1`.
Standalone commands,only for an authorized fresh run:

```bash
python -u -m scripts.experiments.layer_fusion.train_full_f1
python -u -m scripts.experiments.layer_fusion.evaluate_lacf_layer_moe
```

52 focused CUDA-hidden CPU checks passed before launch,including full source/
checkpoint/calibration gates,changed target-feature bytes/labels/crops/revisions,
shared metric integration and preservation of training artifacts. Initial
manifest-schema and JSON tuple/list compatibility failures were corrected
before GPU work; their test logs are retained. CPU checks do not establish live
GPU validation or completed training. Read live logs and completion markers.

## Interpretation and preserved history

The9November cache-loading attempt was owner-stopped before GPU preflight or
fitting. Its records remain under the old `layer_fusion/lacf_moe/` folder.
The10November authorization supersedes that hold for this one named run only.
Existing historical names and compatibility links remain intact.

This is an exploratory full-fold comparison after earlier target exposure.
Source development here is not the inner-held-source validation used by the
original diagnostic table. The comparison changes fusion/readout/capacity
relative to TF2; adapted TF2 also changes encoder adaptation and underwent
shutdown recoveries. It cannot isolate MoE versus attention alone. Do not
claim a complete four-fold study or select changes using these target results.
