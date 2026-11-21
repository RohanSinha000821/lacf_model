# NSF-PINN frozen detector adaptation

Preparation requested20November2026; owner explicitly accepts labelled
adaptations with recorded assumptions. Original:
[NSF-PINN, CVIU271:104875](https://doi.org/10.1016/j.cviu.2026.104875), supplied
as `references/sota/NSF-PINN.pdf`. This is a paper-informed implementation,
not verified author code or a reproduction of the published results.

Official WavLM **Base+**, revision
`4c66d4806a428f2e922ccfa1a962776e232d487b`, remains entirely frozen/eval/no-grad.
The paper's final recipe trains six Transformer blocks: our detector-only scope
is an explicit owner-requested adaptation. No Base/WA/fold/probe checkpoint
substitution. No official complete NSF-PINN checkpoint was confirmed.

Retained components and explicit choices:

- Last-layer768-D frame sequence, mean pooling,768→768→128 global SSL head.
- Shared framewise768→256→256 temporal MLP with GELU; separate256→128 source
  and filter heads; source/filter means plus SSL embedding form384-D input.
- Reconstructor256→256→768 and classifier384→256→2 with GELU/dropout0.2.
  The second temporal input/output width256 follows the described Z; the
  reconstructor and classifier hidden widths256 are unreported study choices.
  Text says two-layer SSL projection whereas Algorithm1 depicts three matrices;
  follow the explicit768→768→128 description and Fig.3.
- CE classification on two logits (Algorithm2/Fig.3), algebraically binary BCE
  on their difference; labels0=BF/1=spoof. No added class weighting.
- Reconstruction MSE over utterance-level768-D vectors; smoothness sums mean
  squared temporal differences of both frame sequences; squared cosine of the
  two pooled128-D vectors. Clamp each loss to[0,10] as Algorithm2 specifies.
  Reductions use Algorithm2's means, rather than Eq.19/23's dimension sums.
- Physics weights all0.1 are **unreported study choices**, not author values.
  Two-epoch linear warmup applies to all three physics terms (Eq.30).
  Algorithm2 instead leaves reconstruction un-warmed and uses absolute cosine;
  the implementation follows Eq.27 squared cosine and Eq.30 all-term warmup.
  These unresolved paper/algorithm discrepancies prevent exact reproduction.

SAM is retained, rho0.05, with AdamW3e-4/wd1e-4, batch32, per-update cosine
to zero, patience7/max100. LR/rho/batch/patience follow Section4.1; zero cosine
floor and maximum100 are explicit study choices (paper reports61 but no maximum).
Two genuine detector forward/backward passes restore original weights before
AdamW. Run the frozen encoder **once per batch**, reuse its detached features
for both passes. SAM updates every detector module including reconstruction.
SpecAugment/Mixup are off, disclosed omissions from the paper; their strengths
are unreported and removing them avoids an invented augmentation search.

Native inputs: mono16kHz, random64000 training/center64000 dev/test, right-zero-
pad shorter audio, no extra waveform normalization. Keep temporal features;
the existing pooled WavLM/CLAP cache cannot supply physics trajectories. No
full frame-level cache is created automatically; its storage cost is substantial.
Already16kHz recordings use exact seek/read of the chosen native segment;
other rates retain full resampling before cropping, preserving boundary samples.

Safe CPU inspection, with no pretrained/data access:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. \
  python scripts/nsf_pinn/train.py --fold f1 --inspect
```

Future command, **not launched by preparation**:

```bash
PYTHONPATH=src:. python scripts/nsf_pinn/train.py --fold f1 --seed 1234 \
  --data-root /mnt/drive/audio-deepfake-cache --num-workers 16 \
  --eval-workers 4 --prefetch-factor 2 --cpu-threads 2
```

Common training/preflight/checkpoint/source-export behavior is documented in
[LHCC preparation](../lhcc/README.md). Output family is
`outputs/nsf_pinn_frozen_v1/<fold>/<seed>/`, with native4s input and detector-only
checkpoint format. Canonical three-domain source train/dev only, strict macro
source EER selection and source calibration; no target feedback. LACF-MoE uses
different inputs, class sampling, losses and capacity: report these explicitly.
Native comparison and matched-duration controls are separate. No real-source/GPU
feasibility, runtime/VRAM measurement, fit or target evaluation has run during
implementation. The paper's4060/8GB observation does not prove our LODO batch
fits alongside existing jobs. Pretrained loading is local-only; Base+ was
previously found on SSH2, not SSH1. No automatic transfer/download is performed.
