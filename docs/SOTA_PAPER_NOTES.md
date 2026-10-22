# SOTA paper reading notes for the LACF comparison

## Owner-proposed frozen layer fusion (implementation, 2026-10-22)

Reread the actual main DOCX including tables and loss/adapter equations. C0–C4
are a later owner-requested experiment, not an architecture prescribed by that
original plan. Preserve its CLAP anchor geometry, full35-D relation features
and adapter/classifier; explicitly replace the primary three-loss objective
with detection BCE only for this comparison. The new experts/router are
isolated in `layer_fusion/`. This adds a no-auxiliary full-relation control (C0),
uniform and global layer averaging controls, and near-matched global/input
expert routing. It does not rewrite A4–A9 or rename this comparison as A10.
See [the experiment guide](LAYER_FUSION_EXPERIMENTS.md). No MoE advantage,
novelty claim or fine-tuning benefit is established by implementation/CPU tests.

Step4 fidelity check,19October2026: reread the original main DOCX including
equations/tables and supplementary pipeline/model-plan sections on adapter,
pre-linear LayerNorm classifier and A4–A9. The stage uses raw last-layer WavLM,
not the learned mixture studied in Step3. Preserve semantic0.25 where defined;
A8 removes BF loss, making A9 with BF weight0 the same control. Report that
alias transparently instead of treating a duplicate fit as independent evidence.
Diagnostic cached CLAP projections receive the original API/primary FP32
normalizations. Shared module construction keeps primary state keys/counts;
exact synthetic backend/loss/gradient parity is checked. No primary redesign.

## Source-only information-loss probes (owner authorization, 2026-10-18)

Reread the complete main LACF DOCX including tables/equations and relevant original
pipeline/master-plan pages for Step 3. The owner's later diagnostic instructions
override the older experimental ordering: frozen-feature logistic/MLP probes use
inner leave-one-source-out on F1–F3 and seeds1234/1235/1236. They do not replace
the initial single-seed primary study. Raw pooled WavLM or a fresh learned
13-layer mixture is probed without the anchor adapter; CLAP projected features
use existing L2 normalization. Concatenation combines the learned mixture and
normalized CLAP. The MLP reuses the prescribed 64→16 hidden/head ordering, and
logistic is one linear layer. Pure detection BCE applies to these embedding
probes; anchor-specific auxiliary losses belong to the later relation ablations.
Approved optimization settings are reused, not selected from target performance.
No trained encoder, old adapter or old layer weights initialize the probes.

Evaluation execution fix, 2026-10-06: the initial shared-GPU 8 GiB cap, not
physical GPU exhaustion, stopped native WavLM scoring on a 124-second recording.
The owner requested no allocator ceiling. Full-utterance batch-one BF16 scoring,
checkpoint selection, source thresholds and normalization are unchanged.
Checkpoint-bound CSV recovery preserves verified prefixes; the known original
F1 pre-journal prefix requires explicit adoption, recorded with its hash/count.
This is an I/O/memory execution change, not a new paper-derived model recipe.

Owner decision, 2026-10-06: completed WavLM batch-96 folds may be evaluated
independently now, with source-only checkpoint/threshold selection and no
target-guided tuning of any model. This revises the earlier comparison-wide
target-access timing, not the paper recipes or score contract. F1/F2/F3 are a
partial result; F4 will be added using its own later completed checkpoint,
without rerunning earlier folds. Record the independent-run authorization;
do not claim that the wider LACF/baseline comparison was already frozen.

Read on 2026-10-01. These are research notes, not instructions from the papers to change this project. The experimental decisions live in `EXPERIMENT_PROTOCOL.md`. Source papers are the five PDFs supplied by the project owner; author configurations are linked below. Preserve this file with the project so the paper-specific facts and unresolved reproduction questions survive beyond this chat.

Project decision update on 2026-10-02: the new WavLM family uses batch 96 under
`wavlm_bs96`. The batch-64 discussion below records the earlier decision; its
completed F1 checkpoint remains separate. The paper's batch 32 is unchanged as
a reported fact. Batch-96 feasibility passed one synthetic optimizer step;
F1 must be retrained at 96 before combining the four folds in the new family's
final comparison. See the latest protocol for the queue and target-freeze gate.

| Method | Paper and key idea | Reported training facts | Reproduction issue for this project |
| --- | --- | --- | --- |
| WavLM-WA | `Wavlm- Backend.pdf`, Stourbe et al., ASVspoof 2024: weighted aggregation of WavLM Base representations with a spoofing backend. | Best WA system: 4 s random training crop, full-utterance test, batch 32, Adam with encoder LR `2e-5` and backend LR `5e-3`, 5% LR decrease per epoch, weighted CE with bona-fide/spoof weights `9/1`; maximum 100 epochs and patience 50. The reported best system used MUSAN noise, RIR and codec augmentation. | Project-selected batch 64 is an adaptation. Current script has no augmentation, initial-weight regularization or layer-wise LR decay; describe results as a simplified baseline unless those components are added. A prior batch-96/128 checkpoint is a pilot, not a batch-64 final run. |
| Multi-level SSL feature gating | `SSL Feature gating.pdf`, Tran et al., ACM MM 2025: XLS-R 300M hidden-state gating, MultiConv, CKA dissimilarity, attentive pooling. | Paper describes full utterances with dynamic padding, batch 5, Adam LR `3e-6`, weight decay `1e-4`, weighted CE with bona-fide/spoof weights `0.9/0.1`, RawBoost, patience 3; final objective is written `L_CE + L_CKA`. [Pinned author config](https://github.com/hoanmyTran/dissimilarity_deepfake_detection/blob/4de821cad840ef6b4f78905f9cce0523c956d988/config.yaml) gives maximum 10 epochs and eval batch 1. | Resolved 2026-10-18: `transform:null` disables the 64,600-sample path; YAML disables RawBoost. Released CKA detaches its gradient, and block/time `.view` misorders features. Our disclosed adaptation repairs these, retains no RawBoost and uses source macro EER. See the implementation findings below; this is not an exact published-result reproduction. |
| SAM-AASIST | `SAM Anti Spoofing.pdf`, Huang et al., Interspeech 2025: sharpness-aware minimization applied to AASIST; SAM is an optimizer, not a standalone detector. | 4 s crop, RawBoost, weighted CE with bona-fide/spoof weights `0.9/0.1`, LR `1e-4`, weight decay `1e-4`, cosine floor `5e-6`, SAM radius `0.05`. [Author SAM-AASIST config](https://github.com/nii-yamagishilab/SAM-AntiSpoofing/blob/main/configs/sam/aasist_sam.yaml) sets batch 24 and 100 epochs. | The paper's `m=32` describes sharpness measurement, not its training batch. The released config also names an In-The-Wild validation set; it must not be used when that domain is held out. Replace all validation inputs with the fold's source development splits. |
| LHCC | `LHCC.pdf`, Zhang et al., Information Fusion 134 (2026) 104358: low/high-level cross-layer consistency from XLS-R features through AFM and DCM. | Section 4.4: 16 kHz mono, exactly 64,600 samples through truncate/repeat, XLS-R 300M, batch 8, Adam LR `1e-6`, weight decay `1e-4`, maximum 10 epochs, early-stop patience 2, RawBoost. Weighted BCE is stated without numeric class weights. | Paper defines bona fide=1/spoof=0 and spoof score `1 - p_bona_fide`; convert carefully to project convention. It appears to call XLS-R frozen in one place while its algorithm describes joint end-to-end optimization; determine actual intended backbone training before implementation. Earlier project summary incorrectly treated batch/LR/epochs as unreported; use the paper's explicit values. |
| AASIST | `AASIST.pdf`, Jung et al., ICASSP 2022: integrated spectro-temporal graph attention on raw waveform. | 16 kHz, 64,600-sample input, Adam LR `1e-4`, cosine schedule; paper averages three seeds. [Author config](https://github.com/clovaai/aasist/blob/main/config/AASIST.conf) specifies batch 24 and 100 epochs; the released training implementation supplies weight decay `1e-4` and cosine minimum `5e-6`. | Its released config can evaluate a held-out set whenever development improves. The LODO implementation must not use target results for selection. Map original label order and class weights to project bona-fide=0/spoof=1 convention. |

Publication policy: all models share source/target folds, target isolation, score direction, selection criterion and evaluator. The owner's single-seed compute-budget exceptions use seed 1234 only for families `wavlm_bs96`, `aasist` and initial `lacf`/`lacf4s`; disclose unequal replication. Preserve model-specific input and optimization recipes, since equal batch sizes would not mean equal compute or equal methodological fidelity. If batch 128 is explored for AASIST or SAM-AASIST, give it a separate sensitivity result selected solely from source development data; do not silently substitute it for the paper-informed primary configuration. Report parameter count, memory, wall-clock time and optimizer updates so efficiency differences are visible. Do not include an interrupted WavLM pilot in the final comparison table.

## P2–P4 reference findings and implementation (2026-10-05)

Read the actual main plan's evaluation section and the supplementary new plan's
normalization contract, plus relevant portions of the CFAD, SpeechFake,
ASVspoof 5 and ASVspoof 2019 dataset papers. P2 external tests, P3 generator
decompositions and P4 robustness support the corresponding comparisons for all
included models, without changing their native preprocessing or training recipe.
The existing approved macro-source-dev selection criterion remains unchanged;
the main plan's pooled source-dev wording does not override that later decision.

Verified local metadata: SpeechFake `test_all.csv` has `generator` values TTS,
VC, NV (and `-` for genuine), so no generator classification is inferred from
model names. ASV5's distributed README specifies ten fields despite its stale
"FIVE columns" heading; codec is field 4, original `CODEC_SEED` field 6, attack
field 8, label field 9 (one-based). Codec '-' is uncoded. The paper describes
an exhaustively coded subset and singly coded remainder, so paired robustness
must restrict to actual original-ID intersections, not assume every utterance
has every codec. CFAD paper Tables IV/V describe independently distributed
clean/noisy/codec partitions and seen/unseen conditions; local filenames carry
noise/SNR or codec suffixes, preserved as metadata, not newly generated audio.

Additional primary references inspected: [official ASVspoof 2021 key schema](https://github.com/asvspoof-challenge/2021/blob/main/eval-package/README.md),
the installed MLAAD README/meta.csv schema, and the installed PartialSpoof
README/utterance-level eval protocol. DF full keys identify eval/progress/hidden
and notrim/trim; use eval/notrim only. The local original trial-ID files and
VCTK/VCC source-mapping package do not provide these labels. MLAAD requires the
referenced genuine M-AILABS audio; PartialSpoof is an utterance stress test in
this implementation, not a segment-localization experiment. Neither missing
keys nor genuine audio were fetched or fabricated. The shared evaluator is
implemented for registered score CSVs, with native exporters for the two models
already implemented. Real-data/GPU evaluation and the global freeze remain
pending; a source-only P2 deployment-checkpoint choice must still be recorded.

## WavLM seed-budget decision (2026-10-05)

The owner limits the current WavLM batch-96 family to seed 1234 across the four folds because of training time. Do not run additional WavLM seeds 2345/3456 or substitute the old batch-64 F1. This is a project compute-budget adaptation, not a paper-derived recipe or a finding of seed stability. Report single-seed point estimates and conditional target-bootstrap intervals, without across-seed standard deviation. Other models' seed plans are unchanged; disclose this unequal replication in comparisons. The comparison-wide target freeze remains required.

## AASIST implementation provenance (2026-10-04)

- Architecture: [authors' AASIST.py at pinned commit](https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/AASIST.py), copied byte-for-byte to `sota/aasist_arch.py`; SHA-256 `9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50`. NAVER's MIT notice is included as `sota/AASIST_LICENSE`.
- Numerical recipe: [AASIST.conf](https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/config/AASIST.conf), [optimizer/scheduler implementation](https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/utils.py) and [training implementation](https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/main.py). Cosine updates follow optimizer steps; frequency augmentation defaults to false when absent from the config. The ordinary baseline therefore does not add frequency masking or SAM's RawBoost.
- Input policy: [authors' data_utils.py](https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/data_utils.py) uses random train crops, first-segment dev/test truncation and repetition of short audio. Our adapter fixes the exact-length `randint(0)` case and permits the final valid crop start. Non-16 kHz data are resampled before applying the fixed sample policy.
- LODO adaptations: output logits reordered from original `[spoof, bona fide]` to `[bona fide, spoof]`; CE weights reordered accordingly; domain-balanced replacement draws; no dropped final batch; macro-source-dev checkpoint selection; patience 5; no upstream SWA stage, target-triggered selection or in-training target scoring. No pretrained ASVspoof weights are used, preventing contamination of F4's held-out dataset.
- Verification: 297,866 parameters; synthetic CPU full batch 24 × 64,600, FP32 forward/backward/Adam step passed with finite loss and gradients. Real-data/GPU feasibility and complete source-only development/reload checks remain pending. No target scores were read or generated by implementation tests.

## AASIST single-seed adaptation (owner decision, 2026-10-06)

The owner selects seed 1234 only for AASIST family `aasist`, across F1–F4.
This supersedes the previous three-seed project plan; architecture, batch 24,
FP32 and the source-only training/selection recipe are unchanged. The original
paper reports averages and best results from three independent seeds; our
single-seed study is a disclosed reproduction adaptation, not that three-run
average or evidence of seed robustness. Do not select the seed on target scores
or claim published performance is guaranteed. Report one point estimate per
fold, conditional target-bootstrap intervals and no across-seed SD. Real-data/GPU
validation remains pending; no AASIST training was launched by this decision.

## AASIST source-only GPU feasibility (2026-10-07)

The actual AASIST paper was consulted again before the owner-authorized F1
launch. Architecture, batch 24, FP32, input policy, optimizer, sampling,
source-only selection and the single-seed adaptation remain unchanged.
Three real source-data GPU optimizer steps passed with finite loss/gradients;
checkpoint reload reproduced scores on six source-dev examples within the
recorded tolerance. A fresh seed-1234 F1 model was then started; pilot weights
are not transferred. This is a feasibility/correctness check rather than a new
scientific candidate or a completed full-development validation. Full source-dev
scoring and selected-checkpoint reload checks remain pending. Pilot evidence and
measured CUDA peaks remain in local ignored outputs and the handoff; no target
results informed these actions or adaptations.

## LACF main-plan fidelity (2026-10-08)

The following initial precision/seed/batch choices are historical; the later
owner-approved Part 1 decisions below supersede them.

Read the actual owner-designated `LACF_Audio_Deepfake_Implementation_PlanV2.docx`
completely, with table cells and mathematical objects, alongside the original
`Updated_Pipeline.pdf` and `New_plan_LACF_Model.pdf` supplements. LACF is kept
outside SOTA code. The primary architecture is implemented; only synthetic CPU
behavior has been verified. The exact eight concepts/three templates, normalized
template-mean prototypes, Base+ masked last-state pool, 768→512 adapter,
35-dimensional relation ordering, 35→64→16→1 head, 0.07 temperature and
`L_det + 0.25 L_sem + 0.10 L_BF` follow those originals.

The pipeline's classifier diagram resolves the main DOCX's brief wording:
LN before 35→64 and before 64→16; GELU/Dropout(0.2) after each, followed by
16→1. Later approved macro source-EER selection/centered dev cropping and the
recorded batch 4×8/constant-LR assumptions remain authoritative. No pooled
source-EER checkpoint selection or all-four-source retraining is introduced.

The owner delegated missing processor/precision choices after they were flagged.
FP32 and standard checkpoint processors are recorded implementation choices,
not settings claimed to appear in the main plan. [WavLM Base+ configuration](https://huggingface.co/microsoft/wavlm-base-plus/raw/main/preprocessor_config.json)
sets waveform normalization false and returns validity masks. [Unfused CLAP configuration](https://huggingface.co/laion/clap-htsat-unfused/raw/main/preprocessor_config.json)
sets 48 kHz, a 10 s feature window and repeat-padding; the selected physical
segment prevents its random long-audio truncation from selecting another view.
The installed Transformers API returns projected 512-dimensional CLAP features.
No CLAP paper original is supplied in the canonical reference set. No weights
were downloaded; all model-loading paths are local-only.

Frozen encoders also stay in eval mode, preventing dropout/augmentation/state
updates. Auxiliary-loss switches omit inapplicable terms explicitly and feature
configuration determines head dimensions/unused branches. FT4 and ablation
execution remain deferred. Full source/GPU validation, final freeze and native
LACF score export remain required before comparison results.

## LACF Part 1 review and owner adaptations (2026-10-08)

The main DOCX (including tables and equations), checksum-verified original
pipeline/model supplements and current implementation were reviewed again.
Exact architecture/prompts/losses/input interval remain unchanged. The owner
now selects seed 1234 only, BF16 mixed precision with explicit FP32 sensitive
calculations, and configurable physical batch/accumulation/loaders. These are
owner adaptations, superseding the prior FP32/three-seed/fixed-batch defaults;
no target result informed them. Effective 96/128 remain examples until validated.

Parameter accounting is 657,920 adapter + 3,559 head = 661,479 trainable.
Encoder normalization and CLAP's internal projected-feature normalization are
kept FP32 as well as backend normalization/probabilities/JS/entropy/losses.
Encoders remain frozen and eval. Source-dev BF16 matches training; selected
checkpoint reload is validated before fold completion. AdamW 3e-4/1e-4 and
constant LR do not scale with batch. The sequential launcher stops on failures
or completion/hash/source-reload mismatches; no training resume is added.

The review/tests use synthetic CPU fixtures with CUDA hidden. Actual cached
checkpoints, real source audio, CUDA BF16, memory and throughput remain
unvalidated. No GPU test, training, download, target evaluation or publication
was performed. Single-seed summaries have no across-seed SD; the shared
conditional target-bootstrap contract is unchanged.

Later owner decision (2026-10-08): initial LACF may proceed while transfer
checksum verification remains pending, using the explicit unverified-copy
acknowledgment and truthful provenance. This does not waive canonical source
membership/counts/file decoding, source-only GPU checks, full source-dev
scoring/reload or the common recipe freeze, and does not establish byte integrity.
Official checkpoints have now been separately authorized for acquisition;
actual validation progress and measurements are local run records, not synthetic
test evidence or completed target results.

Subsequent owner report (2026-10-08): SSH1's checksum verification completed at
05:35 with all 1,243,408 manifest files matched and no differences/failures.
SSH2 records the owner-relayed evidence; no independent SSH2 checksum rerun is
claimed. The pending-copy exception is historical. Full source-only GPU,
source-dev/reload and recipe-freeze gates remain required before final training.

## Supplied SOTA summary and optimization review (2026-10-09)

Read the complete 11-page `SOTA_Model_details.pdf`, including tables/equations,
from the verified local extraction. The newly attached PDF is byte-identical to
the indexed original (SHA-256 `f459a3dc0e0f59a96db791176fb6c7f16cd499ae9fdc8d46fa7a45251f5a70e2`).
It supports the LACF-4s duration comparison alongside primary LACF-10s. Original
papers/released configurations and later owner decisions still resolve recipe
conflicts. Owner now authorizes 4 s F1→F4 first, then fresh 10 s F1→F4; both use
seed 1234 and equal effective batch. WavLM's native full-utterance dev/test remains
explicitly different from the later matched-four-second scoring control.

The stopped first 10 s run's two epochs/checkpoints remain preserved. Optimized
source-dev execution caches only frozen features of fixed centered crops, then
recomputes current trainable backend scores each epoch. Training random crops are
not cached. Encoder/backend development batches remain one. CPU resampler reuse,
fewer device/CPU barriers and clearer phase/progress logs preserve the primary
recipe. CLAP's fixed 10 s feature window remains repeat-padded for 4 s inputs;
reducing physical evidence does not proportionally reduce its encoder compute.

289 CUDA-hidden synthetic tests passed. Bounded official-weight source GPU
checks are recorded separately from full canonical source-dev/reload validation;
modified code requires new duration-specific freeze records before final runs.
Short training probes measured ~82→92 audio/s at 10 s and ~176 at 4 s, batch 132.
No target performance was consulted, no SOTA model recipe changed, and no
completed four-fold result or independent SSH2 transfer checksum audit is claimed.

Latest owner override, 2026-10-09: stop the4 s-specific work and train10 s only.
The duration comparison is deferred. Preserve old full10 s source/reload evidence
and validate the optimized execution against its saved pilot scores in bounded
source-only regression; record the reuse scope explicitly. First-development
frozen-feature extraction is still a full pass, but is performed after the first
training epoch, rather than rerunning both old validation passes before training.
All approved numerical choices and source-only checkpoint selection stay fixed;
no target result informed this scheduling change.


Owner evaluation-timing decision, 2026-10-12: removed comparison-wide freeze
as a prerequisite for all completed SOTA evaluations. This changes operational
timing, not architecture, published-recipe adaptations, source-only selection,
calibration or metrics. No target results may inform any model's tuning/design.
Earlier per-family access restrictions are historical; reference documents do
not override this later owner decision.

Owner plan, 2026-10-12: stop the LACF sequence after normal, verified F2
completion and hold F3/F4. The owner clarified this is planning only; SSH1
artifact copying will be arranged separately. No exporter implementation or
evaluation launch is authorized by this plan alone. Native P1 evaluation of
frozen F1/F2 and comparison with matching completed WavLM batch-96 folds are proposed as a preliminary
two-fold study. Native input policies differ (LACF centered up-to-10 s;
WavLM full-utterance dev/test), and frozen versus fine-tuned encoders are part
of the system comparison. This does not isolate either factor or establish a
completed four-fold comparison. The later owner instruction authorizes source-only
preparation and the minimal native exporter. These are implemented and their
F1/F2 source checks have passed; target evaluation remains deferred.
Target-informed revisions are exploratory and require independent confirmatory
data for new generalization claims; source calibration remains fixed.

The padding review identified native WavLM Base+ behavior: its first GroupNorm
includes right-zero-padded time steps before the attention mask is applied.
Masked pooling does not undo this dependence. Measure feature/score changes on
source recordings before deciding whether a revised training recipe is needed;
the mechanism alone does not establish an EER cause or invalidate completed runs.
An evaluation/reporting fix needs no retraining. Changing encoder normalization
or training features would require fresh runs for consistent revised F1/F2 rows;
preserve the original checkpoints and recipe when evaluating the existing system.

Later owner instruction, 2026-10-12: native P1 evaluation of the preserved F1/F2
LACF models is now authorized after the passed preparation. F1 SpeechFake test
then F2 CFAD clean unseen test run sequentially from the local cache, with the
original normalization/precision/scoring recipe. This permits evaluation only;
no target feedback may select a feature correction, architecture or training
setting. F3/F4 remain held and a two-fold comparison remains preliminary.

AASIST execution-only adaptation, 2026-10-14: the released sinc frontend builds
constant filters at initialization but clones/transfers them each forward.
An optional adapter subclass stores identical filters as a nonpersistent device
buffer, preserving convolution arguments, parameters and checkpoint keys; the
vendored original remains unchanged. Cache defaults off, and masking falls back
to the original implementation. Exact CPU frontend outputs/input gradients and
full-network eval scores match, but source GPU equivalence/performance remain
unvalidated. Detached loss reporting now avoids its additional per-batch device
to host transfer; the finite-loss guard and optimizer recipe are retained.
No precision, augmentation, sampler, crop or architecture change is implied.


AASIST operational update, 2026-10-14: after completed F2, owner authorized
parallel F2 P1 evaluation and fresh F3 training. Bounded source-only GPU checks
passed exact original/cached scores, training gradients and eight optimizer
updates. A short preloaded-data probe measured ~31.65 to ~34.02 audio/s; this
is not an end-to-end speedup guarantee or full-development validation. F3 uses
seed 1234, unchanged batch-24 FP32 recipe, 16 train workers/8 dev workers,
prefetch 2, CPU threads 2 and the explicitly enabled fixed-filter cache. The
option's global default remains off. Worker-count changes alter per-worker
random crop streams but retain their policy/distribution. Full selected-source
reload validation remains required. F2 scoring retains batch one and its source
calibration gate; no F4 or F3 target testing is launched. Local evidence is in
`outputs/aasist_f3_launch_20261014/` and `outputs/aasist_f2_evaluation_20261014/`.


## SSL feature gating: paper/code review and adaptation (2026-10-18)

Read the complete ten-page supplied `SSL Feature gating.pdf`, including its
experimental tables, the relevant actual overall DOCX and SOTA overview, and
the selected released model/data/optimizer/pooling/MultiConv paths. Primary
sources are the [paper](https://doi.org/10.1145/3746027.3754568) and
[author repository at 4de821c](https://github.com/hoanmyTran/dissimilarity_deepfake_detection/tree/4de821cad840ef6b4f78905f9cce0523c956d988).
The repository is MIT licensed; its inherited ESPnet MultiConv and asv-subtools
pooling operations use Apache-2.0 notices, retained alongside the adapter.
No pretrained anti-spoofing checkpoint or author dataset CSVs are imported.

Resolved discrepancies and declared adaptations:

- Full recordings are intentional. `transform:null` bypasses the released
  64,600-sample transform; its `max_len` value is inactive. The training collator
  right-zero-pads to the batch maximum, and neither the frontend nor pooling
  receives a padding mask. Preserve that behavior; evaluation at batch one
  has no added padding. The common mono loader/torchaudio resampler replaces
  released librosa 0.9.2, without waveform standardization.
- Paper section 4.3 and Table 4 include RawBoost. Released YAML sets algorithm
  zero; dataset CLI's default four does not override that zero. After the owner
  requested fidelity and lower cost, select no RawBoost. This omits a component
  credited in the paper and must be disclosed. XLS-R's own feature masking,
  dropout and layerdrop remain enabled; “no RawBoost” does not disable those.
- Released `cka(...).item()` and `torch.tensor(cka_values)` detach the auxiliary
  objective. The adapter keeps its gradient, using centered linear Gram
  matrices and the released denominator epsilon 1e-6. Common HSIC scaling
  cancels. Only the zero-variance denominator is safely clamped before sqrt;
  singleton/constant batches yield zero CKA with finite gradients.
- Released `[B,4,T,128].view(B,T,512)` interleaves block/time positions. Use
  `cat(block_outputs,dim=-1)` so every position contains its four block
  features. Four pooling heads now align with those four blocks; CKA uses
  the four 128-D pooled means, averaged over six unordered distinct pairs.
  The paper's diagonal-inclusive-looking notation is ambiguous; the released
  upper-triangle implementation excludes diagonal pairs. Coefficient is 1.
- Preserve the configured four concat-fusion MultiConv blocks: expansion 1024,
  kernels 3/7/11/15, depthwise merge kernel 15, SiLU gate, dropout 0.1 and
  ESPnet LayerNorm epsilon 1e-12. The outer gating uses actual SiLU-based
  SwiGLU from the code, rather than the paper equation's sigmoid-only form.
  Four-head attentive mean/std pooling feeds a 1024→512→2 SELU classifier.
- Released YAML declares `optimizer_gamma:0.9`, but the actual optimizer has
  no scheduler. Preserve Adam 3e-6, weight decay 1e-4, constant LR, batch 5,
  FP32, maximum 10 epochs and patience 3. XLS-R is fine-tuned end to end.
- Released validation loader erroneously returns the training dataset; YAML
  points validation at an ASVspoof evaluation CSV; model construction also
  reads evaluation metadata. None is retained. Use only each fold's source
  train/dev readers, equal-domain natural-class sampling and strict macro
  source-dev EER selection instead of released validation loss. Seed before
  constructing the model. Reorder logits/CE weights to project labels and
  export `logit_spoof-logit_bona_fide`; do not fit score direction to targets.

Official pretrained frontend:
[facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m/tree/1a640f32ac3e39899438a2931f9924c02f080a54),
revision `1a640f32ac3e39899438a2931f9924c02f080a54`, 24 Transformer layers plus
projected frontend state, 1024 dimensions. The loader is offline-only and
pinned; reload reconstructs from the saved frontend configuration and strict
full model state without downloading another encoder. The common environment
uses Transformers 4.57.6/Torch 2.8.0 instead of released 4.42.3/2.3.1; SDPA was
already supported in that released Transformers version. Project before
stacking levels to avoid an extra full-dimensional hidden-state stack; this
preserves the mathematical gate but may change floating-point rounding.
Do not claim byte-equivalent full-system scores or training trajectories.

Verification is synthetic CPU only: differentiable CKA against centered-Gram
math, nonzero auxiliary gradients into the frontend, finite degenerate-case
gradients, all-block updates, reload, full audio/padding, source-only selection
and native score/metric provenance. A separate direct comparison with pinned
upstream operations gives exact CPU outputs and input gradients for selected
MultiConv and pooling paths after state mapping. Parameter counting from the
official configuration on the meta device gives 318,083,586 total/trainable:
315,438,720 frontend and 2,644,866 backend. This is not a GPU allocation or
throughput measurement. Local evidence is in
`docs/local/ssl_gating_implementation_20261018/source_parity.json`.
Actual source-data/GPU feasibility, encoder short-recording handling and full
source-dev/reload validation remain pending. No pretrained weights were fetched,
no job launched and no claim of reproducing published target results is made.

Preparation update, 2026-10-18: the owner requests readiness without training.
Downloaded the exact pinned official XLS-R files on SSH1; the weight SHA-256
matches the published LFS checksum. Actual offline encoder loading reports no
missing/mismatched keys; seven unused pretraining projection/quantizer keys
are expected when extracting Wav2Vec2Model from a pretraining checkpoint.
The full adapter passes finite synthetic CPU inference and exact repeated
inference with CUDA hidden. All 318,083,586 parameters remain trainable.
Bounded F1 source-cache decoding is a separate CPU input check, not a completed
source/GPU validation or fitted experiment. No optimizer was created.

Fine-tuning clarification: paper section 4.3 explicitly fine-tunes the model,
and the released optimizer covers the complete encoder plus backend. The
existing implementation already does this: initialize XLS-R from public speech
pretraining, initialize the detector backend freshly, then fit both together
on source training only. Retain that approach for methodological fidelity;
there is no additional pretraining-from-scratch stage. Freezing XLS-R would
reduce training compute and memory but change the baseline. Neither approach
can be claimed superior for unseen-domain generalization without source-only
evidence; no frozen variant or target-driven selection was introduced.


SSL-gating launch update, 2026-10-18: the owner requested F1 with16 training
workers. Three source-only batch-five FP32 GPU optimizer steps have finite
loss/gradients; six source-dev scores are exactly reproduced after checkpoint
reload (tolerance1e-5). Peak8.04GiB allocated/8.60GiB reserved is a bounded probe,
not a full-memory or throughput result. Source batches were unmodified full
recordings, dynamically padded; no target opened and pilot weights did not
initialize the fresh F1 run. Its epoch1 then exceeded the protective10GiB
allocator budget alongside active AASIST; no valid epoch/checkpoint completed.
The owner elected to keep F1 stopped, with no restart queued. This failure does
not establish inability to fit a free35GiB device and does not authorize
changing batch, crops, precision or freeze policy. Local evidence is preserved
under outputs/ssl_gating_f1_launch_20261018/. No published performance claimed.


## SSL-gating source batch feasibility (2026-10-22)

Owner-authorized F1 source GPU checks retained full recordings, end-to-end
FP32, Adam and CE+CKA, with a19GiB allocator ceiling to leave CUDA overhead
within an approximately20GiB process budget. Batch5 passed a bounded22.096s
check but failed after the actual sampler exposed28.128s recordings; batches6
and8 also failed. These are allocator-budget failures, not proof of full-device
OOM. Smaller batches were measured but the later explicit owner constraint is
batch5orhigher; none was adopted. The documented batch5 recipe remains held,
with no final training/restart or crop/precision/freezing change. Bounded
source checks are not full-development validation or worst-case memory proof.
Local evidence: docs/local/ssl_gating_batch_benchmark_20261022T152059/.

Paper section4.3's full-recording dynamic padding is retained. The released
transform:null makes its64,600-sample max_len inactive. Memory-saving execution
changes would need source-only parity and throughput checks; no activation
checkpointing or other optimization was applied by this benchmark.


## SSL-gating20s adaptation and launch checks (owner decision, 2026-10-22)

After reviewing the paper, source memory failures and external advice, the owner
authorizes version2: random contiguous20s training crops only for longer audio,
after16kHz resampling, with unchanged short audio and dynamic batch padding.
Retain exact batch5/FP32, optimizer/loss/sampler and full-recording batch-one
source development/test. Tag new runs ssl_gating_20s and preserve old uncapped
failures. This supersedes the preceding full-training-input hold; it is a
source-feasibility adaptation rather than an exact paper reproduction. No cap
was selected using development EER or target scores. Unmasked padding remains
a disclosed training/evaluation difference; no masking or bucketing added.

Existing Step2 source metadata on SSH2 was read for F1 sources only. Train
maxima are13.188s/28.912s/60.013s for ASV2019/ASV5/CFAD;887ASV5 and15CFAD
train recordings exceed20s. Source dev maxima are11.594s/22.435s/60.013s.
Canonical counts and the selected longest-dev file headers were verified on
SSH1; no held-out SpeechFake/ASV5eval/other target inspected for this decision.
This metadata reuse is not a new full-audio/content-integrity scan.

129 related CUDA-hidden synthetic tests pass. Real-source12batch5×20s updates
pass with finite losses/gradients, including a deliberate all24-layer execution
check with ordinary GPU dropout and SpecAugment active:15.740GiB peak allocated,
17.777GiB reserved under19GiB allocator ceiling. Nine full source-dev scores,
including60.013s CFAD, reproduce exactly after fresh strict reload. All20s
stress throughput~1.97audio/s under shared load is not representative epoch
throughput. Full source calibration remains a later requirement. No activation
checkpointing or mixed precision needed for these bounded checks. Final F1
starts from original pretrained XLS-R and fresh head, never timing weights.
Evidence: outputs/ssl_gating20s_f1_launch_20261022/. SSH2 is prepared for later
source training only; additional folds/seeds and target evaluation are not queued.
