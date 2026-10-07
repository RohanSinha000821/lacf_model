# SOTA paper reading notes for the LACF comparison

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
| Multi-level SSL feature gating | `SSL Feature gating.pdf`, Tran et al., ACM MM 2025: XLS-R 300M hidden-state gating, MultiConv, CKA dissimilarity, attentive pooling. | Paper describes full utterances with dynamic padding, batch 5, Adam LR `3e-6`, weight decay `1e-4`, weighted CE with bona-fide/spoof weights `0.9/0.1`, RawBoost, patience 3; final objective is written `L_CE + L_CKA`. [Author config](https://github.com/hoanmyTran/dissimilarity_deepfake_detection/blob/main/config.yaml) gives maximum 10 epochs and eval batch 1. | Released default config also has `max_len: 64600`, `rawboost.algo: 0`, `transform: null`, and monitors validation loss. Check whether its actual run path reproduces the paper's full-utterance/augmentation claims. Our common checkpoint rule uses source macro EER, so that is an explicit study adaptation. |
| SAM-AASIST | `SAM Anti Spoofing.pdf`, Huang et al., Interspeech 2025: sharpness-aware minimization applied to AASIST; SAM is an optimizer, not a standalone detector. | 4 s crop, RawBoost, weighted CE with bona-fide/spoof weights `0.9/0.1`, LR `1e-4`, weight decay `1e-4`, cosine floor `5e-6`, SAM radius `0.05`. [Author SAM-AASIST config](https://github.com/nii-yamagishilab/SAM-AntiSpoofing/blob/main/configs/sam/aasist_sam.yaml) sets batch 24 and 100 epochs. | The paper's `m=32` describes sharpness measurement, not its training batch. The released config also names an In-The-Wild validation set; it must not be used when that domain is held out. Replace all validation inputs with the fold's source development splits. |
| LHCC | `LHCC.pdf`, Zhang et al., Information Fusion 134 (2026) 104358: low/high-level cross-layer consistency from XLS-R features through AFM and DCM. | Section 4.4: 16 kHz mono, exactly 64,600 samples through truncate/repeat, XLS-R 300M, batch 8, Adam LR `1e-6`, weight decay `1e-4`, maximum 10 epochs, early-stop patience 2, RawBoost. Weighted BCE is stated without numeric class weights. | Paper defines bona fide=1/spoof=0 and spoof score `1 - p_bona_fide`; convert carefully to project convention. It appears to call XLS-R frozen in one place while its algorithm describes joint end-to-end optimization; determine actual intended backbone training before implementation. Earlier project summary incorrectly treated batch/LR/epochs as unreported; use the paper's explicit values. |
| AASIST | `AASIST.pdf`, Jung et al., ICASSP 2022: integrated spectro-temporal graph attention on raw waveform. | 16 kHz, 64,600-sample input, Adam LR `1e-4`, cosine schedule; paper averages three seeds. [Author config](https://github.com/clovaai/aasist/blob/main/config/AASIST.conf) specifies batch 24 and 100 epochs; the released training implementation supplies weight decay `1e-4` and cosine minimum `5e-6`. | Its released config can evaluate a held-out set whenever development improves. The LODO implementation must not use target results for selection. Map original label order and class weights to project bona-fide=0/spoof=1 convention. |

Publication policy: all models share source/target folds, target isolation, score direction, selection criterion and evaluator. Seeds follow the owner's declared single-seed compute-budget exceptions for WavLM and AASIST below; disclose unequal replication. Preserve model-specific input and optimization recipes, since equal batch sizes would not mean equal compute or equal methodological fidelity. If batch 128 is explored for AASIST or SAM-AASIST, give it a separate sensitivity result selected solely from source development data; do not silently substitute it for the paper-informed primary configuration. Report parameter count, memory, wall-clock time and optimizer updates so efficiency differences are visible. Do not include an interrupted WavLM pilot in the final comparison table.

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
