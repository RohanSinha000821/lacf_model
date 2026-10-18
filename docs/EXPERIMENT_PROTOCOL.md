# Cross-domain experiment protocol (v4)

## Owner-staged information-loss investigation (2026-10-14)

The owner requests diagnostics before redesign, with results shared after each
step before proceeding. Step 1 is authorized now: preserved LACF F1/F2 source
development and already-evaluated F1/F2 targets only, and learned weights from
completed WavLM-WA F1–F3 checkpoints. This is descriptive checkpoint diagnosis,
not new training, calibration, a changed score recipe or target-based tuning.
Keep the existing LACF F3 evaluation running and report its completed result.
GPU Step 1 inference waits for that evaluation to finish rather than competing
with it. CPU checkpoint/prototype/layer-weight inspection may proceed now.

WavLM F4 training remains held by the owner's latest instruction. F4 and external
sets (ASVspoof 2021 DF, MLAAD/M-AILABS and PartialSpoof) must not be inspected or
evaluated until a final model is frozen. Later design selection must use inner
leave-one-source-out splits: two source domains for fitting, third source domain
for validation, rotated within each fold. F1/F2/F3 target results cannot select
new design choices. Steps 2–4 are subsequent stages, not an automatic queue.

Future trained configurations from Step 3 use owner-specified seeds 1234, 1235
and 1236; existing seed-1234 primary runs/results are unchanged.

Later owner authorization, 15 October: Step 1 results have been shared and Step 2
may proceed. Cache the source train/dev union for F1–F3, using canonical readers;
no test/eval or external recordings are read. Each training recording has the
centred view and two fixed random views; development has the centred view only.
All views use one native physical interval up to 10 s, independently resampled
to 16/48 kHz. Crop seed 1234 and canonical domain/split/ID identify the fixed
random offsets independently of loader workers and future fitting seeds. Identical
crop intervals share stored rows, retaining their logical view aliases. This fixed
crop diagnostic differs from the primary continuous random-crop training policy.

Store FP32 validity-mask-pooled projected frontend plus twelve Transformer
outputs, consistent with WavLM-WA's 13-representation definition; Hugging Face
`hidden_states[0]` is not substituted for the frontend. Store the raw 512-D CLAP
audio projection before its API's internal normalization. The encoders remain
foundation-pretrained, frozen/eval, BF16 with sensitive calculations FP32; no
completed fold's adapter, classifier or learned layer weights enter this cache.
Native WavLM inputs are never padded to form a mixed-length encoder batch.
Bounded source parity/speed checks precede extraction. Initially, 15 October,
batch-one arithmetic was retained when batching changed BF16 feature values.
Later owner speed request, 16 October: accelerate this new diagnostic cache
through equal-length WavLM batches using the existing full forward and independent
CLAP batches of fixed-size processed mel inputs. Retain BF16/FP32 calculations
and measure/report rounding differences; do not require bit-exact batch-one values
for the accelerated segment. Existing primary/checkpoint scorers and Step 1 retain
their original numerical policies. No deeper split of the WavLM forward is used.

Preserve completed cache shards and their original config. An explicit
`--continue-cache` verifies completed manifest chains and interrupted-split
payload hashes, canonical identities and fixed-crop aliases before extracting
missing records only. The accelerated segment has separate code/benchmark/batch
provenance; each new shard binds its execution record. Storage grouping-window
size can change without changing physical crops, examples or encoder architecture.
Do not describe the resulting mixed execution history as globally bit-exact to
batch-one inference. This feature extraction continuation is distinct from
model-training resume, which remains unimplemented. Local cache completion
requires every canonical record and logical view, finite arrays and shard hashes.
The cache is extracted once and reused across all fitting seeds/configurations.

Actual CLAP `repeatpad` repeats complete copies of a short waveform, then
zero-pads the remainder to 480,000 samples. Its configured `rand_trunc` is not
invoked for common physical views already at most 10 s. Do not replace this
with pure zero-padding or change the existing trained input recipe.

The owner approved reusing the existing settings for Steps 3–4: AdamW LR 0.0003,
weight decay 0.0001, constant LR, batch 132, maximum 30 epochs/patience 5, and
the existing 64→16 MLP hidden layers. Initialize trainable components freshly
for every inner split and seed 1234/1235/1236. Select checkpoints strictly by
inner-held source-dev EER (ties retain earlier), report EER/AUROC and equal-domain
macro summaries, and keep the original domain/class-balanced utterance sampling.
Share Step 2 results before moving to fitting; unresolved consequential probe or
ablation specifications must be settled before the affected configuration runs.

### Step 3 source-only probes authorized 18 October

Step 2 completed and its results were shared before the owner requested Step 3.
Cover F1–F3 only. For each outer fold, rotate the third source domain as the
inner validation domain: fit on the other two official source train partitions,
select on the held-in-source official development partition. Other development
partitions and the outer excluded domain do not select that fit. F4 and external
holds remain. There is no new target scoring or use of prior target metrics.

Four inputs each receive logistic and small-MLP heads: raw pooled last-layer
WavLM (768); learned softmax mixture of all 13 pooled WavLM representations (768);
CLAP's L2-normalized projected embedding (512); and weighted WavLM concatenated
with normalized CLAP (1280). This uses native WavLM pooled features without an
anchor adapter or new feature standardization. CLAP normalization follows the
existing model's embedding policy. Thirteen mixing scalars start at zero/uniform
and train jointly with the head for weighted/concat probes; therefore their
logistic head is linear in a learned mixture rather than a fixed layer vector.
Pooling before scalar layer mixing is equivalent to mixing before masked pooling
because all layers share the same validity mask. Cache R0 remains the projected
frontend, not Hugging Face hidden_states[0].

The logistic head is one linear layer. The MLP reuses the existing pre-linear
LayerNorm, 64→16 hidden dimensions, GELU/dropout0.2 and scalar spoof-logit order.
Use detection BCE only: these probes do not produce anchor probabilities, so
semantic-group/BF-JS losses are inapplicable. The primary model and its losses
are unchanged. Every fit uses newly initialized heads/mixing weights, never a
completed LACF/WavLM checkpoint or timing-trained state. No encoder is loaded.

Retain AdamW LR0.0003, weight decay0.0001, constant LR, physical/effective batch132,
accumulation1, BF16 operations with FP32 mixing/normalization/loss, maximum30epochs
and patience5. Strictly lower inner source-dev EER selects; ties keep the earlier
checkpoint. Source train sampling uses the existing inverse domain/class bucket
weights and replacement, with epoch draws equal to the number of utterances in
the two fit domains. Select one of the three logical fixed training views uniformly
per sampled utterance; identical crop aliases do not alter probability or triple
epoch visits. Validation uses the centered view only. This remains a fixed-view
diagnostic study, distinct from the primary continuous-random-crop training.

Verify canonical IDs/labels, complete manifest chains, all payload SHA256 and
array dimensions once when loading CPU-resident source features; vectorized
batch gathers avoid per-example disk/audio reads. The shared source union is
reused, but each fit routes only its two train domains and one inner dev domain.
Fresh selected-checkpoint reload must reproduce source logits/metrics exactly;
shared CSV writing/reading checks IDs/labels and spoof direction before completion.
Existing run directories are refused, and a failure stops the sequential queue.
No model-training resume or automatic failure restart is implemented.

There are 216 fits: 3 outer folds × 3 inner rotations × 3 seeds × 4 inputs ×
2 heads. For each outer/input/head/seed, average the three inner-domain EERs
and AUROCs equally; then report mean and sample SD (ddof1) across seeds1234/1235/1236.
These are inner source diagnostics, not target results or confidence intervals.
Bounded timing uses eight warmup and64 measured updates per fixed input/head
plus source-dev scoring/checkpoint round trips; timing states are preserved as
timing-only and discarded for final initialization. They are not completed
scientific recipe candidates. No optimizer/batch/architecture search is performed.
Share Step 3 results and stop before Step 4.

The existing eight anchors comprise three bona-fide and five spoof concepts.
Step 1 therefore reports both masses, using uniform references 3/8 and 5/8,
respectively; no concepts, group indices, templates, temperature, losses or
normalization are changed to fit the advisory message's 3/8 spoof-mass wording.
JS divergence is reported in natural-log units (nats). Source summaries include
per-domain values and explicitly distinguished pooled/equal-domain summaries.
Native batch-one BF16/FP32 inference is retained, and all recomputed classifier
logits must match preserved score CSVs before diagnostic completion. Partial or
failed diagnostics are preserved; no automatic retries or training resume.

## Owner-authorized native LACF F3 evaluation (2026-10-14)

The owner authorizes native P1 evaluation of the completed seed-1234 F3 LACF
checkpoint on ASVspoof 5 Track 1 eval. Training finished at epoch 13; source-only
selection chose epoch 8, macro source-dev EER 8.258790%, and selected-checkpoint
reload passed. This supersedes earlier F3 target holds only. F4 remains held;
no new training, FT4, ablations, duration controls or P2–P4 are authorized.

Use 32 data-loader workers, the available CPU count, prefetch 2 and the unchanged
batch-one native scorer: a centered common physical interval up to 10 seconds,
16/48 kHz views, frozen encoders, BF16 mixed precision with FP32 sensitive
calculations, scalar spoof logit and source-only calibration. All 680,774
canonical ASVspoof 5 eval recordings must be scored without skipping or changing
input duration to recover failures. Worker count does not authorize larger
encoder batches or numerical/input-policy changes.

Before target access, reproduce complete source CSVs/EER from the existing
verified feature caches and run bounded direct-audio/cache/exported-score parity
using the selected checkpoint and 32 workers. Reuse the existing deterministic
sample policy; a source-only certificate records actual checked counts, hashes,
calibration and validation scope. No repeated full source-audio encode or
padding-sensitivity experiment is required. Preserve F1/F2 certificates and their
writer bytes. Target scores/metrics are written only after the production gate
passes; report both raw and transductive branches and the existing 1,000
conditional bootstrap resamples (seed 2026).

Run records use the existing local evaluation/preparation folders, with F3
launch/evaluation logs under `docs/local/lacf_native_eval_20261012/` and outputs
under `outputs/lacf10s_optimized_20261009/f3/1234/`. The pipeline runs in tmux,
stops on failures, preserves partial work, does not retry automatically and
never advances to F4. Target-copy destination content checksums remain omitted
under the existing owner decision; do not describe that as a passed destination
checksum scan. Evaluation launch is not evaluation completion.

## Owner-authorized LACF F3 launch and loading update (2026-10-13)

Start fresh native 10 s, seed-1234 F3 only; F4 remains held. This supersedes
older F3 scheduling holds, not target-isolation rules. Preserve the primary
architecture, frozen encoders, 661,479 trainable parameters, BF16/FP32 policy,
physical/effective batch 132, accumulation 1, AdamW LR 0.0003/weight decay
0.0001, constant LR, maximum 30 epochs/patience 5 and source-only macro EER
selection/reload. No resume, target evaluation or model revision is authorized.

Bounded F3 source-only forward/backward/optimizer probes compared 2, 16 and 32
workers. Longer 120-step probes discarded 64 warmup steps: 16 workers measured
111.67 recordings/s and 32 measured 116.63; both had less than 0.1% loader wait.
Select 16 training workers, retain 4 development workers and prefetch 2. The
small throughput difference does not justify doubling worker count. These are
short operational measurements, not full-epoch speed guarantees or complete
scientific recipe candidates. Selection uses no target results.

F1/F2 used 2 training workers; changing workers can change random-crop RNG
assignment, so do not claim identical trajectories across loader settings.
The sampler, crop distribution, scientific settings and input/scoring policy
remain unchanged. Reuse the verified centered source-development features;
random training crops continue through both frozen encoders. Review the new
evaluation-only provenance entry separately while preserving all original
training-file hashes and the prior freeze. Local measurements, reviewed freeze
and launch evidence live in `docs/local/lacf_f3_launch_20261013/`. No repeated
full-cache scan or multi-hour encoder validation is implied by that metadata
addition. F3's selected checkpoint still requires full-source reload verification
before its completion marker; no automatic F4 advancement is authorized.

## Planned LACF F2 boundary and preliminary comparison (2026-10-12)

Latest owner clarification: this is a shared plan, not an instruction to implement
the exporter or start evaluation now. The owner will arrange the WavLM artifact
copy from SSH1. The queue advancement hold was already applied; F2 continues
normally and F3/F4 remain held. Actual evaluation execution needs a later owner
instruction. The proposed independent-run policy below describes the plan,
not current authorization to open targets.

Finish the current optimized native-10 s, seed-1234 LACF F2 run using its
unchanged source-only selection, patience and selected-checkpoint reload.
Hold automatic advancement to F3/F4. After F2 completes and both runs are
verified, the proposed next step is native P1 evaluation of the frozen LACF F1/F2
checkpoints before the wider comparison is complete. Compare matching completed
WavLM batch-96, seed-1234 folds: F1 on SpeechFake test and F2 on CFAD clean unseen
test. If subsequently instructed, this would revise the earlier target-access
timing for these two completed-run evaluations only. It does not authorize FT4,
ablations, external tests, duration controls or further training changes.

Keep canonical membership, spoof-score direction, source-only raw and normalized
threshold calibration, prespecified transductive reporting and confidence
intervals. Native LACF uses the centered common interval of at most 10 s fixed
below; native WavLM retains full-utterance development/test preprocessing.
Disclose those input and encoder-training differences: this compares complete
systems, not an isolated test of freezing or duration. Do not choose evaluation
crops/settings using test scores or use historical WavLM F1 numbers as a
substitute for the matching current baseline runs.

Freeze each evaluated checkpoint/configuration and record independent-run
evaluation authorization without claiming a comparison-wide freeze. Native
LACF score export still requires implementation and synthetic/source validation
before target inference; authorization alone does not establish readiness.
Report separate fold results and identify any F1/F2 average as preliminary,
not a completed four-fold result. Test feedback may describe the frozen models;
changes prompted by it are exploratory, cannot silently replace principal rows,
and require independent confirmatory data for new generalization claims.

## Owner-approved independent-fold evaluation (2026-10-06)

The owner explicitly authorizes evaluating completed WavLM batch-96 F1/F2/F3
independently now, then adding F4 after it is trained. This supersedes the
comparison-wide target-access barrier below for these completed-run evaluations;
it does not claim that the other models' recipes or LACF are already frozen.
Use `--confirm-run-frozen`, record the independent-run policy in reports, and
leave `comparison_wide_freeze_confirmed_by_operator=false`. The legacy
`--confirm-protocol-frozen` remains available only when that wider freeze has
actually been completed.

Retain completed-checkpoint/configuration verification, exact source-dev
membership, source macro-EER reload checks and source-only threshold calibration
before opening the target. Target results must not tune WavLM, LACF, other
baselines, prompts, recipes or ablations. Later target-informed changes are
exploratory, not replacements for principal comparison rows. Disclose this
evaluation timing; do not retroactively claim a comparison-wide pre-target
freeze. Future model evaluation decisions must retain these safeguards.

Current launch is native P1 only, F1→F2→F3 at seed 1234: batch-one BF16,
8 workers, prefetch 2, and 1,000 bootstrap resamples (seed 2026). The short
source-only benchmark found under 0.4% loader waiting, so increasing workers is
not supported by evidence. The initial 8 GiB allocator ceiling caused an OOM
on a 124-second full recording despite free physical GPU memory. The owner's
2026-10-06 follow-up removes that ceiling: evaluation now uses available GPU
memory without a configured allocator limit. This is an execution-memory fix,
not a target-performance-driven model or input change. Shared GPU use can still
cause a genuine OOM; any scoring failure stops the queue without cropping,
skipping audio or changing scores.

Interrupted CSV exports are now append-only and have checkpoint-, ordered
membership-, model-code- and input-policy-bound recovery journals. Completed
source CSVs retain their existing manifest checksums. An exception flushes and
registers the valid prefix; the next launch verifies it and scores only the
remaining records. Changed hashes/identities or uncommitted tails after a forced
kill fail closed and are preserved for inspection. The original F1 temporary
export predates journals: its 163,299 ordered rows were checked against canonical
SpeechFake membership and explicitly adopted from the known failed launch.
This one-time adoption records the existing file hash/count, not a claim that
the old exporter had already registered a historical partial-file checksum.

Each fold retains separate immutable `metrics.json`, registered score CSVs,
checkpoint identity and `evaluation.log`. After F1/F2/F3, write
`partial_summary_f1_f2_f3.json`, with `publication_summary=false`, missing F4,
no four-fold mean and no across-seed SD. Once F4 finishes, score F4 only using
the same settings and aggregate existing F1–F4 reports into `summary.json`.
Do not rerun, overwrite or pool earlier target datasets. Partial and complete
summaries are distinct artifacts. P2 external prerequisites and P3/P4 analyses
remain separate from this native P1 queue.

Status updated 2026-10-02: fairness safeguards strengthened with project-owner approval before recorded target evaluation. The fresh batch-64 WavLM F1 seed-1234 run finished through early stopping at epoch 17; its best source-selected checkpoint is epoch 12. Earlier interrupted batch-96/128 runs remain pilots, not valid continuations of the batch-64 recipe. No target score CSV, target metrics report, or score manifest was found in project outputs at this update. The latest explicit project decision takes precedence over older planning documents.

Latest explicit batch decision on 2026-10-02: use **batch 96** for the new WavLM study family `wavlm_bs96`, keeping optimizer, learning rates, scheduler, weights, patience, precision, and loader settings unchanged. One synthetic full-length batch-96 forward/backward/Adam step passed with finite loss and 27.78 GiB peak reserved CUDA memory. This establishes memory feasibility, not target performance or guaranteed GPU saturation. Queue F2, F3, F4 sequentially at seed 1234. Preserve the completed batch-64 F1 under `outputs/wavlm/f1/1234/` as a separate source-only recipe/pilot record. **F1 must be retrained at batch 96 before the four-fold batch-96 publication summary**; do not mix its old checkpoint into that family. Changing batch before any recorded target result is a declared project decision; no target informed it.

Latest explicit seed decision, 2026-10-05: due to training time, **WavLM uses seed 1234 only** across the four batch-96 folds. Seeds 2345/3456 will not be trained for this model. Report this compute-budget limitation explicitly: no across-seed standard deviation or claim of seed robustness is available. Retain the shared target metrics and fixed-model target-bootstrap intervals, which do not estimate training randomness. Other models retain the three-seed plan unless the owner changes them separately. This decision does not waive the comparison-wide target freeze, complete an interrupted fold, or authorize mixing batch-64 and batch-96 checkpoints.

**Owner AASIST seed decision, 2026-10-06:** family `aasist` uses **1234 only**
across all four folds; do not schedule 2345/3456 for AASIST. This supersedes its
earlier three-seed plan while preserving the architecture and training recipe.
A single-seed study is valid with explicit disclosure, but does not reproduce
the AASIST paper's three-run average or measure seed stability. Report point
estimates and conditional target-bootstrap intervals; across-seed SD is
unavailable (`null`). Other models' seed plans remain unchanged unless separately
changed by the owner. This decision authorizes no GPU pilot or training launch
and does not waive source-only selection or the comparison-wide evaluation gate.

## Research question and comparison set

Measure how detectors trained on three source datasets transfer to a fourth, unseen dataset. The main comparison set is WavLM-WA, AASIST, multi-level SSL feature gating, SAM-AASIST, LHCC, and LACF. SAM-AASIST names the optimizer and its underlying network; SAM alone is not a model. LACF-10s is the main proposed method. A separate matched-four-second evaluation compares WavLM-WA and LACF-4s; native WavLM-WA full-utterance evaluation is not a four-second test-time control. RawNet2 and other candidates belong in an extension table only if they are subsequently given the same protocol.

All trainable methods use the same source-domain membership, held-out targets, label convention, score orientation, checkpoint criterion, and final evaluator. Each method retains the waveform duration, architecture, and preprocessing needed to reproduce its method. Batch size and learning rate are model-specific because the models have different memory footprints, optimizers, and pretrained components; the numerical recipe is fixed for each model across all folds and seeds before target results are inspected.

## Data and target isolation

Use the official dataset-specific readers in `src/audio_deepfake_detection/data.py`. Raw datasets under `/mnt/salt/datasets/audio-deepfake` are read-only. The disposable `/mnt/drive/audio-deepfake-cache` mirrors protocol-selected original audio and must preserve the readers' membership and file formats. Never use a fixed cached crop or cached target labels for training.

| Fold | Source train and development domains | Final held-out target |
| --- | --- | --- |
| F1 | ASVspoof 2019 LA, ASVspoof 5 Track 1, CFAD clean | SpeechFake `test_all` |
| F2 | ASVspoof 2019 LA, ASVspoof 5 Track 1, SpeechFake | CFAD clean unseen test; seen test reported separately |
| F3 | ASVspoof 2019 LA, CFAD clean, SpeechFake | ASVspoof 5 Track 1 evaluation |
| F4 | ASVspoof 5 Track 1, CFAD clean, SpeechFake | ASVspoof 2019 LA evaluation |

Use only each source's official train split for optimization and official development split for model selection and threshold calibration. ASVspoof 5 development data are excluded entirely in F3. Do not use any target train, development, or evaluation label to choose a model, seed, prompt, loss weight, hyperparameter, checkpoint, or threshold. Target evaluation data are opened only after the run's checkpoint and configuration are frozen. External ASVspoof 2021 DF, PartialSpoof, and MLAAD/M-AILABS tests follow after the four-fold study is fixed.

Labels are `0 = bona fide` and `1 = spoof`. Every model emits one raw score for which **larger means more spoof-like**. Two-logit models return `spoof_logit - bona_fide_logit`; one-logit models return their spoof logit. A sample is predicted spoof when `score >= threshold`.

Target isolation applies across methods: target scores, metrics, errors, listening examples, or unlabeled distributions from one detector must not guide another detector's architecture, recipe, prompts, or ablations. The sole authorized use of unlabeled target score statistics is the frozen transductive evaluation branch. The 2026-10-06 owner-approved completed-fold policy above supersedes the comparison-wide precondition for the authorized WavLM runs, retaining target isolation. Official source train/dev use in another prespecified fold remains permitted; it does not authorize adapting recipes to held-out evaluation results.

## Training and selection rules shared by all models

- Run source-only feasibility pilots under the comparable tuning-effort policy below, then freeze each recipe before its final fold/seed runs. A pilot may establish batch feasibility and complete source-development evaluation. Record each adaptation and its source-only reason. Never revise recipes in response to any method's target results.
- Final runs use seeds **1234, 2345, and 3456**, except the owner's declared compute-budget exceptions: **1234 only** for families `wavlm_bs96`, `aasist` and the initial `lacf`/`lacf4s` studies. Disclose unequal replication when comparing models; do not treat their target-bootstrap intervals as across-seed uncertainty. Seed Python, NumPy, PyTorch, CUDA, sampler, and DataLoader workers. Record library/CUDA versions, accelerator, code revision, and complete run configuration. Deterministic algorithms are best effort; record any nondeterministic operation rather than silently changing model behavior.
- One epoch makes as many replacement draws as there are source-training records in that fold. For the standard baseline sampler, choose a source domain uniformly, then an utterance uniformly within that domain; retain each domain's natural class ratio. Use the same sampler for all standard baselines unless the reproduced method requires a different one. LACF uses a declared domain-and-class-balanced sampler: choose domain uniformly, class uniformly, then utterance uniformly. Its A0–A10 ablations use this same sampler. Report the sampling difference when comparing LACF with a standard baseline.
- Use the model-specific batch and optimizer recipe below. Batch means examples per optimizer device; effective batch is batch times gradient accumulation. For accumulation, divide each microbatch loss by the actual number of microbatches in that optimizer step. Do not silently change batch, accumulation, loss weights, learning-rate schedule, input duration, or augmentation between folds.
- For WavLM on the 32-core H200-35C host, use 16 training workers, 4 workers per source-development loader, prefetch factor 2 per worker, pinned host memory, and persistent workers. These settings followed an early, pre-checkpoint F1 loader pilot that showed I/O wait with 8 workers. Other models start with source-only throughput pilots. Evaluation batch size is 1 for full-utterance or variable-length input. Fixed-length models may use a larger evaluation batch only after verifying that every per-utterance score matches the batch-1 result within a documented numerical tolerance. Loader throughput changes do not alter the sampler or model recipe.
- Evaluate each source development domain separately after every epoch. Select the checkpoint with the lowest **unweighted mean of the three source-development EERs**. Improvement requires a strictly lower macro EER; ties retain the earlier checkpoint. Early stopping uses only that macro EER. Save both best and last checkpoints. Never select on pooled source EER or target EER.
- Use mixed precision only where specified and verified for the model. Record the precision and any allocator setting. A CUDA out-of-memory error is a failed pilot, not a reason to change only one fold's recipe.

## Prespecified model recipes

Published-method starting values below were checked against the five supplied papers and, where available, the authors' released configurations; these take precedence over the earlier `SOTA_Model_details.pdf` summary. LACF starting values come from the supplied implementation plan and updated pipeline. Values marked **study choice** were not established by the original authors and must be fixed after a source-only feasibility pilot. This table is a prespecified starting recipe, not a claim that every value was reported by the authors. Batch 128 is not the main-study default for any of these baselines; it may only be evaluated as a separately named batch-size sensitivity experiment.

| Model | Train input; evaluation input | Batch × accumulation | Optimizer and initial LR | Schedule; maximum epochs; early-stop patience | Loss and other fixed settings |
| --- | --- | --- | --- | --- | --- |
| WavLM-WA | random 4 s at 16 kHz; full utterance | **96 × 1 (latest project decision; paper used 32; earlier batch-64 F1 kept separately)** | Adam; WavLM `2e-5`, layer weights/classifier `5e-3` | exponential `gamma=0.95`; 100; 5 (**study choice; paper patience 50**) | weighted CE for `[bona fide, spoof] = [9, 1]`; BF16; 13 learned representation weights. Current script omits the paper's MUSAN/RIR and codec augmentation, initial-weight regularization, and layer-wise LR decay; do not call it an exact reproduction until these are implemented and verified. |
| AASIST | random 64,600-sample train crop after 16 kHz resampling; first 64,600 samples for dev/test; repeat short audio | **24 × 1** | Adam `1e-4`, weight decay `1e-4`; FP32 | per-update cosine to `5e-6`; 100; 5 (**study-choice patience**) | weighted CE, reorder original class weights to `[0.9, 0.1]` and logits to bona-fide/spoof order; official architecture. Frequency masking off per released config; no RawBoost; no SWA under shared single-best source-only checkpoint policy. |
| SSL feature gating | full utterance at 16 kHz, dynamic right-zero-padding in training; unpadded full utterance in dev/test | **5 × 1** | Adam `3e-6`, weight decay `1e-4`; FP32, full frontend fine-tuning | constant LR per released optimizer; 10; 3 | XLS-R 300M, 25 states, 128-D SwiGLU gating, four MultiConv blocks, four-head pooling; weighted CE `[0.9, 0.1]` plus differentiable CKA, coefficient 1. No RawBoost per owner cost/fidelity direction on 2026-10-18 and released YAML. Explicit layout/CKA repairs; see the dated adaptation below. |
| SAM-AASIST | AASIST's 64,600-sample policy | **24 × 1** | SAM with Adam base optimizer at `1e-4` | cosine to `5e-6`; 100; 5 (**study-choice patience**) | `rho=0.05`, weight decay `1e-4`, RawBoost; weighted CE `[0.9, 0.1]`. Released config includes an In-The-Wild validation split: replace it with source-only development splits in every LODO fold. |
| LHCC | exactly 64,600 samples at 16 kHz, truncated/repeated | **8 × 1** | Adam `1e-6`, weight decay `1e-4` | schedule not reported; 10; 2 | XLS-R 300M low/high-level features, AFM, DCM, weighted BCE and RawBoost. Numeric class weights are not reported. Paper uses bona fide=1/spoof=0 and `1 - p_bona_fide` as spoof score; map labels and scores explicitly. |
| LACF-10s frozen | one common physical interval, at most 10 s; resample it separately to WavLM 16 kHz and CLAP 48 kHz | configurable; default 4 × 8, effective 32 pending source-only pilot | AdamW `3e-4` for adapter/classifier, weight decay `1e-4`; no automatic LR scaling | constant LR (owner-approved); 30; 5 | BF16 mixed precision with FP32 sensitive calculations; 661,479 trainable adapter/head parameters; seed 1234 only; `BCEWithLogits + 0.25 semantic BCE + 0.10 bona-fide JS`; freeze WavLM and CLAP; eight fixed anchors and temperature `0.07`. |
| LACF-4s frozen | same 4 s physical interval at 16/48 kHz | configurable, same effective batch as LACF-10s (old 8 × 4 assumption superseded) | same as LACF-10s | same as LACF-10s | same as LACF-10s; only input duration and feasible microbatch differ. |

For optional LACF-FT4, initialize from the corresponding frozen model, unfreeze only the top four WavLM Transformer blocks at `1e-5`, keep CLAP frozen, and retain the adapter/classifier LR unless a source-only pilot establishes a different fixed value. Report frozen and FT4 results separately. For optional SAM variants other than SAM-AASIST, create a separate named recipe and result row.

AASIST implementation status, 2026-10-04: the unmodified authors' network at commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1` is vendored with its MIT license, using random initialization rather than released trained weights. The LODO adapter corrects the upstream exact-length random-crop error, includes the last valid crop start, resamples non-16 kHz sources, and retains all epoch draws (`drop_last=False` rather than the released train loader's `True`). The upstream trainer's SWA stage and development-triggered target evaluation are not used; select the single lowest macro source-dev EER checkpoint as required above. These are disclosed study adaptations, not an exact published-run reproduction. Batch-24 CPU forward/backward/Adam feasibility passed with finite loss/gradients; CUDA memory/throughput, real-source development scoring/reload and source-only recipe validation remain pending while WavLM occupies the GPU. Loader settings start at train workers 8, dev workers 4, prefetch 2, evaluation batch 1. No target evaluation or comparison-wide freeze completion is implied by implementation readiness.

Model-specific preprocessing is preserved in the main comparison. Report training, development, and test duration separately. The matched-four-second evaluation and LACF-4s versus LACF-10s duration ablation are separate prespecified tables under the safeguards below. No target-dependent crop choice is allowed.

The SSL-gating full-input/RawBoost discrepancy is resolved in the 2026-10-18 adaptation below. Before implementing LHCC, resolve the paper's apparent conflict between a stated frozen XLS-R backbone and its end-to-end optimization algorithm; record the decision and treat any alternative as an ablation. For WavLM-WA, either implement the published augmentations and regularization for the primary baseline, or label the current no-augmentation implementation as a simplified baseline and avoid claiming a faithful reproduction. Keep these decisions source-only and recorded before inspecting target results.

Paper-specific reading notes and source links are in `SOTA_PAPER_NOTES.md`.

## Fair-comparison safeguards fixed on 2026-10-02

### 1. Match physical evaluation evidence for duration claims

Keep native input policies in the main table. Label the current WavLM row **4 s train / full-utterance dev and test**. For planned native LACF evaluation, fix a deterministic centered interval of at most 10 s for LACF-10s and at most 4 s for LACF-4s, on both source development and target. The WavLM and CLAP branches always receive the same interval before separate 16/48 kHz resampling.

Prespecify an additional **matched 4 s test-time evidence** comparison:

- Score WavLM-WA and LACF-4s on the same centered physical interval of every utterance. On the original decoded mono waveform of length `N` at rate `r`, select `L=min(N,4*r)` samples starting at `floor((N-L)/2)`, before resampling. Retain the original utterance ID. Crop selection never depends on labels, scores, model, or attack type.
- Keep all available signal for clips shorter than four seconds. Disclose encoder-required padding/repetition; it supplies no new observed audio. Do not add multiple-crop selection or sliding-window aggregation to this controlled arm.
- Apply the same interval policy to source-development and target scoring. Derive new raw and normalized thresholds from the controlled source-development scores; full-utterance thresholds cannot be reused for cropped targets.
- Use each method's frozen source-selected checkpoint and record its SHA-256 and original selection policy. Verify WavLM's native source EER against its completion record first. The controlled source EER need not equal the full-utterance checkpoint-selection EER; that difference is not a reload error.
- Store controlled scores/metrics separately under the existing output convention, using distinct result identities such as `wavlm_eval4s` and `lacf4s_eval4s`. Record duration, crop rule, training recipe, and checkpoint-selection policy. Never mix native and controlled rows in the same fold/seed summary.

This controls final scoring evidence. Encoder pretraining, optimizer, and native checkpoint selection can still differ and must be disclosed. A fully matched training/development/test-duration claim would require a separately prespecified WavLM run selected on four-second source development; do not relabel the current run that way. For LACF-4s versus LACF-10s, retain the same recipe except duration and necessary microbatch/accumulation, with effective batch unchanged.

The controlled exporter is planned work; the current WavLM evaluator still implements native full-utterance scoring. This policy change does not claim that controlled scoring is implemented.

### 2. Make tuning opportunity and actual effort comparable

Start from the paper and official implementation. Tag each recipe value `PAPER-EXACT`, `OFFICIAL-CODE`, `SOURCE-ONLY-ADAPTATION`, or `PROJECT-DECISION`; record paper/code conflicts and label simplified reproductions explicitly.

Allow each principal method at most **three complete source-only candidate recipes**, including its starting recipe. Declare candidates and the intended search before executing them. Use F1 source train/dev and pilot seed 1234 for recipe selection; compare the candidates' best checkpoints using source-development macro EER, with ties retaining the earlier candidate. Reuse the selected recipe across every fold and seed. A validated published baseline can retain its starting recipe without spending all three trials; LACF receives no extra undisclosed search allowance.

Log correctness repairs and loader checks separately; they cannot disguise additional scientific candidates. Numerical LACF adapter/loss choices explored during pilots count toward its budget. Prespecified A0–A10 ablations test hypotheses and are not a pool from which to retrospectively choose the principal method.

Record all candidates, adaptation reasons, pilot fold/seed, selected epoch and source EER, failed attempts, development-evaluation counts, optimizer updates, GPU hours, peak memory, and training/evaluation time. Report actual effort when fewer trials are used. Comparable opportunity does not require equal batch sizes, epochs, or GPU hours. Disclose pretrained encoders and total/trainable parameter counts separately.

### 3. Match sampling and augmentation for mechanism claims

The main table preserves the declared method recipes. Standard baselines use domain-balanced sampling with natural within-domain class ratios; LACF and A0–A10 use domain-and-class-balanced sampling. Disclose sampler, augmentation, class/loss weights, input durations, and encoder freeze/fine-tune policies. Main-table gains concern the whole system and recipe; they do not alone isolate relation features.

For LACF mechanism ablations, hold source membership, sampler, crop policy, seeds, effective batch, augmentation, applicable encoder checkpoints/freeze policy, checkpoint criterion, and evaluator fixed. Use LACF's WavLM Base+ for its speech-view controls; the native WavLM Base baseline does not substitute for that architecture control. Match adapter/head capacity where feasible and disclose necessary differences. Frozen LACF starts with no augmentation; if a logged source-only candidate changes this before the global freeze, its selected augmentation applies to all applicable mechanism controls.

| ID | Prespecified control | Tested change |
| --- | --- | --- |
| A0 | LACF WavLM encoder with binary head | Speech-view reference under the LACF recipe |
| A1 | LACF CLAP audio encoder with binary head | Audio-language-view reference |
| A2 | Concatenated WavLM/CLAP features with binary head | Ordinary fusion versus language-anchor relations |
| A3 | Earlier three-branch proposal | Previous fusion design; disclose required architectural differences |
| A4 | CLAP anchor probabilities only | Language anchors without WavLM |
| A5 | WavLM anchor probabilities only | WavLM projection without cross-view relations |
| A6 | Both anchor distributions | Common anchor space before relation features |
| A7 | A6 plus absolute difference and elementwise agreement | Agreement/disagreement features |
| A8 | A7 plus JS divergence and both entropies | Full relation representation without bona-fide JS loss |
| A9 | A8 plus bona-fide JS loss | Proposed frozen LACF |
| A10 | A9 plus top-four WavLM fine-tuning | Optional fine-tuning contribution |

Keep the WavLM semantic loss at its common selected value for A5–A9 where defined; change only the tested component/loss. A10 intentionally changes encoder freezing. Controls without an applicable view/distribution omit undefined losses and document those omissions. Do not claim an architecture-only contrast if objective or capacity also necessarily differs.

### 4. Prevent target feedback from crossing model boundaries

Original 2026-10-02 precaution: before inspecting any method's first held-out target result, freeze the comparison-wide protocol, principal result rows, LACF architecture/prompt bank, A0–A10 hypotheses, duration-control rules, tuning budgets, and source-validated final recipes for all principal methods. The owner revised target-access timing for completed WavLM folds on 2026-10-06 and completed SOTA runs on 2026-10-12. The LACF F1/F2 comparison described above remains a proposal pending an execution instruction. All no-target-guided-tuning rules remain in force; the wider freeze is not represented as complete.

Freeze these eight concepts and their ordering:

1. `natural human speech`
2. `authentic human voice recording`
3. `naturally produced human voice`
4. `synthetic speech`
5. `text-to-speech generated voice`
6. `voice-converted speech`
7. `AI-cloned human voice`
8. `neural-vocoder generated speech`

The first three are bona-fide concepts and the remaining five are spoof concepts. Use exactly `{concept}`, `a recording of {concept}`, and `this audio contains {concept}` for every concept. Normalize each frozen CLAP text embedding, average the three for each concept, and L2-normalize its prototype. Target audio, scores, and errors cannot guide prompts or ordering. Initial temperature and loss coefficients remain in the recipe table; settle any permitted source-only numerical trials before target feedback.

Each final freeze record contains a date, code revision, tagged recipe, pilot evidence/search log, known reproduction limitations, and confirmation that no target results guided decisions. At this update, WavLM has a completed source-only run; remaining methods still need implementation and source validation. The comparison-wide gate is therefore **not yet complete**.

Once target feedback exists, the principal study stays fixed. A later design/recipe change prompted by it is exploratory, needs separately identified results, and requires independent confirmatory data for a new generalization claim. It cannot silently replace a frozen principal row. Unlabeled z-score transfer remains the sole prespecified target-statistic adaptation.

## Shared final evaluation

The model hands the evaluator an utterance ID, dataset, binary label, and raw spoof score. Source-development scores are saved per domain. Target scores are generated once from the frozen best checkpoint. All metrics below use identical code for every model.

1. **Selection:** source-domain EERs and macro source EER determine the checkpoint and early stopping as above.
2. **Ranking:** report held-out target EER and AUROC. EER is a descriptive target metric; its target-derived crossing point is never used as a deployable threshold.
3. **Strict raw transfer:** pool raw source-development scores, derive source APCER operating thresholds at 1%, 5%, and 10%, freeze them, then report target APCER, BPCER, and ACER at each threshold. With high scores indicating spoof, `APCER(t) = P(score < t | spoof)`, `BPCER(t) = P(score >= t | bona fide)`, and `ACER = (APCER + BPCER)/2`. For each requested rate `a`, sort pooled source spoof scores ascending and select the element at zero-based index `floor(a * n)` as the threshold; this yields source APCER no greater than `a` under the stated `>=` decision rule, including ties.
4. **Unlabeled z-score transfer:** normalize each source-development domain independently using its own score mean and population standard deviation (`ddof=0`), pool normalized source scores, and derive the same three APCER thresholds. Normalize target raw scores with the **unlabeled target score distribution's** own mean and population standard deviation, then apply the frozen normalized thresholds. Reject a domain if its standard deviation is at most `1e-12` or any score is non-finite. This branch uses target score statistics and must be labeled **transductive**, not strict zero-shot. Report target APCER/BPCER/ACER for this branch; EER/AUROC need only be reported once because a positive-scale z-score preserves ranking.
5. **Uncertainty and aggregation:** for each fold and seed, report 95% class-stratified percentile bootstrap intervals for target EER, AUROC, and transferred APCER/BPCER/ACER, using 1,000 target resamples and bootstrap seed `2026`. Keep source-calibrated thresholds and the full target's unlabeled z-score statistics fixed inside the bootstrap. For three-seed models, report mean ± sample standard deviation for each fold. For single-seed WavLM, AASIST and initial LACF, report the seed-1234 point estimate, no across-seed standard deviation (`null` in summaries), and the per-run conditional target-bootstrap intervals. Report an equal-weight mean across four folds, retain per-fold results and disclose the replication difference. Do not pool targets of very different sizes into one headline score or label F1/F2/F3 alone as a complete four-fold result.

The `>=` tie convention, population standard deviation, degenerate-score check, threshold order statistic, and bootstrap design must be implemented once in the shared evaluator and tested with synthetic scores before final model comparisons.

## P2–P4 supplementary evaluation (implementation added 2026-10-05)

The main overall plan's P2–P4 apply to proposed methods and the baselines used to
support the corresponding claims. `scripts/evaluate_extended.py` provides one
shared report path; native score export currently supports the implemented
WavLM and AASIST only. Other detectors must export the same spoof-oriented CSV
contract with matching checkpoint/metadata provenance when implemented. These
analyses do not require new training or additional WavLM seeds. Implementation
and synthetic CPU checks are **not** completed real-data evaluation or permission
to bypass the comparison-wide freeze.

- **P2:** ASVspoof 2021 DF is the primary external test. Use official DF full
  evaluation keys, selecting `eval` and `notrim`, never progress/hidden trials.
  Before opening any P2 target, document the single deployment checkpoint's
  source-only fold/configuration selection reason in an immutable selection
  record bound to model, fold, seed, completion file and best-checkpoint hashes.
  Reuse that same record/checkpoint for DF, MLAAD/M-AILABS and PartialSpoof;
  do not choose the best fold on external results. The code does not invent a
  deployment-fold choice or waive the remaining models' seed plan. P2 raw
  threshold transfer is strict zero-shot; the separately labeled unlabeled
  z-score branch is transductive. PartialSpoof is an utterance-level stress
  test, not a segment-localization claim. MLAAD negatives are the unique genuine
  M-AILABS files referenced by official `original_file` metadata, not unrelated
  real audio. Disclose the available MLAAD release/languages, genuine-reference
  deduplication and class counts; do not claim the installed release is v10
  without confirming it. ASVspoof 2021 and PartialSpoof reuse earlier ASV source
  material, so these stress tests are not claims of wholly independent corpus
  provenance. No optional corpus is a hyperparameter-development set.
- **P3:** reuse F1 SpeechFake test scores for official `TTS`, `VC`, `NV` and
  per-model groups; reuse F3 ASV5 and F4 ASV2019 scores for per-attack groups.
  Each spoof-generator/attack row includes the full parent bona-fide reference,
  explicitly recorded with counts; do not pool these overlapping rows. Report
  actual attack IDs within their dataset namespace. A held-out dataset's IDs
  were not observed in that dataset during source training, but this does **not**
  establish cross-dataset generator-family novelty. Family-level known/unseen
  claims still require a verified cross-dataset mapping; the implementation
  marks this unknown rather than inferring it from matching/different ID names.
- **P4:** F2 CFAD clean/noisy/codec unseen tests are primary; seen tests are
  separate secondary reports. Read the distributed test waveforms without
  regenerating corruptions. Report condition, noise, SNR, noise×SNR and codec
  groups. F3 ASV5 reports C00–C11 codec groups; map official `-` to uncoded C00.
  Match only declared original IDs (ASV5 `CODEC_SEED`, CFAD checked filename
  convention and class/source directory) for descriptive strict-raw clean/error
  differences. Average multiple variants within each original before averaging
  originals, record matched class counts, and do not attach unpaired bootstrap
  intervals to paired deltas. A relation-stability LACF variant is later work,
  not silently included in the main detector.

All thresholds remain those of the completed native source-development run;
source macro EER and CSV hashes are verified before evaluation metadata is
opened. P3 and ASV5 P4 subsets retain the **full target partition's** unlabeled
z-score statistics, never per-generator/per-attack/per-codec recalibration.
CFAD's separately distributed clean/noise/codec test corpora each use their own
complete unlabeled partition statistics; their subgroups retain those parent
statistics. Fixed source thresholds and fixed parent target statistics are
retained in class-stratified 95% percentile bootstrap intervals (1,000 resamples,
seed 2026). Single-class subgroups report the defined class-conditional error
and its interval; EER, AUROC and ACER are `null`, not invented. These intervals
are conditional on the frozen model, do not estimate seed variability, and are
descriptive across many overlapping subgroup comparisons, not multiplicity-
corrected significance tests. Metric rates remain fractions; paired differences
may be negative and are not clamped.

Store supplementary artifacts under
`outputs/<family>/<fold>/<seed>/analyses/<p2|p3|p4>/<dataset>/`, separate from P1.
Bind score exports to canonical membership/labels, metadata, source calibration,
best checkpoint and completion hashes; refuse mismatched/replaced/unregistered
scores and existing report overwrites. P3/ASV5 P4 reuse verified P1 CSV bytes.
The default supplementary command does CPU reporting only; `--export-scores`
explicitly enables GPU inference at batch 1 using the model's unchanged native
preprocessing. All commands require operator `--confirm-protocol-frozen`;
this is an acknowledgment, **not an automated proof** of scientific freeze.
WavLM P1 also accepts the narrower owner-approved `--confirm-run-frozen`
acknowledgment described at the start of this document; it does not assert a
completed comparison-wide freeze.

Initial availability inspection (before the 2026-10-06 acquisition):
`/mnt/salt/datasets/audio-deepfake` exists (the spelling `datsets` does not).
CFAD robustness, SpeechFake generator annotations, ASV5 codec keys and
PartialSpoof utterance eval keys were present. ASV2021 archives contained
trial-ID lists and source mapping metadata, without full CM keys; genuine
M-AILABS audio was not found among the top-level corpora. No raw dataset,
active training run or existing result was changed by that implementation.

**2026-10-06 acquisition update:** the owner authorized a separate tmux download
queue. Official DF full keys passed organizer MD5 verification and were installed
at `asvspoof2021/extracted/keys/DF/CM/trial_metadata.txt` under the existing dataset
root. The current reader selects 533,928 `eval`/`notrim` entries (14,869 bona fide,
519,059 spoof); these are metadata counts, not experiment results. M-AILABS
acquisition is in progress, with intended genuine root `mailabs/extracted/`;
verify the local acquisition completion/reference-validation records before use.
The download helper discovered malformed quoting under the current default
CSV interpretation of installed MLAAD Amharic Edge-TTS metadata. Its path-prefix
inventory avoids transcript parsing; the shared evaluator remains unchanged and
requires a parser correction/validation before MLAAD evaluation. No checkpoint
selection rule, model recipe, evaluation membership, freeze gate or seed policy
was changed by acquisition, and no P2/P3/P4 evaluation was launched.

## Evidence, outputs, and release gate

Store each final run under `outputs/<model>/<fold>/<seed>/` with `training.log`, `best.pt`, `last.pt`, `config.json`, source-development score CSVs, target score CSV, and `metrics.json`. The configuration records all recipe values, the selected epoch, source macro EER, checkpoint hash, code revision, dataset/cache roots, seed, package versions, precision, and allocator setting. Score CSVs contain at least `utterance_id,dataset,label,raw_score`.

Before a model enters the final table, verify protocol membership and local-cache counts, F3 target isolation, score orientation, source sampler frequencies, one forward/backward step at its fixed batch, complete source-development scoring, checkpoint reload, and synthetic tests for all shared metrics and threshold branches. The model's assumed recipe values must be checked and frozen using source-only evidence. Preserve interrupted or failed pilot logs separately from final runs; do not present the saved epoch-1 WavLM F1 checkpoint as a finished fold result.

## Operational updates: AASIST launch and native target cache (2026-10-07)

The owner authorized AASIST F1 training, seed 1234, and a separate native P1
test-audio cache while dataset acquisition continues. The recorded recipe is
unchanged. A short real-source GPU feasibility and six-example reload check
passed before fresh training; full source-dev and selected-checkpoint validation
remain pending. This does not establish comparison-wide freeze completion or
authorize AASIST target evaluation.

The target-cache copier preserves whole original file bytes, relative IDs,
metadata and labels, with SHA-256 and canonical membership verification. Require
all requested partition completion records before using that cache. Copying
files does not change model-specific preprocessing, the native versus
matched-duration distinction, any scientific gate or the validity of earlier
scores. Do not rerun or overwrite valid folds merely to switch storage roots.
See README for commands and the code guide for the copying/validation flow.

## LACF primary implementation and fidelity (2026-10-08)

This initial implementation entry records the earlier FP32/three-seed choices.
The later owner-approved Part 1 review below supersedes precision, seed and
fixed-batch assumptions while preserving the architecture and input processors.

The primary frozen architecture/trainer is implemented under `lacf/` and
`scripts/lacf/`, separate from SOTA code. The actual main DOCX, including all
tables/equations, and relevant original supplementary plans were checked.
`Updated_Pipeline.pdf` resolves the classifier ordering to pre-linear LN(35)
and LN(64), with GELU/Dropout(0.2) after each hidden linear layer, then 16→1.
Eight concepts, three templates, individual-template normalization before
prototype averaging, last-state masked WavLM pooling, adapter, temperature,
relation ordering/dimensions and all three loss equations retain plan values.

The later macro source-dev EER decision supersedes the main plan's older pooled
selection wording. The existing centered source-dev interval, batch 4×8,
constant LR and three LACF seeds are preserved. Primary optimization includes
all epoch draws and divides accumulated loss by the actual number of
microbatches in each update, including the final incomplete group. Bona-fide
JS is computed within each microbatch, with zero for an all-spoof microbatch;
it is not replaced with a new effective-batch/class weighting rule.

Processor/padding and precision were underspecified in the main plan. After
this was flagged, the owner delegated those implementation choices. Use FP32
and the selected checkpoints' standard processors: WavLM Base+ does not
normalize waveform amplitudes, right-pads with zero and supplies a validity
mask; unfused CLAP repeat-pads short audio, with no fabricated CLAP padding mask.
The same physical interval is selected before independent resampling; CLAP
cannot select a second crop. Too-short clips fail rather than being skipped.
Processor settings/revisions and enabled feature/loss/fusion configuration are
recorded in every run. Frozen WavLM/CLAP stay eval/no-grad in training.

Sources for processor behavior are the [WavLM Base+ checkpoint settings](https://huggingface.co/microsoft/wavlm-base-plus/raw/main/preprocessor_config.json)
and [unfused CLAP checkpoint settings](https://huggingface.co/laion/clap-htsat-unfused/raw/main/preprocessor_config.json),
cross-checked with the installed Transformers 4.57.6 implementation. No CLAP
original paper is included in the supplied canonical reference set; its API
behavior was verified in primary checkpoint settings and installed source.

Explicit relation-group/loss configuration and a dimension-declaring fusion
interface prepare later ablations without implementing unspecified variants or
changing the loop. Defaults implement the primary model. FT4, ablation runs,
matched-duration execution and a native target exporter remain later work.
Shared score orientation, calibration, raw/transductive reporting and confidence
interval code are unchanged. Implementation tests are synthetic CPU checks,
not source-only recipe/GPU validation or target-access authorization. Training
requires verified cache completion, available separately acquired checkpoints,
an authorized source-only pilot and the existing freeze discipline.

## LACF Part 1 owner decisions and review (2026-10-08)

Initial `lacf` uses seed **1234 only** across F1–F4. This overrides the main
plan's three-seed reporting and the earlier local implementation. Summaries
retain per-fold point estimates and conditional target-bootstrap intervals,
with across-seed SD unavailable (`null`); no seed-stability claim is supported.

Both frozen encoders remain eval/no-grad. Only adapter and relation classifier
train: **657,920 + 3,559 = 661,479** parameters. Use BF16 autocast for training
and source development, with FP32 parameters/optimizer state. Explicit FP32
boundaries cover encoder/backend normalization, masked mean pooling, embedding
normalization, anchor cosine/softmax, logarithms, JS/entropy and all losses.
CLAP's projected features are cast before its internal L2 normalization.
No gradient scaler, precision fallback, resume or automatic LR scaling.

AdamW LR 0.0003, weight decay 0.0001, constant LR, 30 max epochs, patience 5
and strict equal-domain macro source-EER selection remain fixed. Ties keep the
earlier checkpoint. Physical batch, accumulation, train/dev workers and
prefetch are configurable and recorded. Existing 4×8 defaults are conservative
starting values, not an owner-fixed final batch. Effective 96/128 are candidate
examples, not prescriptions; choose and freeze actual settings with source-only
evidence, then keep them consistent across folds. Changing physical microbatch
can change the bona-fide JS averaging even at equal effective batch; record it.

The simple launcher runs fresh F1→F2→F3→F4 and advances only after verified
completion identity, checkpoint hash, recipe and recorded source-dev reload.
It refuses existing fold directories and duplicate queues, stops on any
training/verification failure and performs no target scoring. Synthetic CPU
BF16 and queue fixtures establish implementation behavior only. Part 2 must
resolve any correctness blocker, verify source-cache completion and local
checkpoint availability, and validate authorized source-only CUDA integration,
memory/throughput and reload behavior before starting final training.

## Owner checksum exception for initial LACF (2026-10-08)

Historical exception, superseded by the subsequent completion report; the current
trainer no longer accepts its expired acknowledgment flag. The owner explicitly
waived waiting for the ongoing transfer checksum job;
leave that job running. This overrides the checksum-completion launch barrier
for initial LACF only. It does not establish byte integrity. Use the separate
`--acknowledge-unverified-cache-copy` flag, never `--confirm-cache-verified` for
this case. Provenance records `cache_verified_by_operator=false`,
`unverified_cache_copy_acknowledged=true` and
`cache_copy_checksum_status=unverified_owner_waiver` in each run/checkpoint.
No cache-verification completion marker is created. Canonical source train/dev
membership/labels/counts, required file existence and decoding remain mandatory,
as do actual source-only GPU feasibility, full source-development scoring,
selected-checkpoint reload and a common frozen recipe before final training.
Any later transfer checksum mismatch must be reported immediately.

### Subsequent checksum completion report (2026-10-08)

The owner relayed SSH1's successful verification at 05:35 on 8 October:
1,243,408 manifest files (approximately 180 GB), no differences or failures,
and verification finished. This supersedes the pending-checksum exception for
future runs. Record its evidence source as owner-relayed SSH1 confirmation;
SSH2 did not independently rerun the transfer checksum scan and does not have
the original manifest/log. The canonical audio check on SSH2 independently
passed membership/counts and full decoding for 1,243,402 required audio files;
do not equate that count with all manifest entries or claim the six-entry
difference has been reconciled.

The latest owner instruction retains source-only GPU validation as a gate
before full training, superseding the intervening request for immediate launch.
Continue only the preserved source-validation pilot, without new optimizer
updates or final-training initialization from its weights. Complete source-dev
scoring and exact checkpoint-reload score/EER checks before freezing the recipe
and launching fresh F1–F4 runs with `--confirm-cache-verified`.

## Historical owner duration order and frozen-development optimization (2026-10-09)

The owner stopped the first 10 s F1 run after two completed epochs to optimize
execution/logging, then authorized LACF-4s before fresh LACF-10s, each F1→F4,
seed 1234 only. Preserve stopped checkpoints and logs; no resume or pilot-weight
initialization. The supplied SOTA details PDF is a duration-comparison reference,
not authority to replace original paper recipes. Both duration arms retain the
same effective batch and approved primary architecture/losses/optimizer/precision;
source-only operational feasibility determines loading settings. No target
performance or training data from a held-out fold may guide the recipe.

Only centered source-development frozen encoder outputs may be persistently
cached. Random training crops still execute both encoders every epoch. The cache
binds exact canonical records/labels, duration, processors, revisions, code,
precision and environment. Encoder/backend dev batches stay one; variable-length
WavLM group normalization prevents assuming batched scores equal batch-one.
Current adapter/head scores and EER are recomputed each epoch. Reload verifies
unchanged frozen states and full canonical cached source scores; bounded direct
full-audio/cache comparisons check encoder execution fidelity. No score, adapter
output or target cache is used to select settings.

The old source-only 10 s preflight passed, but its code hashes cannot authorize
modified code or a 4 s run. Require a new duration-specific longest-input memory,
finite update, full source-dev scoring/reload record before launching that arm.
All-source canonical decoding and the owner-relayed transfer-checksum result
remain valid evidence; do not repeat these scans each epoch or claim an
independent SSH2 checksum audit. Physical/effective batch 132, accumulation 1,
train workers 2, dev workers 4, prefetch 2 are the measured candidate; retain the
same batch across durations. Short 10 s probes measured ~82→92 audio/s after
optimization; 4 s measured ~176 audio/s. These operational measurements are not
additional complete scientific recipe trials or final-fold results.

Finite losses and trainable gradients still block invalid optimizer updates.
CPU resampling kernels are reused without changing waveform arithmetic;
normalization/probabilities/JS/entropy/losses remain FP32. The equivalent masked
bona-fide JS reduction may change FP32 rounding, so do not claim identical
training trajectories. Duration-specific bounded dev logits matched the original
implementation and cached features exactly. The initial frozen-feature build
still costs a full encoder pass, with visible progress; reuse saves subsequent
passes. Training/export resume, FT4, ablations and target evaluation remain off.

Existing WavLM native source checkpoint selection and native test scoring use
full utterances despite 4 s training. Matched-four-second source calibration and
eventual target reporting remain separately labeled controls under this protocol;
LACF-4s training alone does not make existing WavLM reports a matched control.

## Latest owner decision: 10 s only (2026-10-09)

The owner stopped the 4 s-first queue to avoid another duration-specific multi-hour
preflight. Run fresh optimized **10 s F1→F4 only**, seed 1234; 4 s is deferred,
not automatically queued afterward. Preserve the interrupted 4 s check and old
10 s run. Approved architecture, sampler, losses, precision, optimizer, batch,
workers, selection and no-resume policy remain unchanged.

Retain the already passed full native-10 s source-development/reload evidence,
verifying its file hashes, canonical membership/labels/counts, scores/EER and
checkpoint identity. Before fresh training, bounded current-code regression
loads that original validation pilot to compare representative plus longest
source-dev direct/cached logits exactly with the saved scores, and repeats the
current longest-input finite update/memory check. This validates an execution
refactor without spending another pair of full scoring passes; label the new
record as reused full evidence plus bounded regression, not full current-code
rescoring. A mismatch blocks training. The optimized and original duration/input
policies and frozen official encoders are unchanged; no target data is used.

Fresh final folds never load pilot-trained adapter/head weights. The unavoidable
one-time full frozen-feature extraction is deferred to the first development
phase after the first training epoch, and source selection/reload uses current
backend scores thereafter. Completion still requires the full per-source scores,
strict macro source-EER checkpoint rule and selected-checkpoint reload. No byte
checksum or canonical decode scan is repeated. No 4 s, target evaluation, FT4,
ablations or failed-fold restart is authorized by this change.


## Current target-access policy (owner decision, 2026-10-12)

Completed SOTA runs may be evaluated independently. The owner removed the
comparison-wide freeze prerequisite for all SOTA models, superseding earlier
WavLM-only and AASIST-F1-only exceptions and older mandatory confirmation
instructions. This is permission to evaluate authorized completed runs, not
permission to launch all folds or additional protocols automatically.

Keep completion/checkpoint identity, source-only model/recipe selection,
source-dev reload verification and calibration, canonical target membership,
metrics and confidence intervals unchanged. Target results must never guide
any model's architecture, prompts, recipe or tuning. P2's documented source-only
checkpoint selection record and explicit native score-export permission remain
required; P2/P3/P4 have not been launched by this policy change.

Evaluation commands no longer need `--confirm-protocol-frozen` or
`--confirm-run-frozen`. The flags remain accepted only for old command
compatibility and do not assert a completed comparison-wide freeze. New reports
record `comparison_wide_freeze_confirmed_by_operator=false` and the independent
completed-run access policy. Existing valid metrics/provenance are preserved.

## LACF source-only pre-evaluation preparation (owner authorization, 2026-10-12)

F2 has finished; keep F3/F4 held. The owner now authorizes source-only F1/F2
checkpoint/cache verification, a bounded padding diagnostic and minimal native
score-export implementation. Target evaluation and retraining remain deferred.
This overrides the earlier planning-only restriction for preparation, not for
target access. No optimizer update or feature-recipe change is part of this work.

`python -m scripts.lacf.evaluate --run-dir <completed-run> --fold <fold>` defaults
to source preparation. It verifies the completed primary checkpoint, unchanged
frozen weights/prototypes and encoding code, processor revisions, canonical
source-cache membership/labels/checksums and recorded runtime, then exports
batch-one source scores and reproduces per-domain/macro source EER. It never
rebuilds a missing or mismatched cache. Source thresholds use shared calibration.
The separate bounded direct-audio audit records its sample IDs and scope in
`source_validation.json`; it is not a full direct-audio rescore or proof of
population-wide equality. Historical cache metadata lacks some dependency and
driver versions; retain that limitation rather than assert they were verified.

The source-only padding diagnostic compares individual encoding with WavLM
right-padding to 10 s in a logical batch of 132. Hold CLAP features fixed and run
the backend one recording at a time to isolate WavLM changes. Report class/domain
feature distances, logit shifts and explicitly labelled subset EERs. These are
diagnostics, not new complete recipe trials or target results. Do not patch
normalization into an existing model immediately before evaluating it.

The native exporter opens targets only with explicit `--evaluate-target` and
matching fresh-process source validation. A later owner instruction is still
required. Retain centered common up-to-10 s intervals, short recordings, dual
resampling, frozen eval encoders, BF16 logits and FP32 sensitive calculations.
Reuse the common raw/transductive metrics, 1,000-resample bootstrap (seed 2026),
canonical membership, score manifests and safe partial-score recovery. This
does not add training resume. Custom storage folder names record scientific
`model_family=lacf`; summary `--family lacf` applies the seed-1234 policy while
keeping F1/F2 summaries preliminary and across-seed SD unavailable.

Source preparation has passed on SSH2 for the original F1/F2 checkpoints. The
ignored pre-evaluation report records full cached EER reproduction, bounded
bit-exact direct/cache checks and the measured padding/batch sensitivity. The
diagnostic is consequential for feature consistency, but its subset EER changes
are mixed and do not establish that a revised recipe improves accuracy. Original
development and future native target scoring share the same individual-recording
policy. Keep that system intact unless the owner selects a separate revised
training recipe; reporting changes alone require no retraining.

## Native LACF F1/F2 evaluation launch (owner instruction, 2026-10-12)

The owner explicitly requested starting F1 and F2 evaluation and confirmed that
the recordings are on `/mnt/drive`. This supersedes the preceding deferral for
these two original checkpoints. Fresh source/checkpoint/scorer bindings passed;
canonical local target counts are 346,307 SpeechFake test recordings and 21,000
CFAD clean unseen recordings. Existing transfer evidence verifies destination
presence/sizes; full destination checksums remain waived and are not claimed.

Run sequentially in one durable tmux session, F1 then F2, with no automatic retry
and no overwrite of completed metrics. Keep centered common up-to-10 s crops,
both encoders frozen/eval, encoder/backend batch one, workers 4/prefetch 2,
BF16 with FP32 sensitive operations, existing source thresholds and the shared
raw/transductive reports and 1,000-resample seed-2026 conditional bootstrap.
Recorded padding sensitivity is a limitation of the preserved original recipe;
no normalization patch or target-guided design/tuning is part of evaluation.
F3/F4 remain held; no FT4, ablations, duration controls or P2–P4 are authorized.
Launch/status and target-copy integrity evidence are local-only under
`docs/local/lacf_native_eval_20261012/`; completion requires actual registered
target scores and successful final reports, not this launch instruction.


## SSL feature gating: implementation adaptation (2026-10-18)

The owner requests the third baseline and prefers the accurate released
implementation with lower cost. After reviewing the actual Tran et al. paper
and pinned author code, use family `ssl_gating`, no RawBoost (`algo:0` in the
released YAML), and the fixed recipe in the table above. This is an adapted
baseline without the paper's waveform augmentation; no claim of matching its
published EER or lower total training cost follows. Preserve the existing
three-seed plan for this family until explicitly changed.

Retain full-utterance 16 kHz input, dynamic right-zero-padding for training,
batch-one unpadded development/test, all XLS-R parameters trainable, and its
pretrained internal SpecAugment/dropout/layerdrop settings. No waveform
normalization or padding attention mask is added, matching the released path.
Use the project's existing mono decode/resampler and SDPA attention on the
common environment; record the exact frontend configuration/software versions.
Short recordings that the native encoder cannot process must fail visibly,
not be skipped, repeated, cropped or silently assigned a different recipe.

Repair the released detached CKA and incorrect block/time reshape: concatenate
four block representations at each time position, pool their means/stds, then
penalize mean absolute biased linear CKA over the six distinct pooled-mean
block pairs. Compute it in FP32 with gradients and coefficient 1 alongside
weighted CE. Zero-variance and single-example batches have zero CKA with finite
gradients. Keep the released SiLU-based SwiGLU and block LayerNorm epsilon
1e-12. The paper's sigmoid-only equation and ambiguous diagonal summation do
not override the released gate and distinct-pair convention. These repairs are
explicit method adaptations, not numerically identical upstream training.

Replace the released validation-loader/metadata path with the fold's three
source dev splits only. Domain-uniform/natural-class sampling, no dropped
last batch, strict macro-source EER selection, source threshold calibration,
spoof-positive scores, shared P1 metrics and conditional CIs remain unchanged.
Fresh checkpoint reload and reproduced source macro EER are required before
opening a target. No target feedback was used to choose this implementation.
Only the pinned official pretrained XLS-R initializes training; never load the
released anti-spoofing detector. No training/evaluation, weight download or GPU
validation is authorized or claimed by this implementation entry. Before any
launch, obtain weights and separately authorize source-only full-batch GPU,
loading/short-audio, source-dev and reload validation. No P2–P4 native exporter,
partial-score recovery or training resume is provided for this model yet.


## SSL-gating F1 worker/launch status (owner decision, 2026-10-18)

The owner selects 16 training workers for SSL gating, retaining 4 development
workers, prefetch2, pinned memory and persistent workers. This replaces its
initial four-worker default; no measured worker-speed benefit is claimed.
Batch5, FP32, end-to-end fine-tuning, full recordings, Adam/loss/sampler and
source-only selection remain unchanged. Only F1 seed1234 was authorized, not
additional seeds/folds or target evaluation; the family seed plan is unchanged.

Source-only real-GPU preflight passed three optimizer steps and bounded
six-example checkpoint reload. Peak allocated8.04GiB/reserved8.60GiB is limited
to those samples, not a complete epoch, full dev validation or worst-case bound.
A protective 10GiB allocator budget was used while preserving active AASIST.
The subsequent fresh F1 attempt hit that budget during epoch1 and stopped before
any epoch checkpoint/completion. This is not evidence of a full-device OOM.
The owner then explicitly chose to leave F1 stopped: no queue, restart or
checkpoint resume. Preserve failure artifacts and seek a new instruction for
any future fresh launch; no automatic input/precision/batch/recipe change.
