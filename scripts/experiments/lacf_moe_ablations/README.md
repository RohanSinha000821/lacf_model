# LACF-MoE ablations

Owner-authorized core training: A0 reference reuse, then A1–A6, F1–F4,
seed1234 only. This follows the owner's reduced-study launch instruction.
Additional ablations B1–B6 are definitions only and are NOT queued.
Target evaluation is not queued by this source-training launcher.

## Names and locations

Study: `lacf_moe_ablations`. Use `MoE-A0` etc. in reports to distinguish these
IDs from the original main plan's A0–A10, Step4 and C0–C4/D1–D2.
The owner requested sequential numbering before implementation; no ablation
runs existed under the previous proposed numbering. Existing historical IDs,
code, artifacts, cache arrays and reference results remain unchanged.

- Source: `src/audio_deepfake_detection/experiments/lacf_moe_ablations/model.py`
- Runner: `scripts/experiments/lacf_moe_ablations/train.py`
- Tests: `tests/experiments/lacf_moe_ablations/test_ablations.py`
- Training: `outputs/experiments/lacf_moe_ablations/A1_clap_embedding/f1/1234/`, etc.
- Launch: `outputs/experiments/lacf_moe_ablations/launch/`
- Future evaluation: `outputs/experiments/lacf_moe_ablations/evaluations/` (not launched).

Every run records its ID/name, enabled branches, fusion, features, losses,
parameter inventory, cache/encoder provenance and code hashes. Reuse shared
sampling, training math, metrics and export utilities; no foundation encoders
are loaded or executed. Main LACF, MoE and SOTA source files are unchanged.

## Core matrix

| ID | Folder name | Definition | Question | Trainable parameters |
|---|---|---|---|---:|
| A0 | `A0_reference` | Existing three-expert layer MoE + original full35-D relation backend | Main reference, verified reuse | 967,338 |
| A1 | `A1_clap_embedding` | Normalized CLAP audio embedding512 → MLP | CLAP alone, without anchors | 35,041 |
| A2 | `A2_embedding_concat` | [MoE WavLM adapted embedding512; CLAP512] → MLP | Ordinary fusion without anchors | 1,032,612 |
| A3 | `A3_wavlm_anchors` | Layer MoE + adapter → WavLM concept probabilities8 → MLP | WavLM anchor pathway without CLAP audio | 965,556 |
| A4 | `A4_uniform_layers` | Plain mean of13 raw WavLM vectors replaces layer MoE; retain original adapter/35-D backend | Entire layer-MoE contribution; capacity also changes | 661,479 |
| A5 | `A5_layer_and_fusion_moe` | A0 + three-expert relation-fusion MoE before existing classifier | Additional cross-branch processing | 981,463 |
| A6 | `A6_global_fusion_routing` | Same as A5; global learned fusion weights replace recording-dependent fusion router | Routing versus extra expert capacity | 981,288 |

Only detection BCE is enabled in every row; semantic/BF loss weights are zero.
Removing them would not change the current reference. Frozen text prototypes
are read from identical preserved primary buffers only; no old learned weights
initialize fresh fits. New expert/head construction isolates initialization RNG,
retaining shared adapter/layer-expert weights and classifier tail initialization
where applicable. Common weights are freshly seeded, not copied from trained runs.

Core order is A1 F1→F2→F3→F4, then A2 F1→F2→F3→F4, through A6.
This is 24 fresh fits, plus verification/reuse of four existing A0 runs.
Seed1234 is adopted for this owner-approved reduced comparison; earlier
three-seed diagnostics retain their original plan. No other seeds are queued.

## Additional ablations (held)

| ID | Name | Definition | Previous proposal ID |
|---|---|---|---|
| B1 | WavLM embedding-only | MoE WavLM embedding512 → MLP | A1 |
| B2 | CLAP anchor-only | CLAP concept probabilities8 → MLP | A4 |
| B3 | Anchor pair | [p_W;p_C]16 → MLP | A6 |
| B4 | Agreement/difference | [p_W;p_C;abs(p_W-p_C);p_W*p_C]32 → MLP | A7 |
| B5 | Global layer routing | Same three layer experts with global learned weights | A9 |
| B6 | Fusion MoE only | Uniform layer averaging + A5 fusion MoE | A12 |

They are deferred, not missing results, and have no implementation/run entry
in the core trainer. No automatic expansion follows from target results.
For traceability, retained core IDs changed from the draft as follows:
A0→A0, A2→A1, A3→A2, A5→A3, A8→A4, A10→A5, A11→A6.
The completed four-expert model is a supplementary existing comparison;
it is not the main reference and is not retrained here.

## Architecture retained from the main reference

WavLM Base+ and unfused CLAP remain frozen through the existing shared cache.
WavLM groups are R0–R3/R4–R7/R8–R12. Each mean768 feeds a residual
LN768→64→GELU→768 expert; a recording-dependent softmax router mixes them.
The original adapter is LN768→512→GELU→dropout0.2→512, then FP32 L2 norm.
CLAP uses the preserved reference double FP32 normalization of its raw512-D
projection. No per-layer normalization or new feature standardization is added.

Keep the original eight concepts (three bona-fide/five spoof), three templates,
prototype bytes and temperature0.07. Let v_W/v_C be512-D normalized embeddings
and p_W/p_C be their8-D concept distributions. Original full relation features:

`r = [p_W,p_C,abs(p_W-p_C),p_W*p_C,JS(p_W,p_C),H(p_W),H(p_C)]` (35-D).

The head remains LN(D)→Linear(D,64)→GELU→dropout0.2→LN64→Linear(64,16)→
GELU→dropout0.2→Linear(16,1). Only its input block changes where D changes.
Changed readout dimensions/capacity are disclosed: A0 versus A2 does not isolate
anchor semantics alone, and A0 versus A4 does not isolate routing alone.

## Fusion MoE (A5/A6)

Placement: `p_W,p_C → original r[35] → fusion MoE[35] → original35→64→16→1 head`.
Both rows retain the original WavLM layer MoE. Three independent trainable
residual experts see the SAME complete joint relation vector:

`E_i(r) = r + Linear(64,35)(GELU(Linear(35,64)(LN(r))))`.

A5 router: LN35→Linear(35,3)→FP32softmax, final linear zero-initialized.
A6 replaces this router with three learned global logits, initialized zero.
Both combine experts as `sum_i softmax(logits)_i * E_i(r)` in FP32. Every
expert executes; no top-k, extra dropout, new temperature, balancing loss or
bona-fide/spoof expert assignment is introduced.

Each expert has4649parameters; three add13947. A5 router adds178; A6 logits
add3. A5/A6 differ by175parameters, so this is near-capacity-matched, not exact.
Their experts/initial predictions match. A0 versus A5 alone cannot distinguish
extra capacity from routing; retain A6. Record source routing means, entropy,
variation and dominance, separately for bona-fide/spoof, during final scoring.

## Cache, recipe and gates

Reuse `outputs/experiments/lacf_diagnostics/frozen_features/` without relocation.
Verify canonical source membership and payload hashes ONCE when loading all
source train/dev arrays. Subsequent fits receive only their fold's six source
partitions. Target recordings never enter this bank or fitting/selection.

Preserve center+two fixed random up-to10s training views and centered source
development, native unpadded encoder inputs. Recordings≤10s retain identical
physical views; logical aliases do not create extra crops or epoch visits.
Preserve original extraction-segment provenance and acknowledged batching drift.

Batch132/accum1, inverse domain/class bucket mass with replacement, original
epoch recording count, AdamW3e-4/wd1e-4/constantLR, max30/patience5, strict
equal-domain macro source-dev EER/ties earlier. BF16 with sensitive FP32;
retain reference BF16-rounded scalar logits rather than silently rescoring FP32.
Feature workers0/CPUthreads2; CUDA allocator cap1GiB, minimum2GiB live free,
minimum75GiB available RAM before loading the approximately62.6GiB arrays.
Preserve SSL/other jobs. Resource waits do not stop other jobs or alter recipes.

A0 links only after checkpoint/completion/provenance and full source-score
reload verify. Each fresh control/fold first runs8 warmup+16 measured source
updates, full source-development scoring and fresh-model exact reload. Discard
all preflight weights and initialize final training afresh. Bind passed GPU
proof to the model/code/cache/fold. At completion, reload the selected model,
require bit-exact complete source scores/metrics, export canonical IDs/labels,
reproduce exact source calibration, bind all artifacts and verify completion
before advancing. No automatic retry/resume/overwrite or failed-model reuse.
Logs show batch progress/throughput/ETA and per-domain/epoch source metrics.

Launch only through the guarded `launch/run.sh`. Core trainer:
`python -u -m scripts.experiments.lacf_moe_ablations.train`.
No target evaluator or additional-ablation entry point is supplied here.
Future evaluation requires source-verified checkpoints and its own authorization;
reuse verified matching target feature caches rather than re-encoding audio.
Report per-fold and equal-weight four-fold means, EER/AUROC, raw and separately
transductive z-score APCER/BPCER/ACER at ALL source1%,5%,10% settings, with
shared1000conditional bootstrap(seed2026). These intervals do not measure
seed variability; small BF16-rounded differences remain qualified.

This study is exploratory after earlier target exposure. Fixed mechanism
contrasts do not select a new model from target performance. Any subsequent
architecture/hyperparameter choice uses inner source-only validation; external
sets remain held. No reference/past score/calibration/metric artifact is changed.
Original main-plan A0–A10 remain distinct; earlier three-branch designs,
auxiliary-loss additions and encoder fine-tuning are outside this frozen queue.
