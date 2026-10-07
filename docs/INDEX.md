# Project documentation

Research and implementation notes live here. Keep the root `README.md` for
onboarding/package metadata and the root `AGENTS.md` for agent instructions.
No training, evaluation, cache or dataset directory was moved by this cleanup.

## Reading order

1. Read the local `PROJECT_HANDOFF.md` if it is available. It is intentionally
   ignored by Git: it contains unpublished experiment results, local paths,
   active-run snapshots and conversation history. It is a guide, not a live
   monitor or a verbatim transcript. Ask for restoration when needed in a
   fresh checkout; do not assume it exists there.
2. Read [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the latest agreed
   folds, recipes, target-isolation rules, normalization and metric contract.
   Dated updates supersede older historical paragraphs.
3. Consult [the reference index](../references/INDEX.md) and the relevant actual
   supplied PDF/DOCX before model/dataset implementation decisions. Those
   originals are local-only and need their own backup.
4. Read [SOTA_PAPER_NOTES.md](SOTA_PAPER_NOTES.md) for paper/code interpretation,
   provenance and reproduction limitations. Notes do not replace the papers.
5. Read [CODE_STUDY_GUIDE.md](CODE_STUDY_GUIDE.md) for file/function inputs,
   outputs and the cache → train → score → report → summary flow.
6. Consult [REVIEW_FIX_REPORT.md](REVIEW_FIX_REPORT.md) as the preserved
   2026-10-02 audit. Its then-current seed/status statements are historical,
   not overrides of later decisions.

## What belongs in Git

Versionable: this index, the protocol, paper interpretation notes, code guide,
historical review, root README/AGENTS, source code, synthetic tests, environment
manifests, reference index/checksums and the vendored architecture's license.

Local-only: `PROJECT_HANDOFF.md`, optional `local/` notes, reference full
PDF/DOCX originals, `outputs/` (including weights, scores, logs, metrics and
recovery journals), raw/cached audio, model/environment caches and credentials.
Ignore rules do not delete local files and do not untrack already tracked files.
Any deliberate release of unpublished results or supplied documents needs a
separate owner decision and, for documents, appropriate publication rights.

## Local supervisor report

[WavLM F1–F3 report](wavlm_f1_f3_report/README.md) contains an editable standalone
LaTeX document, a PDF reading copy, scientific plots and verified report data.
The entire folder is intentionally ignored because it contains unpublished
results. It is a descriptive report of completed native P1 folds; it does not
change the experiment protocol or establish the final four-fold comparison.
Its README records the PDF export method and LaTeX compilation status.

A separate [concise WavLM F1–F3 results report](wavlm_f1_f3_short_report/README.md) presents the same
completed folds with simpler tables and an explicit bootstrap explanation.
It preserves the detailed companion report; both numerical reports remain local-only.

## Documentation maintenance

Update the protocol and paper notes for material study decisions. Refresh the
local handoff after consequential work, recording dated observations separately
from enduring decisions. Preserve the original chat for details a summary
cannot guarantee to capture. Always inspect actual processes/files before
claiming an old snapshot is still current.
