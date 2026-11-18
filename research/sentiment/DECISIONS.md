# Sentiment preparation decision log

## 2026-11-17 — Scope agreed

- Source: Max's meeting report and follow-up instructions.
- Downstream sentiment is a testbed for comparing RST inventories and relation
  granularity. A novel sentiment model or state-of-the-art score is unnecessary.
- Reuse an established relation-aware method; keep a common local sentiment
  signal and structure/position controls. Gold document sentiment is required;
  gold RST is optional diagnostic material.
- Exact trained fine-inventory checkpoints and model/policy provenance remain
  with the colleague. Corpus preparation and non-RST controls can proceed.

## 2026-11-18 — Preparation protocol fixed before baseline outcomes

- Use the audited SFU English raw archive (SHA-256
  `3c53884a218c6c218b128440a004de8168d549ebd911277a47c8e7fac28c03ac`)
  as the first testbed:400 reviews, eight domains,25 per sentiment/domain.
- Preserve decoded text with canonical LF line endings for future parsing.
  Use a separate whitespace-normalized content hash for identity; exclude
  filenames/domain metadata from learned features.
- Freeze five outer folds and four inner folds, stratified by domain and
  label, with seed42. Every review is held out once; all representations will
  share the same saved fold IDs.
- Baseline configuration was written before fitting real reviews:
  `baseline_config.json`, protocol `sfu-text-controls-v1`.
- Compare training-prior, log-length LR, and word TF-IDF unigram/bigram LR.
  Tune only C over[0.1,1,10] by mean inner macro-F1, with smaller-C tie break.
  Vocabulary, IDF, scaling, and classifier are fitted inside each inner split.
- Keep a fixed0.5 threshold for LR. The tied prior baseline chooses class0
  through the training-prior argmax. Main score is pooled out-of-fold macro-F1
  over negative/positive; accuracy and class/domain summaries are secondary.
- Create a separate minimal environment with a committed `uv.lock`, rather
  than installing the heavy legacy parser stack to run text controls.
- A bounded GPT-6.2 Sol subagent implemented corpus preparation; an independent
  reviewer checked leakage, source integrity, and validation coverage.

## 2026-11-18 — Corrections made before fitting real baseline data

- Use streaming JSONL reads so Unicode line separators within review text do
  not split valid records.
- Compare prediction counts to ID keys, not dictionary payload values.
- Verify prepared-file hashes against preparation provenance before fitting;
  recompute normalized-text hashes and check metadata/fold agreement.
- Assert domain-by-label quotas explicitly and regenerate the declared folds;
  merely requiring both classes would also accept an unintended domain-held-out
  evaluation.
- Add a narrow Git ignore exception for the sentiment tests: the inherited
  `test_*.py` ignore rule would otherwise hide them from a future commit.
- Validate these behaviors with fixtures and corruption/leakage regression
  cases before training the real text controls.

## 2026-11-18 — Repository strategy proposed

- Recommend a personal fork with `feature/sentiment`, retaining the colleague's
  repository as `upstream`; see `REPOSITORY_WORKFLOW.md`.
- No GitHub fork or remote change has been made. Existing work remains local
  and uncommitted until the repository arrangement is chosen.

## 2026-11-18 — First real text-control run completed

- No configuration was changed in response to its real held-out outcomes.
- All14 tests passed before the real fit. Five outer-fold fits per control
  produced one held-out prediction per review and control,1200 prediction
  records overall.
- Pooled macro-F1/accuracy: prior0.3333/0.5000;
  length-only LR0.6166/0.6175; TF-IDF LR0.8274/0.8275.
- The fixed run completed in the isolated environment; all source/input/output
  hashes, fold membership, and saved metrics passed the verifier.
- A separate agent used standard-library calculations to verify raw archive
  text/metadata, fold balance, hashes, and pooled/per-fold/per-domain metrics.
  It reported no remaining defect in those checks.
- The length signal supports keeping the planned length/structure control.
  No RST comparison, parser evaluation, or framework claim has been made.
- Portable aggregate results and full provenance are in
  `results/sfu-text-controls-v1.json`. Per-document predictions/models remain
  in the ignored local artifact tree.

## 2026-11-18 — Local repository cleanup

- Created `feature/sentiment` from the existing `dev` tip; the colleague's
  branches were not changed. The initial local sentiment commit records the
  source, tests, protocol, lockfile, documentation and aggregate results.
- Preserved the supplied paper/email snapshot on disk and ignored its
  `RST_relation_granularity/` directory. Generated corpora, fitted models,
  predictions and the virtual environment remain ignored.
- Fixed the inherited broad `results/` rule with an explicit allowance for
  the two portable aggregate/verification JSON files.
- Max's fork is now configured as `origin` and `feature/sentiment` tracks its
  pushed branch. The colleague's repository is `upstream`.

## 2026-11-18 — Parser validation protocol

- Use the supplied full eRST and RST-DT artifacts through the colleague's
  pinned `dev` loader and the stock PCC and NLDT routes. Load each parser
  separately in FP32 on one personal GTX 1080 Ti. Match the checkpoint,
  config and ordered label table hashes after transfer.
- Keep the branch's scene splitter and PCC offset-based segmentation for
  this pilot. Reuse the cached EDUs for all four parsers. The original HC3
  scene inputs and trees would be needed to check historical tree equality.
- Require valid binary trees and faithful RS3 exports on two fixtures
  before running the six SFU reviews chosen by predeclared length quantiles.
  Reuse the pinned research-branch coarse maps exactly. Report unmapped
  relations and withhold coarse counts when any observed label is unmapped.
- Keep the supplied checkpoints, local reviews and per-scene trees out of
  Git. Preserve hashes, ordered alphabets and aggregate results.
- Recomputed the eight saved HC3 LR MF+WAD fine/coarse test F1 values
  using the validation-selected thresholds. All matched the supplied
  metrics and the corresponding 1,932 feature-matrix test identities.
- All eight fixture parses and 60 SFU scene parses passed. The pilot
  observed 553 relation instances per inventory, all covered by the maps.
  Keep the map gaps visible for the full corpus and confirm their handling
  before fitting the fine/coarse sentiment comparison. The pilot used the
  existing SentSplit scene policy and did not establish historical HC3
  parse identity or evaluate parser accuracy against gold trees.

## 2026-11-18 — HC3 reference replay

- Checked parser inference on six original HC3 test texts matched by
  canonical identity, separately from the SFU parser smoke test. All 24
  document–inventory feature comparisons matched.
- Refit the paper's LR and RF MF+WAD comparisons from the supplied
  matrices. RF reproduced the F1 table exactly. The LR refit preserved
  the four fine/coarse findings with changes under .001 F1.
- Checked ArgMicrotexts gold counts against the public annotations.
  Replaying the draft's prediction comparison requires the colleague's
  evaluation protocol or outputs. [Detailed reproduction record](HC3_REPRODUCTION.md).

## 2026-11-18 — Full SFU parser audit

- Parsed all 400 frozen reviews on 1,012 scenes with shared PCC segmentation.
  The first pass hit converter alignment errors on the same 31 scenes
  across all four inventories. The targeted retry supplied the original
  EDU text to the converter and retained the decoder's native tree spans.
- Recovered all 124 affected parses. Verified the retained original trees,
  12 converter control trees, all 60 pilot trees, and complete four-inventory
  coverage. Every observed relation maps under the original research maps.
  [Full parser audit](results/sfu-full-parser-audit.json).
