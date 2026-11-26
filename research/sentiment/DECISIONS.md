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

## 2026-11-18 — Preliminary EDU scorer check

- Tested the binary SST-2 DistilBERT checkpoint at revision
  `714eb0fa89d2f80546fda750413ed43d93601a13` with its released ONNX model
  and tokenizer on six invented strings using ONNX Runtime 1.22.1 on CPU.
  The obvious praise and complaint got the expected signs.
- Factual clauses received large scores in either direction. For example,
  `The box measures 20 centimeters.` scored −0.954 and
  `The package arrived on Tuesday.` scored +0.966 on
  `P(positive) - P(negative)`. Score magnitude alone is a poor proxy for
  neutrality on these examples. The probe motivates checking a candidate
  with an explicit neutral class on representative review EDUs before
  freezing the scorer. Scorer selection and SFU sentiment fitting remain
  pending. [Pinned checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13)

## 2026-11-18 — Shared scorer agreed

- Use one frozen EDU sentiment scorer across all inventories, maps, and
  folds for the first experiment. Fit the review-level model within the
  frozen SFU folds.
- Scorer selection remains open. Logistic regression is a candidate
  review-level model using sentiment and RST features. This would parallel
  the classifier used in the HC3 experiment.

## 2026-11-18 — Neutral-aware scorer shortlist

- Compared the three-way CardiffNLP model and VADER 3.3.2 on 12 invented
  clauses, with four factual statements, four instances of praise, and
  four complaints. The neural probe used the quantized ONNX export at
  revision `f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445` on CPU and summarized
  polarity as `P(positive) - P(negative)`. [Pinned export](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest/tree/f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445)
- All four factual statements had neutral as their most likely class
  with CardiffNLP. VADER assigned them zero compound polarity. For
  `I would never buy it again.`, CardiffNLP gave −0.870 and VADER gave
  0.000. Scores on these examples support trying CardiffNLP first. The
  model was trained for tweets and still needs an EDU sanity check on
  review text. SFU review outcomes played no part in this probe.
- An EDU scorer can supply positive, neutral, and negative scores to a
  binary review classifier. The review targets remain positive and
  negative. The scorer and input features are still proposals.

## 2026-11-18 — Scorer check on real SFU EDUs

- Sampled 32 cached EDUs from distinct reviews, one per domain and length
  band. The assistant judged each isolated span before running either
  scorer. Overall review labels and downstream results played no part.
- Seven fragments were unclear out of context. On the remaining 25,
  CardiffNLP matched 23 judgments and VADER matched 15. CardiffNLP matched
  10/11 factual, 10/10 positive, and 3/4 negative judgments. Its two
  disagreements still illustrate the limits of scoring isolated spans.
  These are one reviewer's sanity-check judgments. [Audit](results/sfu-edu-scorer-check.json)
- Recommend freezing the tested CardiffNLP export and keeping all three
  scores for each EDU. This recommendation precedes the final scorer
  decision and any review-level fitting.

## 2026-11-18 — CardiffNLP scores frozen

- Max approved the tested three-class CardiffNLP scorer. Use its pinned
  ONNX export and shared PCC EDUs to generate one cache of negative,
  neutral, and positive probabilities for all conditions and folds.
- Verified 36,850 scores on 1,012 scenes from 400 reviews. Keep the
  three probabilities through the scoring stage. The review-level
  prediction will still use the two SFU labels. [Score manifest](results/sfu-edu-scores-v1.json)

## Equal-weight EDU baseline

- Use each review's mean negative, neutral, and positive EDU scores as
  the three classifier inputs. Give each EDU equal weight across scenes.
- Use the text controls' frozen outer and inner folds, logistic
  regression settings, C grid, and 0.5 threshold. Standardize the three
  features within each fitting partition, as for the length-only control.

## Nuclearity-only comparison

- Classify an EDU as central when its path from its scene root follows
  only nucleus branches. An EDU below any satellite branch is supporting.
  A one-EDU scene is central, and either branch of an NN split can stay central.
- Across all scenes in a review, average the three frozen scores separately
  for central and supporting EDUs, giving every EDU equal weight within its
  group. If there are no supporting EDUs, use the review mean for that group.
- Fit logistic regression on those six inputs for each inventory using
  the equal-weight baseline's scaling, training folds, C grid, and threshold.
  Compare held-out predictions against the equal-weight baseline for the
  same reviews. This step measures the chosen nuclearity grouping and
  uses no relation names.

## Fine/coarse relation comparison

- Keep the six nuclearity features. At each internal node, average EDU
  scores separately in its left and right spans. Pool the resulting six
  numbers by relation label over all nodes and scenes in a review. Each
  occurrence contributes equally to its label's mean.
- Use the parser's declared alphabet for feature columns, in sorted order.
  Use all zeros when a label is absent from a review. This encoding can
  carry relation-presence information as well as sentiment.
- Use the audit's pinned research maps to merge fine sums and occurrence
  counts before computing coarse means. Keep the source trees and EDU
  scores fixed. Require a map entry for every observed relation.
- Use the same logistic regression settings and inner and outer folds as
  the nuclearity control. Report the paired fine-minus-coarse difference
  in macro-F1, prediction changes, and label coverage for each inventory.

## Fine/coarse stability check

- Use the saved held-out predictions. Draw 20,000 paired bootstrap
  samples of reviews with seed 42, keeping the number of reviews in each
  fold, domain, and gold-label group fixed. Each draw uses the same reviews
  across all inventories and both granularities.
- Report the 2.5th and 97.5th percentiles of the fine-minus-coarse
  macro-F1 difference. Also report each fold's and domain's difference,
  and recalculate the overall gap with each fold omitted in turn.
  These intervals describe resampling the saved predictions and hold
  the original fits and splits fixed.

## HC3-style motifs on SFU

- Count connected three-node motifs using the directed, relation-labeled
  graphs from the HC3 experiment. Preserve its counting of symmetric
  matches and its scene-normalized, occurrence-weighted depth.
- Learn the catalog of all observed fine motifs from each fitting
  partition, including during inner validation. Apply the HC3 relation
  maps to that catalog for the coarse features. Keep the same occurrences
  in both conditions, and ignore fine motifs absent from the catalog in
  both. The features are motif frequency and depth per review.
- Reuse the SFU nested folds, logistic regression, training-only scaling,
  C grid, and primary macro-F1. Use the same paired bootstrap draws for
  fine/coarse and between-inventory comparisons. These are pointwise
  intervals conditional on the held-out predictions.

## 2026-11-25 to 2026-11-26 — Granularity confirmation and paper scope

- **Outcome measure.** Fine and coarse labels are compared by held-out
  log-loss, with macro-F1 reported alongside. Exploration differences were
  under one macro-F1 point and changed sign between models, and macro-F1
  responds only when a review crosses 0.5.
- **Mechanism checks.** Fine labels were shuffled within their coarse groups
  (scrambled-label control), and one-scene Amazon trees were cut in two (split
  ablation). Both ran on the exploration data. [RUNNING](RUNNING.md#finecoarse-diagnostics-with-log-loss-and-scrambled-labels)
- **Confirmation.** A disjoint 16,000-review Amazon cohort was drawn. The
  analysis plan was frozen at tag `granularity-confirmation-plan-v1` before
  any fit. Max chose a 0.002-nat margin and the relation-gate model as the
  only primary test. [Plan and results](GRANULARITY_CONFIRMATION.md)
- **Paper scope (Max and colleague, 2026-11-26).** Amazon is the single main
  sentiment corpus and SFU becomes an appendix.
- **Baselines.** Text controls and a fixed-setting roberta-base fine-tune run
  on the same folds. They are descriptive and were added after the
  confirmation. A zero-shot LLM baseline on the university Jarvis Qwen
  deployment was dropped because the configured endpoint returned HTTP 404.
  [Baselines](RUNNING.md#baselines-on-the-confirmation-cohort)
