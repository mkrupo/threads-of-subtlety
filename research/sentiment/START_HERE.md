# RST granularity in one page

**Project question.** When do different discourse standards and finer
relation labels help a prediction task? We use authorship detection,
argumentation, and sentiment to study that question.

## What fine versus coarse means

For HC3 and the sentiment experiments, PCC and NLDT use stock UniRST. GUM/eRST and RST-DT use
the colleague’s fine-tuned checkpoints. Each inventory predicts its own
trees from the same text chunks and PCC clause boundaries. Within an
inventory, **both runs use the same trees**. Fine keeps the predicted
relation names. Coarse merges some names after parsing, such as eRST
concession and contrast into `adversative`. The draft also reports a
separate check against human annotations, where the separately trained
coarse GUM/eRST and RST-DT parsers scored higher. [Model provenance](MODEL_PROVENANCE.md)

## What we know so far

- **HC3 authorship.** The classifier uses how often small discourse
  patterns occur and how deep they are in the trees. Fine scores higher
  than coarse for all four inventories with both classifiers. The saved
  paired analysis supports all four logistic regression gains and the
  PCC random forest gain after correction. We reproduced the downstream
  results from the saved features and checked fresh parsing on six
  answers. [HC3 checks](HC3_REPRODUCTION.md#hc3-classifiers)
- **ArgMicrotexts.** The draft reports better argument predictions with
  human-annotated RST than with predicted RST. We checked the public gold
  relation counts. Verifying the performance comparison requires the
  evaluation setup or saved predictions. [Argumentation check](HC3_REPRODUCTION.md#argmicrotexts)
- **SFU sentiment.** On 400 English reviews, equal-weight sentiment
  averages scored **0.817 macro-F1**. Averaging central and supporting
  material separately scored **0.822–0.832**. Adding sentiment by relation
  label scored lower in every run. Coarse led fine in three inventories,
  and all four paired resampling intervals include a tie. [SFU results](RUNNING.md#finecoarse-relation-pooling) · [Stability check](RUNNING.md#finecoarse-stability)
- **Amazon sentiment.** On 2,000 shorter reviews, equal-weight pooling
  scored **0.863** and adding relation blocks improved all eight
  text-order conditions over their nuclearity controls. Coarse NLDT
  reached **0.884**, a gain of **2.00 macro-F1 points** over nuclearity.
  [SFU–Amazon results](replication/results/amazon/comparison.json)
- **Amazon confirmation, 2026-11-26 (pre-registered).** Weighting tree links
  by relation label lowered log-loss on a fresh cohort of 16,000 reviews in
  all standards. In the relation-gate model, fine labels beat mapped coarse
  labels in all four standards. Holm p ≤ .014, log-loss is lower by
  0.0014–0.0040 nats, and macro-F1 is higher by 0.1–0.3 points. Real fine
  labels beat labels scrambled within their coarse groups. The fine advantage
  grows with training size and is clear from a few thousand reviews. The
  2,000-review exploration had similar point estimates but intervals too wide
  to separate fine from coarse. [Plan and results](GRANULARITY_CONFIRMATION.md)
  On the same reviews, TF-IDF reaches .882 macro-F1, about the same as the
  RST models, and fine-tuned roberta-base reaches .942.
  [Baselines](RUNNING.md#baselines-on-the-confirmation-cohort)
- **HC3-style motifs on SFU.** With motif frequency and depth as the
  review's only inputs, fine led in three inventories. Every fine/coarse
  and between-inventory interval includes a tie. [Motif results](RUNNING.md#hc3-style-motifs-on-sfu)
- **UCI sentence check.** Over half of its 3,000 sentences had just one
  EDU. It documents how the pipeline behaves with very little discourse
  structure. [Three-corpus results](replication/results/uci/comparison.json)

## How SFU gets one prediction per review

Long reviews are split into scenes of up to about 492 XLM-R tokens. Each
scene gets its own trees, which can only link material inside that scene.
We score each clause-sized unit (EDU) once for negative, neutral, and
positive sentiment. Logistic regression gets the pooled scores across
all scenes. In “The screen looks great, but the
battery dies by noon,” the nuclearity model can learn to treat the main
point and supporting material differently. The relation model also sees
the sentiment on each side of the concession, pooled by relation name.
[Parse audit](results/sfu-full-parser-audit.json) · [How the features work](RUNNING.md#nuclearity-control)

## What goes into the paper

**Agreed on 2026-11-26.** HC3 remains the main granularity experiment.
**Amazon is the single main sentiment corpus**, with 2,000 exploration reviews
and a pre-registered confirmation on 16,000 new reviews. **SFU moves to an
appendix** as a robustness check on long reviews. Its relation-pooling
results ran in the opposite direction from Amazon's. Length, label
definitions, category mix and sample size all changed together.

Open work for the write-up:
- Rewrite `paper/sentiment.tex` around Amazon; it still covers SFU only.
- Add the Amazon Reviews '23, Chistova (2023), roberta-base, TweetEval and
  fastText references to `paper/sentiment.bib`.
- Call the primary model the *relation-gate model*. Earlier notes also call
  it "Experiment 3", "dependency weighting" and "weighting connections".
  Its definition is in [DEPENDENCY_PILOT](DEPENDENCY_PILOT.md) and
  `relation_gate.py`.
- SFU numbers differ slightly between documents. `DEPENDENCY_PILOT.md` refits
  with fixed C = 1, while the meeting notes and LaTeX use nested choice of C.
  Cite one and say which.
- An LLM zero-shot baseline was dropped because the Jarvis endpoint returned
  404. roberta-base serves as the neural baseline.

The meeting notes below describe the superseded SFU and Amazon exploration.
[Meeting notes](replication/MEETING_NOTES.md) · [Runbook](RUNNING.md) ·
[LaTeX section](paper/sentiment.tex) · [BibTeX](paper/sentiment.bib) ·
[Method sources](RELATED_WORK.md#attribution-for-the-completed-sfu-experiments)
