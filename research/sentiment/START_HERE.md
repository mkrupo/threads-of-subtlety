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

Use HC3 as the main granularity experiment and SFU plus Amazon as the
two review sentiment corpora. Relation pooling changed direction across
the corpora. Length, label definitions, category mix, and sample size
changed together. The meeting notes cover both review experiments and
what each classifier sees. The committed LaTeX section covers SFU.
[Meeting notes](replication/MEETING_NOTES.md) · [Runbook](RUNNING.md) ·
[LaTeX section](paper/sentiment.tex) · [BibTeX](paper/sentiment.bib) ·
[Method sources](RELATED_WORK.md#attribution-for-the-completed-sfu-experiments)
