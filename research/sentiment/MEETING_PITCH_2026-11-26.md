# Sentiment and RST granularity: pitch for 2026-11-26

## One sentence

On review sentiment, fine-grained RST relations beat coarse ones in all four
standards on a pre-registered, held-out test of 16,000 Amazon reviews. The
gain appears only when the model can use fine labels cheaply and has enough
training data.

## What we did

- **Task.** Classify Amazon reviews as positive (4–5 stars) or negative (1–2
  stars). A frozen CardiffNLP model scores every EDU. The RST tree decides how
  those clause scores combine.
- **Model.** The tree becomes satellite→nucleus links between EDUs. Each link
  is described by the sentiment of its two ends. The model learns one weight
  per relation label for how much that label's links count. *Fine* uses the
  parser's labels and *coarse* maps them to broader groups. Both use the
  same trees. Adapted from Chistova (2023).
- **Standards.** PCC, NLDT, GUM/eRST and RST-DT parse the same EDUs.
- **Design.** We explored on 2,000 Amazon reviews and 400 SFU reviews. We then
  froze the analysis plan in a tagged commit and tested on 16,000 *new*
  Amazon reviews with no shared reviews, reviewers or products. There was one
  primary test per standard, with Holm correction and a 0.002-nat margin.

## What we found (confirmation cohort)

1. **Relation labels help.** Labelled links lowered log-loss by 0.013–0.019
   nats against unweighted links. All p < .001.
2. **Fine beats coarse in all four standards.** Fine had 0.0014–0.0040 nats
   lower log-loss, Holm p ≤ .014, and 0.1–0.3 points higher macro-F1. Fine gate
   models reached .880–.887 macro-F1.
3. **The distinctions themselves carry signal.** Real fine labels beat 200 out
   of 200 shuffles within their coarse groups. Shuffled fine labels perform
   like coarse labels. Some learned weights are interpretable and stable across
   standards. *Contrast* counts 2.3–3.8× as much as *concession* in all four,
   although coarse schemes merge them. In PCC and GUM/eRST, *result* also
   counts 2–3× as much as *cause*. This does not hold in RST-DT or NLDT.
4. **Fine labels need data.** With 1,000 training reviews fine and coarse tie.
   The fine advantage grows with every doubling of training data, Holm p < .001
   in all four standards. The exploration estimates were similar in size, but
   2,000 test reviews gave intervals too wide to separate fine from coarse.
   SFU's 320 training reviews per fold sit at the low end of the curve.
5. **Model capacity matters.** Relation pooling, which spends six features
   per label, was hurt by fine labels at 2,000 reviews. At 16,000 it benefits
   in two standards. RST-DT, with 69 labels, is still too sparse for it.

## How this fits the paper

- HC3 remains the main granularity result. Sentiment is a second task with
  the same message and a mechanism: fine labels help when the model can afford
  their sparsity.
- **Amazon is the main sentiment experiment.** It has a large sample, one tree
  per review, star labels and an explore-then-confirm design.
- **SFU goes in an appendix as a robustness check.** It has 400 reviews, so
  one review equals 0.25 F1 points. Long reviews are split into unconnected
  trees. Our split ablation shows that the splitting itself costs little.
  SFU mainly lacks the data volume that fine labels need.

## Caveats to state up front

- The gain is reliable but small in accuracy: about 0.1–0.3 F1 points.
- *Updated after the baselines, 2026-11-26.* On the 16,000-review cohort, the
  RST models (.880–.887 macro-F1) match a word and word-pair TF-IDF baseline
  (.882), with slightly better log-loss. Fine-tuned roberta-base reaches .942.
  We use RST to study which relations carry sentiment, not to beat accuracy
  baselines. [Baselines](RUNNING.md#baselines-on-the-confirmation-cohort)
- Coarse labels are fine labels renamed after parsing, which isolates label
  granularity. Separately trained coarse parsers produce different trees and
  are a different question.
- English reviews from four product categories, parsed by available UniRST
  checkpoints: stock for PCC and NLDT, fine-tuned for GUM/eRST and RST-DT.

## Questions for the colleague

*Outcome, 2026-11-26:* Amazon is the main sentiment corpus and SFU goes to an
appendix. The branch and the plan tag are pushed to Max's fork. The HC3
scrambled-label control is still open.

1. Is Amazon as the main sentiment corpus, with SFU in an appendix, acceptable
   for the paper?
2. Should HC3 also get the scrambled-label control? It is cheap and would show
   that fine labels carry information there too, beyond extra parameters.
3. Should the confirmation plan and its tag be pushed to the shared fork now,
   for a public timestamp?

Sources: [plan and results](GRANULARITY_CONFIRMATION.md) ·
[exploration diagnostics](RUNNING.md#finecoarse-diagnostics-with-log-loss-and-scrambled-labels) ·
[split ablation](RUNNING.md#split-tree-ablation-on-amazon)
