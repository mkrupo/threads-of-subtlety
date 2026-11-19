# RST and sentiment in one page

**Our question** is whether the choice of discourse standard, or the detail
of its relation labels, changes how well we can predict a review’s
sentiment. This adds an English sentiment task alongside the paper’s
authorship and argumentation work.

## From review to scenes and EDUs

SFU supplies 400 English reviews labeled positive or negative. The median
review is 727 BERT tokens. To parse them, we pack paragraphs into chunks of
about 492 XLM-R tokens, using sentence boundaries inside long paragraphs.
We call each chunk a **scene**. There are 1,012 scenes, and 271 reviews have
more than one. Each parser builds a tree per scene, so links across cuts are
missing. [Length audit](results/sfu-token-lengths.json) · [Parse audit](results/sfu-full-parser-audit.json)

The PCC segmenter splits each scene into **EDUs**, usually clauses. All
four parsers receive those same EDUs. CardiffNLP gives each EDU scores for
negative, neutral, and positive sentiment. We pool those fixed scores
across scenes before logistic regression predicts one positive or
negative label per review. [Score manifest](results/sfu-edu-scores-v1.json)

## What the classifier sees

Take “The screen looks great, but the battery dies by noon.” Suppose the
tree treats the battery complaint as central and links the two clauses
with a concession. A side of a relation can also be a whole span of clauses.

- **Equal weight.** Average all EDU scores across the review's scenes.
  The classifier gets one set of three averages.
- **Nuclearity.** Average central and supporting EDU scores separately.
  An EDU is central if its path to the scene root follows only nucleus
  branches. The classifier learns how to use the six averages. This is
  the structural control because it uses the tree's prominence information
  and omits relation names.
- **Relations.** Keep the six nuclearity inputs and add sentiment by
  relation name. For the concession, the left side is the screen praise
  and the right side is the battery complaint. We average the EDUs on each
  side, then average those pairs over all occurrences of the label across
  the review's scenes.

## Which parsers and labels we compare

PCC and NLDT use released UniRST weights. GUM/eRST and RST-DT use the
colleague's fine-tuned checkpoints. We parse each scene once per inventory.
The fine run uses the emitted names. The coarse run uses the research
map when grouping the same relations. For example, eRST concession and
contrast both become `adversative`. The EDU cuts, attachments, and
nuclearity stay fixed within each fine/coarse pair. [Model provenance](MODEL_PROVENANCE.md)

## What we found

On the same 400 held-out reviews, equal weight scored **0.817 macro-F1**
and nuclearity scored **0.822–0.832**. All eight fine/coarse runs scored
below their nuclearity controls. Coarse was ahead in three inventories,
with the largest gap in RST-DT. The paired review-resampling intervals
include a tie in all four inventories. [Results and reproduction](RUNNING.md) · [Stability check](results/sfu-relation-stability-v1.json)

We also replayed the colleague's HC3 features for six answers and refit
his downstream classifiers from the saved features. HC3 used scene cuts
too, with far fewer answers split than in SFU. [HC3 check](HC3_REPRODUCTION.md)
