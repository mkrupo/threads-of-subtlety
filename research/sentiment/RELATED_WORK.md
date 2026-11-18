# Prior work to build on — checked 2026-11-17

Purpose: identify a feasible method and the actual research gap. This is a targeted search, not an exhaustive literature review or a novelty certification. The implementation plan is in [PLAN.md](PLAN.md).

## Closest Potsdam work

**Uladzimir Sidarenka, also known as Wladimir Sidorenko, 2019, _Sentiment Analysis of German Twitter_.** The dissertation identifies Manfred Stede as supervisor. Chapter 6 concerns discourse-aware sentiment; **section 6.4.2, printed pp. 161–165 / PDF pages 177–181**, specifically examines parsing quality and relation schemes. It retrains DPLP on PCC with different relation sets, then tests downstream sentiment. Benefits depend on the method and corpus. This is direct precedent for a granularity question, not just generic sentiment background. [Primary thesis](https://arxiv.org/pdf/1911.13062).

The same lab's [author page](https://angcl.ling.uni-potsdam.de/people/sidarenka.html) connects this researcher to discourse processing, sentiment, PotTS, and released software. Its biographical “current” wording is old; use it for research provenance, not present employment status.

Implication for our study: cite the prior relation-scheme experiment. A proposed extension is modern multi-inventory comparison plus **fixed-tree coarsening**, which separates the loss of relation distinctions from changes caused by training/parsing with another label set. This is a candidate contribution, not a claim that no other work has used that control.

## Released DASA implementation: a concrete reuse opportunity

Repository: [WladimirSidorenko/DASA](https://github.com/WladimirSidorenko/DASA), archived/read-only. Inspected revision: **`618d9060a5fd6f567628c8dec5e26943c8c49ad4`**. The README lists PotTS and SB10k input preparation, automatic EDU sentiment scores, automatic RST trees, train/test commands, and baseline/structured methods. [Pinned README](https://github.com/WladimirSidorenko/DASA/blob/618d9060a5fd6f567628c8dec5e26943c8c49ad4/README.rst).

The repository contains processed JSON, several tree schemes, local/document polarity vectors, and prediction files. One inspected dev file has 99 records, with top-level `tweets`, and record fields `toks`, `msg_id`, `text`, `edus`, `rst_trees`, `label`, and `polarity_scores`. Its tree variants are `chenlo`, `heerschop`, `bhatia`, `zhou`, and `pcc`. This confirms recoverable example artifacts; it does not verify all corpora, folds, model files, or reported scores. [Inspected example file](https://github.com/WladimirSidorenko/DASA/blob/618d9060a5fd6f567628c8dec5e26943c8c49ad4/data/PotTS/dev/7.addition.json).

### Small methods worth inspecting first

| Component | What the inspected code does | Use for us |
|---|---|---|
| [DDR](https://github.com/WladimirSidorenko/DASA/blob/618d9060a5fd6f567628c8dec5e26943c8c49ad4/dasa/ddr.py) | Builds a dependency-style discourse tree, weights EDU sentiment vectors by depth, and uses a small learned output layer. | Simple structural control. Its weighting does not inspect relation names. Fixed-tree relabeling alone should not change its input features. |
| [R2N2](https://github.com/WladimirSidorenko/DASA/blob/618d9060a5fd6f567628c8dec5e26943c8c49ad4/dasa/r2n2.py) | Learns relation-indexed matrices, combines child polarity vectors recursively, and mixes the root result with document scores through a learned parameter. | Candidate relation-aware model: keep local scorer and architecture fixed, vary inventory and label projection. |
| Last / Root / No-Discourse | Baseline methods are present in the repository. | Controls for position, prominent EDU selection, and direct sentiment prediction; not all test label granularity. |
| [Tree representation](https://github.com/WladimirSidorenko/DASA/blob/618d9060a5fd6f567628c8dec5e26943c8c49ad4/dasa/rst.py) | Nodes have IDs, relation-to-parent, nuclearity, children; leaves read token indices and local polarity vectors. | A compact common tree/EDU-score adapter is needed to use the colleague's outputs. This is not the same as motif JSON. |

The [constants](https://github.com/WladimirSidorenko/DASA/blob/618d9060a5fd6f567628c8dec5e26943c8c49ad4/dasa/constants.py) use negative/neutral/positive classes and name several relation schemes. Our proposed review data is binary, so model dimensions and evaluation labels must be adapted rather than silently dropping a class.

### Reuse assessment

**Reproduce the small model/mechanism first; avoid making a complete legacy-environment restoration the critical path.** The [requirements](https://github.com/WladimirSidorenko/DASA/blob/618d9060a5fd6f567628c8dec5e26943c8c49ad4/requirements.txt) include scikit-learn `~=0.19.1`, pystruct `~=0.3.1`, cvxopt, and Pyro alongside PyTorch. The code was inspected but not installed or executed. No exact historical score has been reproduced.

Porting an inspected aggregator into a small modern environment, using recorded new parser conditions, is an **adaptation/extension**. Exact reproduction additionally requires matching historical data, segmentation, parser models, local sentiment vectors, training procedures, and metric definitions. Do not confuse those two achievements.

The older DASA metric tables distinguish a macro score over positive and negative from a micro score over all three classes. A new all-class macro-F1 must not be compared under the same name without explaining the difference.

## Other relevant references

### Bhatia, Ji, and Eisenstein (2015)

[_Better Document-level Sentiment Analysis from RST Discourse Parsing_](https://aclanthology.org/D15-1263/) uses automatic RST parses to aggregate local sentiment through depth weighting and a recursive model. Its relation-aware experiment reduces relations to contrastive/non-contrastive groups rather than testing all native inventories. This supports a small compositional baseline; it does not establish universal gains from fine labels. DASA implements related methods, so the Potsdam implementation can be the practical bridge.

The cleanest corpus match to the 2015 experiment is Pang and Lee's English
movie-review polarity v2.0 (2,000 reviews, ten-fold cross-validation). The
2015 paper attributes its second dataset to Socher et al. (2013), but its
50,000 full reviews, 25,000/25,000 split, and rating cutoffs all match
Maas et al.'s IMDb release. The authors' public code references the
`aclImdb` directory. This points to IMDb, while the paper leaves the
exact input manifest uncertain. [2015 paper, §2.3](https://aclanthology.org/D15-1263.pdf#page=3),
[Cornell polarity v2.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/),
[IMDb release](https://nlp.stanford.edu/~amaas/data/sentiment/),
[original preprocessing code](https://github.com/parry2403/R2N2/blob/c01253cf9bde553ed54ceca08be1dfe6db3a9578/RSTParser/adds.py).

### Taboada, Brooke, Tofiloski, Voll, and Stede (2011)

[_Lexicon-Based Methods for Sentiment Analysis_](https://aclanthology.org/J11-2001.pdf) presents SO-CAL, using lexical semantic orientation with modifiers such as negation and intensification. It is a direct Stede coauthored sentiment reference and a possible transparent local-scoring background, **not itself our four-standard comparison**. Choosing its exact dictionaries or software still requires checking availability and data provenance.

### Taboada, Brooke, and Stede (2009)

[_Genre-Based Paragraph Classification for Sentiment Analysis_](https://aclanthology.org/W09-3909/) distinguishes paragraph functions in movie reviews and uses them for sentiment extraction. It suggests useful controls for plot description versus evaluative text. Functional paragraphs are not interchangeable with RST relation labels.

### Huber, Xiao, and Carenini (2021)

[_W-RST: Towards a Weighted RST-style Discourse Framework_](https://aclanthology.org/2021.acl-long.302/) investigates continuous importance weights in place of categorical nuclearity, with downstream applications. It motivates separating relation effects from importance/nuclearity effects. Reproducing a weighted-RST framework is beyond the initial pilot.

## Consequences for the immediate choice

1. Gold document sentiment is sufficient for our downstream target; gold RST is a diagnostic option.
2. The closest lab precedent supports **EDU sentiment + RST composition**, so start from that mechanism rather than relying only on motif counts.
3. A neural encoder and RST are compatible: the former supplies sentiment vectors; the latter guides their combination.
4. Preserve a uniform aggregation control and a label-independent structural control.
5. Choose a relation-aware component for fine/coarse tests; hold its input scores and training protocol fixed.
6. Explain that we extend an existing line of work rather than discovering discourse sentiment analysis or relation granularity for the first time.
7. A short working reproduction of a baseline is a useful November 23 deliverable. A complete reproduction of the dissertation, all its datasets, or its CRF systems is not required to test our scoped extension.
