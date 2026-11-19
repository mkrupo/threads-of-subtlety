# Prior work to build on — checked 2026-11-17

Purpose: identify a feasible method and the actual research gap. This is a targeted search, not an exhaustive literature review or a novelty certification. The implementation plan is in [PLAN.md](PLAN.md).

## Prior classification on the English SFU reviews

- **Corpus and early classification.** Taboada and Grieve (2004) used the
  original collection of 400 English Epinions reviews. The corpus page
  confirms that the binary labels come from the reviewers' recommendations.
  [Early study](https://www.sfu.ca/~mtaboada/docs/publications/TaboadaGrieveAppraisal.pdf) · [Corpus](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html)
- **Discourse on the same corpus.** Voll and Taboada (2007, Table 3)
  compared scoring adjectives in automatically identified nuclei with
  scoring all adjectives. Balage Filho (2012, Chapter 4) used the corpus's
  sentence-level RST annotations to learn relation weights and reported
  results under four-fold cross-validation. [Voll and Taboada](https://www.sfu.ca/~mtaboada/docs/publications/Voll_Taboada_AusAI.pdf) · [Balage Filho](https://www.pedrobalage.com/assets/pdf/BalageFilho2012UseDiscourseKnowledgeMasterThesis.pdf)
- **Historical lexical results.** Brooke et al. (2009, Table 2) report
  76.50% accuracy for a unigram SVM with ten-fold cross-validation on the
  original Epinions collection. Taboada et al. (2011, §2.8 and Table 4)
  report 80.25% accuracy for full SO-CAL on that collection, called
  Epinions 1. Those texts also informed SO-CAL's dictionary and system
  development. Treat the scores as historical context for the matched
  SFU controls reported here. [Brooke et al.](https://aclanthology.org/R09-1010/) · [Taboada et al.](https://aclanthology.org/J11-2001/)

For the granularity question, the equal-weight EDU baseline holds the
local scorer and its inputs fixed. Nuclearity controls for tree prominence,
whole-review TF-IDF supplies a lexical reference, and log length checks
whether motif-only performance exceeds a basic document statistic.
[Paper section](paper/sentiment.tex) · [BibTeX](paper/sentiment.bib)

## Attribution for the completed SFU experiments

| Piece of the experiment | Source and what we use |
|---|---|
| Reviews and document labels | The SFU Review Corpus supplies 400 Epinions reviews. The target is the reviewer's recommendation. [Corpus source](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html) |
| Parser and fixed-tree comparison | UniRST comes from Chistova (2025). The shared PCC segmentation, the full GUM/eRST and RST-DT checkpoints, and mapping fine labels after parsing follow the colleague's HC3 setup. [UniRST paper](https://arxiv.org/abs/2510.06427) · [Checkpoint and map provenance](MODEL_PROVENANCE.md) · [HC3 checks](HC3_REPRODUCTION.md) |
| Sentiment scores | We chose the frozen English CardiffNLP Twitter RoBERTa model for its three outputs and use the pinned quantized Xenova export. We use its probabilities as EDU features. [Model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) · [TimeLMs](https://aclanthology.org/2022.acl-demo.25/) · [Exact export](results/sfu-edu-scores-v1.json) |
| Combining sentiment with nuclearity | Heerschop et al. (2011) weight sentiment using nuclearity and relations. Bhatia, Ji, and Eisenstein (2015, §3) use discourse depth to reweight EDUs or learn a recursive composition. Our simpler rule puts EDUs reached exclusively via nucleus branches into one group and the remaining EDUs into another. We feed separately averaged probabilities to review-level logistic regression. [Heerschop et al.](https://doi.org/10.1145/2063576.2063730) · [Bhatia et al.](https://aclanthology.org/D15-1263/) |
| Using relation names in sentiment composition | Bhatia et al. use relation-dependent weights for recursively combining children. Sidarenka (2019, §6.4.2) compares sentiment under several relation schemes by retraining the parser on PCC. Our variant averages the EDU probabilities in each left and right child span, pools those vectors by relation name, and fits logistic regression on the pooled features. We apply the coarsening to the same trees following the HC3 experiment. [Bhatia et al., §3.2](https://aclanthology.org/D15-1263.pdf#page=4) · [Sidarenka, §6.4.2](https://arxiv.org/pdf/1911.13062#page=177) |
| Motif comparison | Kim et al. (2024, §3.2) define discourse motifs, motif frequency, and weighted average depth. We follow the colleague's HC3 implementation for graph conversion, M3 counting, depth normalization, and fixed-catalog projection. We assemble a new catalog within each SFU fitting partition. [Kim et al.](https://aclanthology.org/2024.acl-long.298/) · [SFU procedure](RUNNING.md#hc3-style-motifs-on-sfu) |
| Review-level evaluation | Equal-weight EDU pooling, the whole-review TF-IDF control, the two group-averaging rules, and logistic regression on five frozen outer folds with training-only model selection are experimental choices in this study. The neural parser and sentiment scorer stay frozen. [Decision log](DECISIONS.md) |

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
