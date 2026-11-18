# Sentiment workstream — working plan, updated 2026-11-18

Status: **SFU corpus preparation and CPU text controls completed on 2026-11-18. No RST experiment has been run.** Current decisions are recorded at the top of [agent.md](../../agent.md); its older sections retain the repository history. See [RUNNING.md](RUNNING.md) for commands/results and [DECISIONS.md](DECISIONS.md) for the procedure log.

## 1. Goal and immediate scope

Max contributes sentiment analysis in parallel with the colleague's machine-text-recognition and argumentation work. The common question is how RST inventories and relation granularity affect downstream information. It is a representation study, not a claim to improve sentiment state of the art.

Max has explicitly confirmed that neither methodological novelty nor state-of-the-art sentiment results are needed. He acknowledges high-level choices; agents handle routine implementation and independent verification, with brief updates. Keep an explicit register of competing explanations and reproducible records so the eventual paper can accurately describe the procedure, including unsuccessful and null results.

Target is LInDA at EACL 2027, Athens. The [official dates](https://linda-workshop.github.io/dates/) list December 15, 2026 AoE for direct submission; some dates are qualified as ACL suggestions. The next team progress meeting is November 23; no time was supplied. Publication length/track remains undecided.

The sentiment/RST mechanism and lab precedent have been explained, and Max accepted proceeding along this line. Reuse an established small method. Prioritize reproducible controls and source alignment over a more sophisticated sentiment classifier. See [RELATED_WORK.md](RELATED_WORK.md) and the newer [model provenance audit](MODEL_PROVENANCE.md).

## 2. What the prediction task actually is

Input is a review; target is its supplied sentiment label. An existing discourse parser predicts EDU boundaries, a tree, nuclearity, and rhetorical relations. Those predictions are additional inputs to the sentiment model, not evaluation targets requiring a newly annotated gold treebank.

One possible pipeline is:

```text
review -> common EDUs -> fixed local sentiment scorer -> EDU sentiment vectors
                      -> inventory-conditioned RST tree -> aggregation of vectors
                                                        -> document sentiment
document's reference sentiment ----------------------------> training/evaluation
```

The sentiment scorer can be neural. RST organizes or weights its predictions rather than replacing sentiment vocabulary with relation names. A review can praise a screen yet condemn a battery, so document sentiment should not automatically be copied to every EDU as a training label.

Gold RST is optional for assessing parser errors or estimating an upper-bound-like diagnostic. It is unnecessary for the primary downstream comparison. Existing parsers have their own gold-treebank training provenance, which should still be recorded.

Use a matched uniform aggregation baseline: otherwise an apparent benefit could come from reading more text, processing shorter spans, or changing the sentiment encoder rather than from RST. A whole-review BERT baseline with truncation and an RST model reading every EDU do not have matched text access.

## 3. Keep the experimental factors separate

| Factor | Intended levels / status |
|---|---|
| Inventory | PCC; NLDT/RST-classic; GUM/eRST; RST-DT. Exact revisions must be logged. |
| Parsing regime | Shared UniRST machinery versus inventory-specialized checkpoints/backend. Not all combinations currently have located models. |
| Granularity | Fine labels versus a declared coarse projection on each existing tree. Refit downstream models for both. |
| Segmentation and scenes | A fixed shared policy for the primary comparison; PCC-based and GUM-based runs are separate conditions. |
| Sentiment representation | Same lexical/local sentiment signals; vary only the discourse information under test. |
| Sentiment aggregation/classifier | Fixed architecture/protocol across inventory and granularity conditions. |

The first complete study should favor one coherent parser regime across all four inventories, if its checkpoints can be obtained, rather than attempting a full four-by-two-by-two design immediately. Alternative backends can be a scoped sensitivity study.

### Available versus missing models

| Condition | Located implementation | Limitation |
|---|---|---|
| Stock UniRST | `tos/unirst.py`; defaults `tchewik/isanlp_rst_v3`, version `unirst`, four inventory keys. | Older GUM/RST-DT inventories are coarse. Defaults do not recreate full `.new` conditions. |
| Full eRST/RST-DT UniRST | Canonical analysis expects `.new` inputs; fine-tuning scripts and native tables were subsequently recovered in the pinned upstream dependency. | Actual trained artifacts/configs producing historical parses are still unidentified; top-level adapter does not expose the recovered fine-artifact loader. |
| Ettin GUM | Feature branch `62b6fbc`; `larc-iu/dmrst-ettin-400m-gum-12.1.0`, revision `e33906a27f89b2703df8ab19e471aa82141d62cb`. | GUM-based segmentation; native labels; separate parser training. |
| Ettin RST-DT | Same branch; `larc-iu/dmrst-ettin-400m-rstdt-coarse`, revision `26594833545ef713d14e2ea6e9a7670c881c143a`. | Coarse RST-DT checkpoint, not a matched fine RST-DT condition. |
| Ettin PCC/NLDT | No such adapters/checkpoints configured here. | Do not invent them or count the two-model integration as four standards. |

Ettin uses Dipo revision `4debc92e41140e5bcd9aadb987a297244c0fb14a`. Its November 16 conversion now imports `isanlp.annotation_rst.DiscourseUnit`, but `requirements-ettin.txt` does not list `isanlp`, and pinned Dipo's own dependency metadata also does not list it. Treat fresh-environment installation as unverified and check that dependency before running. This is a static dependency finding, not a performed install/failure test.

## 4. What can be reused from the colleague

| Component | Source | Action needed for sentiment |
|---|---|---|
| Shared EDU caching / parsing | `tos/unirst.py`, `make_document_spec` / `run_group` in the numbered preparation script | Add a review loader and stable document manifest; retain status/provenance. Avoid triggering default HC3/MAGE ingestion. |
| Ettin backend / sharding | Feature branch `tos/ettin.py` and preparation code | Optional later backend; distinct outputs, environment, common policy, and model IDs. |
| Document / scene representation | `tos/tos_dataset.py` | Reuse text/EDU/tree/graph fields; define sentiment labels explicitly. |
| Graph and M3 operations | `tos/tos_dataset.py`, `tos/tos_utils.py`, research `scripts/prepare/` | Reuse for a bridge/diagnostic experiment; build catalogs from training portions only. |
| Relation frequencies / structural features | Research `scripts/canonical/build_feature_matrices.py` | Extract generic aggregation logic; replace HC3 filenames, inventory constants, domain suffix logic, and paths. |
| Fine-to-coarse maps and invariants | Research `tos/analysis/granularity.py`, ablation modules, RST-DT mapping | Reuse only where labels actually match; check total assignment and frequency/depth conservation. New relation revisions require validated maps. |
| Model/evaluation principles | Research LR selection, paired comparisons, provenance | Adapt for review sentiment and folds. Do not inherit human-authorship positive-class naming or HC3 expected dimensions. |
| Relation-aware sentiment aggregation | Not present in this repo; closest sources are DASA/R2N2 | Port a small, inspected method to the common tree representation rather than reinstalling every legacy dependency. |

The recent feature code and canonical research code live on different branches. Read from exact commits or use a separate worktree/feature branch when implementation begins. A blind merge of the two layouts is unnecessary for the first pilot.

M3 distributions alone discard the alignment between sentiment-bearing text spans and rhetorical roles. They can test a useful style/structure association, but should not be the only evidence for discourse-sensitive sentiment composition.

## 5. Proposed small model ladder

This is a proposal, not a finalized experiment.

1. **Text control:** simple TF-IDF + LR as an interpretable baseline. A stronger BERT-like text model is optional after the mechanism works.
2. **Local-score controls:** same EDU scorer with uniform aggregation and simple positional controls. Add length/EDU/scene counts where relevant.
3. **Structure control:** nuclearity/depth aggregation with relation labels ignored.
4. **Primary relation-aware model:** an R2N2-style small aggregator or another explicitly relation-conditioned pooling model, trained against document sentiment. Run it with native fine labels and fixed-tree coarse labels.
5. **Optional bridge to HC3:** text plus relation/M3 features, to see whether the colleague's representation transfers even without compositional modeling.

An EDU scorer's weights, calibration, and input text must be shared across inventory conditions with shared EDUs. If it is fitted on this corpus, all fitting belongs inside the training fold; stacked training predictions need cross-fitting where required. Exact scorer choice is still open. No extra sentiment checkpoint has been selected or downloaded.

**Depth alone cannot test relation-label granularity on a fixed tree.** If coarsening only labels changes a genuinely label-independent depth baseline, investigate the implementation. A relation-aware branch is required for the actual fine/coarse question.

For a neural relation-conditioned model, unseen relations need an explicit training-only fallback. Fine and coarse models naturally have different parameter counts; use the same tuning budget, regularization/initialization policy, and report sparsity. A parameter-matched sensitivity is optional, not a reason to delay a small pilot.

### Competing explanations and checks to implement

This is a prospective verification register, not a list of completed experiments. Add new explanations when evidence warrants them and record which checks were run. The aim is to distinguish plausible accounts with a feasible set of controls; no finite study can exclude every possible explanation.

| Explanation | Discriminating check |
|---|---|
| No stable inventory/granularity effect | Paired held-out predictions, uncertainty, fixed folds, and domain-level consistency; preserve null and reversed results. |
| The apparent difference comes from text access or the sentiment scorer | Use shared EDUs, the same local scores and token coverage; compare uniform and position controls. |
| Tree organization/nuclearity explains the effect without relation labels | Label-independent depth/nuclearity control versus explicitly relation-aware aggregation. |
| Fine relation distinctions matter on otherwise identical parses | Compare declared coarse projection against native labels on fixed trees, preserving EDU text, attachments, nuclearity, and occurrence mass. |
| Extra dimensions/parameters or frequency patterns explain a fine/coarse gap | Track active labels, sparsity, parameter counts, tuning budget; use a predeclared frequency/cardinality-matched grouping or tying sensitivity where needed. |
| Parsing quality/backend, corpus revision, or segmentation explains an inventory gap | Log exact checkpoints/training scope/label tables; match input policy; keep alternative backends separate and describe outcomes as parser-conditional. |
| Discourse categories help only as document-level style cues | Compare relation/M3 distributions with sentiment-aligned relation features or composition; shuffle label locations within trees while preserving frequency where the model permits it. |
| Category spelling/order or cache corruption causes an effect | Test bijective renaming/permuted internal IDs with consistent parameter alignment; validate round trips, tree coverage, native labels, shard union/disjointness, and model/policy cache provenance. |
| Failures, data overlap, or small-domain composition explain a score difference | Audit retained IDs/labels, duplicate groups, failure rates by class/domain, train-only vocabulary/catalog/model fitting, and paired success cohorts. |

Minimum reproducibility record per run: source commits, effective model/file hashes, ordered label table and map, corpus/normalization hash, split IDs, seed, configuration, environment, per-document outputs/status, metrics, and exclusions. A small decision log should state the reason for each change and whether it preceded outcome inspection. Distinguish source checks, synthetic invariants, tiny integration runs, and real held-out evaluations in reporting.

## 6. Corpus choice: SFU first pilot locked on 2026-11-18

The [audit JSON](corpus-audit-2026-11-17.json) was generated by [audit_corpora.py](audit_corpora.py), reading both public archives in memory. No review text was committed. Lengths below are whitespace-token proxies, not parser token counts or speed measurements.

| Corpus | Documents / balance | Domains | Median / 95th percentile / maximum length |
|---|---|---:|---|
| SFU English raw reviews | 400: 200 negative, 200 positive | 8 | 572.5 / 1,908 / 6,149 |
| Cornell polarity v2.0 | 2,000: 1,000 negative, 1,000 positive | Movies | 696.5 / 1,355 / 2,678 |

First pilot: **SFU**, for small-scale parsing and domain analysis. Every domain has 25 documents per class. Its corpus/hash and folds were fixed before fitting the November 18 text controls. Cornell remains a possible larger replication with original ten-fold tags, not part of the completed run. Switching to the lab's German PotTS data would add language/domain alignment work.

Actual audit findings:

- No empty reviews or exact duplicate groups after whitespace normalization in either corpus; this is not a near-duplicate/author audit.
- SFU required Windows-1252 fallback for 84 of 400 files. A UTF-8-only reader would fail or corrupt some text.
- SFU's archive includes a nested `MOVIES/movies.zip`; count only the 400 top-level domain review files.
- A simple rating-expression heuristic fired in 20 SFU and 82 Cornell reviews. It is not a leakage determination. Decide any rating cleanup on training material and apply a fixed rule consistently.
- Cornell is lowercased and sentence-tokenized, with one sentence per line. The source's README documents residual rating limitations. Joining those sentence lines with spaces is preferable to misreading them as original paragraph boundaries.
- SFU's advertised RST annotations are sentence-level, not full-document gold. They are optional diagnostic material and do not replace predicted document parses.

Long reviews may become several scenes under the inherited token budget. Cache the exact scene/EDU manifest and share it across conditions; log how many documents are split and which discourse connections become unobservable. Do not call a collection of scene trees a full-document RST tree.

### Evaluation proposal for SFU

Use fixed outer five-fold cross-validation stratified by domain × sentiment, giving 80 test reviews per fold and five reviews per class per domain. Use training-only inner validation for hyperparameters. Preserve each review and any identified duplicate/author group within a fold. Save stable IDs independent of labels, checksums, and folds before inspecting outcome differences.

Primary metric: macro-F1 over the actual sentiment classes; secondary accuracy and class-wise F1. The inherited HC3 human-class F1 is not the new primary measure. Pair comparisons on the same reviews, report failures and retained cohorts, and distinguish fixed-tree fine/coarse inference from overall parser/inventory differences. If repeating folds/seeds, do not count repeated predictions for the same document as independent observations.

## 7. Request for the colleague — drafted, not sent

> For the sentiment track to match your experiments, could you share (1) the actual parsing commands and checkpoint IDs/revisions for each standard under UniRST and any separate models, including the full GUM/RST-DT checkpoints; (2) the shared EDU segmentation and document/scene-splitting policy used for each reported comparison; and (3) the final fine-to-coarse mappings plus the location of the current ArgMicrotexts code/configuration? Which conditions are completed runs and which are still planned?

Max does not currently have these details. This request is about the unavailable parsing/provenance details, not a claim that the existing canonical LR/RF/XGB setup is unknowable.

**Refinement after the dependency audit:** [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) gives a narrower request and the exact expected artifact bundle. Fine-tuning scripts, native label alphabets, and a pinned base model have now been located in the dependency; identifying the actual chosen weights/configs is the remaining issue. Confirm whether the committed downstream maps intentionally differ from the initialization maps and how to handle native labels absent from those maps. ArgMicrotexts code is useful context but is not a prerequisite for the sentiment baseline.

## 8. Sequence toward the next meeting

Completed November 18: deterministic corpus/folds/provenance, locked CPU environment, 14 passing validation tests, and three nested-CV text controls. An independent audit agrees with saved predictions/metrics. No RST effect has been measured. The remaining sequence below is a plan, not a completed schedule.

| Window | Concrete deliverable | Dependency |
|---|---|---|
| Nov 17–18 | Explain sentiment/RST mechanism, inspect lab precedent, select one corpus and aggregator, obtain parser manifest | Conceptual/model decision; colleague request can proceed in parallel. |
| Nov 18–19 | Corpus, folds, and prior/length/text controls complete; uniform/position EDU controls next | Exact local EDU scorer and common segmentation policy still need alignment. |
| Nov 19–20 | A small length-balanced parser/serialization/timing pilot; inspect a few trees | Exact selected model and segmentation policy. First sample selection should avoid tuning on correctness. |
| Nov 20–22 | One matched fine/coarse relation-aware pilot, coverage/timing report, and a few illustrative errors | Successful parser interface and maps. A pilot on few reviews is not a publishable accuracy estimate. |
| Nov 23 Zoom | Show what runs, measured cost, baseline/first comparisons if available, and remaining design decisions | Report actual progress; no guaranteed result or compute-time promise. |
| After Nov 23 | Freeze the feasible condition matrix, complete runs/uncertainty/analysis, write into the shared paper | Scope decision based on pilot, with drafting time before Dec 15. |

Negative/null results are informative about representations if controls and coverage are adequate. Do not keep expanding tasks, parsers, datasets, or model searches merely to obtain a positive result.

## 9. Reproducing only the corpus audit

Public archive locations:

- SFU: `https://www.sfu.ca/~mtaboada/docs/research/SFU_Review_Corpus_Raw.zip`
- Cornell: `https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz`

After obtaining both archives:

```bash
python research/sentiment/audit_corpora.py \
  --sfu /path/to/SFU_Review_Corpus_Raw.zip \
  --cornell /path/to/review_polarity.tar.gz \
  --audit-date 2026-11-17 \
  --output /tmp/sentiment-corpus-audit.json
```

The script is an archive audit, not a trained classifier. Raw downloads from this investigation remain in `/tmp` and are not portable. The recorded hashes allow checking restored copies.

## 10. Sources

- [SFU corpus owner page](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html): polarity/domain construction, raw archive, scope of RST annotations.
- [Cornell dataset owner page](https://www.cs.cornell.edu/people/pabo/movie-review-data/) and [v2 README](https://www.cs.cornell.edu/people/pabo/movie-review-data/poldata.README.2.0.txt): exact version, folds, preprocessing and rating caveat.
- [Bhatia et al. 2015](https://aclanthology.org/D15-1263/): background for discourse-aware sentiment aggregation; see RELATED_WORK.md.
- [Hewett et al. 2019](https://aclanthology.org/W19-4512/): the colleague's argumentation reference. It uses discourse features for central-claim, role, function, and attachment predictions with a structured decoder; it is not a sentiment model or a deterministic RST-to-argument mapping. Its simpler/fuller **argumentative** relation sets should not be confused with our RST-label granularity factor.
- [Current feature README](https://github.com/ruthenian8/threads-of-subtlety/blob/62b6fbc1f7cc3f6011e455f4134af3f3cf138309/README_ETTIN.md); historical canonical mechanics/results are pinned in agent.md.
