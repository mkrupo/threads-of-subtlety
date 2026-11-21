# Research handover: Threads of Subtlety and RST relation granularity

Prepared: **2026-11-12**. Historical investigation performed on 2026-11-11 and 2026-11-12. **Last updated: 2026-11-21**.

Repository: [ruthenian8/threads-of-subtlety](https://github.com/ruthenian8/threads-of-subtlety).

This is a self-contained research-context handover for an agent on another machine. It combines findings from the Git branches, source code, committed research reports, a locally supplied paper/Overleaf snapshot, planning notes, and email excerpts. The intended reader should be able to explain the project to Max without reconstructing its history from those sources.

The repository state described below is a dated snapshot. Preserve its distinction between observed implementation, reported historical outcomes, draft statements, and inference. If later commits change the situation, explain the change rather than silently replacing the historical account.

## Current sentiment handover — 2026-11-21

- **Branch and paper scope.** Start from `origin/sentiment2` on Max's fork. It descends from the frozen SFU snapshot `25499ac` and contains the Amazon and UCI experiments. The current paper direction is to keep HC3 as the main granularity result, use **SFU and Amazon** for review sentiment, and use UCI as an optional appendix check on very short inputs. The committed LaTeX sentiment section covers SFU. The meeting notes give the SFU–Amazon methods and table for integration into the manuscript. [Meeting notes](research/sentiment/replication/MEETING_NOTES.md) · [SFU LaTeX](research/sentiment/paper/sentiment.tex)
- **SFU reproduction.** On the RTX 4080 under WSL2, a fresh run from the raw archive reproduced 400 reviews, 1,012 scenes, 36,850 EDUs, and 4,048 final trees. The shared scenes, all final tree files, CardiffNLP probabilities, and all five discourse/EDU prediction files matched their original hashes. All 32 pooled macro-F1 values matched. The TF–IDF fitted models and text-control prediction JSONL differed at the byte level, with identical metrics and regularization selections. [Fresh SFU audit](research/sentiment/replication/results/sfu-reproduction.json)
- **Amazon extension.** We formed a fixed 2,000-review sample across four product categories with 250 reviews per category and polarity. Stars 1–2 and 4–5 supply the binary labels. The sample covers January 2022 through August 2023, following the scorer's documented pretraining cutoff. No full normalized body matched the published TweetEval sentiment training texts. All reviews fit in one scene, with a median of 48 words and eight EDUs. We reused the four parser configurations, frozen CardiffNLP scorer, and nested evaluation. A converter retry fix uses the parser's tokenization of joined EDUs. It recovered four trees and preserved the other 7,996 from the first retry. [Cohort and overlap provenance](research/sentiment/replication/results/amazon/cohort.json) · [SFU–Amazon report](research/sentiment/replication/results/amazon/comparison.json)
- **Review-level result.** Text-order relation features scored below nuclearity alone in all eight SFU conditions and above it in all eight Amazon conditions. Coarse NLDT went from −3.25 macro-F1 points versus nuclearity on SFU to +2.00 on Amazon, with pointwise paired 95% intervals of [−6.25, −0.25] and [+0.85, +3.15]. Coarse led fine in all four Amazon text-order conditions, with the NLDT gap of +1.30 points clearing zero at the pointwise level. Motif-only fine labels led coarse in three inventories on both corpora. All intervals for the ordering change from text order to nucleus first included zero. Length, sample size, domains, and label definitions changed together.
- **UCI follow-up.** We ran all 3,000 human-labeled sentences from the published release and grouped the 21 normalized duplicate-text groups within folds and bootstrap draws. Of the 3,000 sentences, 1,544 had one EDU and 928 had two. All 12,000 trees verified and all 96,000 held-out decisions were independently checked. On the 1,456 multi-EDU sentences, the models fitted on full folds scored .9141 for equal-weight pooling, .9210 for the best nuclearity condition, and .9196 for the best relation condition. The paired intervals for all positive relation gains on this subset included zero. This is a post hoc diagnostic of existing predictions. [Full UCI and three-corpus results](research/sentiment/replication/results/uci/comparison.json)
- **Reproduction on another machine.** The runbook covers exact raw inputs, model revisions, staging, and checks. The two fine checkpoints come from the collaborator. Each requires its original weights, config, and relation table. Reports provide their SHA-256 hashes. A fresh checkout needs this checkpoint bundle and the public raw/model inputs to run inference. [Runbook](research/sentiment/RUNNING.md)

## Repository cleanup — 2026-11-18

- Sentiment work is now on the dedicated local `feature/sentiment` branch, based on `dev` at `e658d857b1b3a803f91873c3a18258850a9c4e65`. Its initial local commit records the implementation, tests, locked environment, handover, protocol, and selected aggregate results. Use `git log feature/sentiment` to identify the commit; the earlier run manifests correctly retain their pre-commit source hashes.
- `RST_relation_granularity/` remains on disk as ignored local reference material, including the supplied email excerpts. Raw corpora, the virtual environment, fitted models, and per-document predictions also remain local and ignored. The two portable aggregate/verification JSON files are explicitly included despite the inherited broad `results/` ignore rule.
- Max's configured fork `mkrupo/threads-of-subtlety` is `origin`, the colleague's repository is `upstream`, and `feature/sentiment` tracks the pushed branch on Max's fork.
- The colleague supplied the fine-model checkpoints and recommended stock UniRST for PCC/NLDT, shared PCC EDUs, and the canonical research-branch maps. The downloaded checkpoints and canonical analysis artifacts are in the local `HC3_project/` directory.
- The four-model pilot passed both fixtures and 15 scenes from six SFU reviews. All 60 SFU trees and RS3 files were verified against 568 shared PCC EDUs. All eight HC3 LR MF+WAD test F1 scores were independently recalculated from the saved predictions. The canonical maps cover every pilot prediction, with gaps for one possible NLDT label and nine possible RST-DT labels. See [reproduction and outcomes](research/sentiment/RUNNING.md) and the [pilot report](research/sentiment/results/unirst-parser-pilot.json).

## Completed preparation — 2026-11-18

This update supersedes earlier statements that no sentiment classifier has been trained. The study goal remains differences between RST representations; text controls are preparation, not evidence of an RST effect.

- Max authorized parallel preparation while awaiting the colleague's trained-model/provenance reply. A bounded GPT-6.2 Sol worker implemented the corpus pipeline, and independent reviews checked the fold construction, leakage boundaries, and saved results.
- The first SFU pilot is now fixed to the audited archive: 400 unique reviews, eight domains, 200 per polarity. Canonical decoded text, content IDs, exact nested folds, and provenance are saved locally under `artifacts/sentiment/sfu-v1/prepared/`.
- Five outer folds each hold out 80 reviews; four inner validation folds within each outer training set each contain 80 reviews. Each test/validation fold has five reviews per domain and sentiment class.
- A minimal locked CPU environment and fixed `sfu-text-controls-v1` configuration were written before inspecting real baseline outcomes. No parser stack or neural sentiment checkpoint was needed.
- Pooled out-of-fold macro-F1: prior 0.3333; length-only LR 0.6166; TF-IDF unigram/bigram LR 0.8274. These are controls on the sentiment task, not framework/granularity results or a state-of-the-art claim.
- 14 fixture/integration/regression tests passed, covering encoding/duplicates, leakage/folds, training-only vocabulary fitting, metadata exclusion, Unicode JSONL, preparation inconsistency and output corruption. The run verifier checked hashes, identity coverage, and metrics; an independent standard-library audit recomputed the results and checked against raw archive text. Its [standalone verifier](research/sentiment/verify_artifacts.py) and [passing report](research/sentiment/results/sfu-text-controls-v1-independent-verification.json) are preserved.
- Sources and selected result/provenance snapshots are in `research/sentiment/`; raw text, fitted models, and per-document outputs are ignored under `artifacts/sentiment/`. Reproduction commands are in [RUNNING.md](research/sentiment/RUNNING.md); decisions made before scoring and subsequent fixes are logged in [DECISIONS.md](research/sentiment/DECISIONS.md).
- Max asked about using his own GitHub. Recommendation: a personal fork with `feature/sentiment`, `origin` pointing to his fork, and `upstream` pointing to the colleague. Reuse selected adapters/conventions, keep sentiment isolated, and integrate reviewed upstream commits. See [repository workflow](research/sentiment/REPOSITORY_WORKFLOW.md).
- At the end of this preparation stage the checkout was still on `dev` and the new work was uncommitted. The repository-cleanup update above supersedes that Git status. A fork by itself does not supply external trained artifacts.

## Current direction — 2026-11-17: read this before the historical account

This section supersedes older statements about the active goal, the latest feature-branch tip, and the absence of an agreed sentiment workstream. C01–C21 remain the detailed historical reconstruction at their recorded commits.

### Decisions reported by Max after the joint meeting

- Target: **LInDA, the Workshop on Linguistic Information for Downstream Applications, at EACL 2027 in Athens**. The official site currently lists direct submission on **2026-12-15, 23:59 UTC-12 / Anywhere on Earth**. Its dates page qualifies several dates as suggested by ACL; recheck before submission. The workshop itself is listed for March 13 or 14, 2027, still TBC.
- Next progress meeting: **Zoom on 2026-11-23**; meeting time was not provided.
- Scientific aim: compare **PCC, NLDT/RST-classic, GUM/eRST, and RST-DT**, and how their representations affect downstream tasks. Do not silently equate all revisions of NLDT with every possible classical RST inventory.
- Machine-generated-text recognition is one downstream testbed, with substantial existing code/results. The aim is to expose representation differences, not to claim a state-of-the-art detector or assume the publication is complete.
- The colleague continues the argumentation track on ArgMicrotexts, partly following [Hewett et al. (2019)](https://aclanthology.org/W19-4512/), as well as his other experiments.
- **Max's parallel contribution is sentiment analysis**, comparing the standards and fine versus coarse relation representations where the available models permit it.
- Keep parser family/training regime, annotation inventory, relation granularity, segmentation/scene policy, and downstream model as separate experimental factors. They are not a complete crossed design merely because branches mention them.
- Max wants short, understandable findings and decisions, with autonomous handling of straightforward research/preparation. He specifically asked how RST sentiment modeling works and whether gold parses are needed before choosing an implementation.
- In the follow-up, Max explicitly confirmed that a novel sentiment method or state-of-the-art sentiment score is unnecessary: downstream tasks are measurement testbeds for differences between RST standards and possibly relation granularity. Brief high-level decisions stay with Max; agents may independently inspect and validate implementation details, using subagents where useful.
- Reproducibility, competing explanations, and a record of how the study proceeded are priorities. Maintain pinned inputs/models, saved folds/configurations/seeds, controls, failures and null findings, and a decision log; do not select conditions solely because they produce positive results.

Official sources: [workshop CFP](https://linda-workshop.github.io/cfp/), [dates](https://linda-workshop.github.io/dates/), checked 2026-11-17. The CFP requires both a linguistic resource/annotation component and an assessment of its downstream impact.

### Repository refresh actually performed

`git pull --ff-only` was run on the existing `dev` checkout. `dev` was already current at `e658d857b1b3a803f91873c3a18258850a9c4e65`. Fetching during the pull advanced only `origin/feature/ettin-dmrst` among the visible branches:

- New tip: **`62b6fbc1f7cc3f6011e455f4134af3f3cf138309`**, 2026-11-16, replacing temporary tree nodes with `isanlp.annotation_rst.DiscourseUnit` in `legacy_result`.
- Earlier new commit: **`dfd5c6c`**, 2026-11-13, deterministic indexed data shards, independent of GPU IDs, with shard provenance and HC3 domain regrouping before applying split quotas.
- Native multi-document model batching is still not implemented by the pinned Dipo API. Independent processes/shards are a separate capability.
- No new committed sentiment or ArgMicrotexts experiment was located. The old research branch remains at `f06309b62cbc53dce0c98668cac225666dfb64d7`.
- This establishes visible remote activity, not which private checkout the colleague used most recently. The current local branch remains `dev`; no branches were merged or switched.

### Model alignment remains the main external dependency

The available conditions are stock four-inventory UniRST, reported full-inventory eRST/RST-DT UniRST adaptations, and two separately trained Ettin checkpoints (GUM and coarse RST-DT). The two Ettin models do not provide a four-standard fine/coarse design. “Unified versus individual parsers” is a useful informal distinction, but not a sufficiently precise run specification.

**Max confirmed on 2026-11-17 that he does not have the colleague's exact run commands/checkpoint list.** Record this as an open request; do not keep asking Max for material he has said he lacks. Obtain the exact checkpoint IDs/revisions and common segmentation/scene policy from the colleague before a comparable full parsing run. Corpus auditing and baseline preparation can proceed independently. No message has been sent to the colleague by the agent.

**Further recovery in the same follow-up:** the pinned dependency `ruthenian8/isanlp_rst` at `c51480a10a6636ae621fcc4676b12c0099f7a0f9` contains GUM/RST-DT fine-tuning scripts, ordered native label tables, warm-initialization maps, and a fine-artifact loader. Its scripts pin base model revision `9407970f1d9d2435b5f875a0cd14293a16646304`. The missing item is therefore primarily the actual trained artifact/configuration used for historical parses, not an entirely unknown training method. Those defaults do not prove historical run identity: they allow head-only/full-model modes and default to twenty epochs, while the draft describes ten.

The new [model provenance audit](research/sentiment/MODEL_PROVENANCE.md) records exact paths and the narrow artifact request. Static checks found that the canonical RST-DT map covers 60 of the dependency's 69 native relation names, and differs from warm-initialization parents for `preference`, `reason`, and `restatement`. These are different-purpose maps; differences require explanation or a declared sensitivity, not an automatic bug claim. The [audit script](research/sentiment/audit_model_provenance.py) and [results](research/sentiment/model-provenance-audit.json) preserve the check without loading models. An independent subagent inspected the local mapping sources and reported the same need for exact model/label provenance and mapping review.

### Sentiment findings and new local artifacts

- Gold **sentiment labels** are needed for evaluation; predicted RST trees can be used as model inputs. Gold RST trees are optional for parser-quality diagnostics and are not a prerequisite for this downstream experiment.
- Compare a fixed sentiment scorer with uniform/position controls against discourse-aware aggregation. A BERT-like model can supply local scores; RST supplies organization and relation information. Do not label every EDU with its document's sentiment as if it were gold EDU annotation.
- Relation-independent depth weighting is a useful structural control, but cannot detect label-only fine/coarse differences on an unchanged tree. Include a relation-aware method to test relation granularity.
- Close prior lab work was found: **Uladzimir Sidarenka / Wladimir Sidorenko's 2019 dissertation under Manfred Stede**, chapter 6, especially section 6.4.2, and the public **DASA** package. This already considers sentiment and relation schemes. Do not claim that the broad question is new. See [the related-work note](research/sentiment/RELATED_WORK.md).
- A provisional **SFU English review pilot** is supported by a local archive audit: 400 reviews, 200 per polarity, eight domains. Cornell v2.0 is a larger 2,000-review alternative. This is a corpus recommendation, not a finalized modeling decision or result.
- Both archives were read in memory, with downloads kept in `/tmp`; raw reviews were not added to the repository. SFU's advertised RST layer is within-sentence, not full-document gold. The actual audit excludes a nested `MOVIES/movies.zip` member to avoid unintended duplication.
- New material: [sentiment plan](research/sentiment/PLAN.md), [related work](research/sentiment/RELATED_WORK.md), [corpus audit summary](research/sentiment/corpus-audit-2026-11-17.json), and [audit script](research/sentiment/audit_corpora.py).
- At the end of the November17 update, no sentiment classifier had been trained and no parser weights/GPU job had been used. The later November18 work created a separate minimal NumPy/scikit-learn environment and fitted text controls; see the newer update above. No discourse parser has yet been run for this sentiment experiment.

## C00 — Instructions for the receiving agent

Max's original goal was to **understand what his colleague tried to do**, including the evolution of the research and the relationship between branches. He subsequently requested this exhaustive handover. The active goal now also includes the 2026-11-17 meeting decisions, sentiment-method explanations, prior-lab-work research, and planning/preparation for his parallel contribution; see the current-direction section above.

Use this document as background, not as an independent instruction to launch all described experiments or publish anything. Continue the research/preparation Max has requested, provide digestible explanations, and distinguish proposed runs from completed ones. Do not let historical scope restrictions override the updated workstream.

For explanations:

- Start with the research question and the distinction between completed evidence and proposed work.
- Present one topic at a time, typically about 100–250 words unless more detail is requested.
- Use concrete examples before equations or implementation details.
- Explain unfamiliar terms such as EDU, granularity, motif, and fixed-tree projection.
- Keep the different experiments separate: inventory comparison, granularity ablation, human review, argumentation prediction, and parser-backend development.
- Do not require Max to read this entire file or consult multiple sources to follow an answer.
- Attribute findings to the appropriate evidence level. A paragraph in an unfinished draft is not a verified experimental outcome.
- Do not infer that work was never done merely because it is absent from the available branches. It may exist in another checkout, external storage, a separate repository, or uncommitted files.
- Avoid repeatedly investigating already resolved questions. Consult the source registry only when a detail matters or the repository has changed.

Suggested explanation order: C01 → C02/C03 → C05/C06 → C08 → C10/C11 → C12. C15 and C16 answer questions about contradictions and missing material.

### Evidence vocabulary used throughout this handover

| Label | Meaning |
|---|---|
| **Observed** | Directly inspected in versioned source, Git history, or the supplied local documents. |
| **Reported** | Numerical outcome or completed-action claim recorded in reports, plans, drafts, or mails; not independently recomputed during this investigation. |
| **Inference** | Reconstruction connecting several sources; plausible, but not directly established by a complete implementation or result bundle. |
| **Unknown / unavailable** | The inspected material does not settle the question. |

These labels are contextual. For example, it is observed that a report contains a result, but the underlying result remains reported until the raw artifacts are restored and checked.

## C01 — The project in one page

**The central question is whether distinctions between rhetorical relations, often removed by coarsening RST annotation inventories, matter for downstream uses and for interpreting discourse.**

The original *Threads of Subtlety* project used discourse motifs to help distinguish human and machine-generated writing. Daniil Ignatev's extension began by asking whether using richer RST relation inventories would expose useful distinctions hidden by the conventional coarse labels.

The initial plan was ambitious: establish a downstream benefit, use LLMs to enrich existing annotated corpora, train finer-grained parsers, and test those parsers again. The available implementation supports a narrower and more developed quantitative contribution: a controlled HC3 experiment comparing four inventories and fine-versus-coarse features extracted from fixed parses.

The project then broadened toward linguistic evaluation: different RST outputs may describe different defensible readings of a passage, even when an exact-match evaluation calls one wrong. Human review was prepared and later discussed in email, but the completed annotation analysis is unavailable here.

Argumentation prediction on ArgMicrotexts was proposed as a second downstream task, partly because HC3 reflects older model behavior and UniRST parsing was too slow for easy expansion to large datasets. Notes and a separate draft fragment contain preliminary observations, but no committed argumentation experiment was found.

The newest branch replaces the parsing backend with Dipo/Ettin checkpoints. That is an engineering development with changed models and segmentation, not further evidence from the canonical granularity experiment.

The strongest supported claim is:

> On matched HC3 documents, some fine relation distinctions retain predictive information lost when labels are merged on the same trees. The size and statistical support of the benefit depend on the inventory, representation, and classifier.

The available material does **not** establish that more labels are always better, that one RST theory is linguistically superior, that parser disagreements are generally valid alternatives, or that a completed new LLM-reannotated corpus was produced.

## C02 — Snapshot, portability, and how to inspect other branches

### Exact branch state inspected

The first investigation checked the live GitHub branch list on 2026-11-11 and found exactly these four branches; their tips matched the local remote-tracking references. The local references were rechecked on 2026-11-12. The later local recheck was not a new live GitHub branch enumeration.

| Branch | Full commit SHA | Last commit in this snapshot | Role |
|---|---|---|---|
| `main` | `ec72df60f09138835076dce766c12e07c6123eff` | 2025-03-18, `Update README.md` | Original project implementation and README. |
| `dev` | `e658d857b1b3a803f91873c3a18258850a9c4e65` | 2026-11-10, `Batch UniRST segmentation during preprocessing` | Evolving UniRST preprocessing/motif pipeline. |
| `research/unirst2-canonical` | `f06309b62cbc53dce0c98668cac225666dfb64d7` | 2026-11-10, `enable batch processing` | Consolidated research analysis, reports, controlled ablations, review tools. |
| `feature/ettin-dmrst` | `696aa60e70dc194303010844ab4e542b10729ffa` | 2026-11-11, `Keep fitting documents intact for Ettin parsing` | Dipo/Ettin backend derived from `dev`. |

The inspected working tree was on `dev`. The local paper directory `RST_relation_granularity/` was untracked. That matters: **a normal clone does not contain those supplied paper, note, and mail files at these commits**. This handover preserves their substantive content so their absence does not prevent explanation.

Likewise, a normal clone initially checks out the repository's default branch, which was `main` in the inspected snapshot. That branch alone does not show the colleague's research analysis.

### Branch ancestry

The shared base immediately before the major research split was:

`beff52a38d16446e78ab65b6cd0de639ad0f4783` — 2026-09-13, `parse hd`.

```text
main: ec72df6
  |
  +-- UniRST development, August–September 2026
       |
       +-- beff52a (September 13)
            |
            +-- research/unirst2-canonical
            |    September 24 consolidation, October preparation changes,
            |    November 10 batching; tip f06309b
            |
            +-- dev
                 October 30 StoryScope, November 10 batching; tip e658d85
                  |
                  +-- feature/ettin-dmrst
                       November 11 backend integration; tip 696aa60
```

This is a schematic ancestry view, not a full commit graph. The important fact is that the research work was **not merged into `dev`**, and the latest feature branch was built from `dev`, not from the canonical research branch.

### Read-only inspection commands

These commands work without checking out another branch if the corresponding objects have been fetched:

```bash
git status --short --branch
git branch -a -vv
git log --all --oneline --decorate --graph
git show f06309b62cbc53dce0c98668cac225666dfb64d7:README.md
git show f06309b62cbc53dce0c98668cac225666dfb64d7:UNIRST2_RESEARCH_REPORT.md
git show f06309b62cbc53dce0c98668cac225666dfb64d7:PAPER_RELEVANT_RESULTS_TABLES.md
git show 696aa60e70dc194303010844ab4e542b10729ffa:README_ETTIN.md
```

If a clone lacks remote branches, `git fetch origin` is the normal retrieval step when network access is available. Do not assume `git show origin/research/unirst2-canonical:...` can work in an unfetched single-branch clone.

Prefer object inspection for explanation. Do not switch the user's working branch, merge branches, or overwrite local artifacts merely to read the research.

This handover itself was created as a root-level `agent.md`. It only accompanies a future clone if someone commits and pushes it or transfers it separately. Its creation did not itself publish it.

## C03 — Research chronology and scope changes

### Phase A: Original method, 2024 paper / original repository through March 2025

The original project is *Threads of Subtlety: Detecting Machine-Generated Texts Through Discourse Motifs* (ACL 2024). Its method uses DMRST parsing, graph conversion, discourse motifs, and classifiers, including Longformer with text and motif features. HC3 and MAGE appear in that workflow.

The original code and citation are inherited. Their presence does not mean the colleague completed an exact reproduction under unchanged data, parser, features, and metrics.

### Phase B: UniRST adaptation and scaling, August–September 2026

The history shows substantial work replacing the parsing stage, sharing segmentation, making runs resumable, adapting graph/motif scripts, optimizing M6/M9 operations, pruning motifs, persisting classifier outputs, and adding Human Detectors parsing.

Representative commits:

| Date | Commit | Subject / interpretation |
|---|---|---|
| 2026-08-18 | `3ec47a6` | `support unirst`; subsequent same-day patches address segmentation, caching, resumability, and parsing. |
| 2026-08-19 | `37e67cf` and following | Extract motifs from new outputs; improve M3/M6/M9 calculations and distribution handling. |
| 2026-08-22 | `479a236` | `prune extracted motifs`. |
| 2026-08-24 | `13aa34b` | `persist classifier outputs`. |
| 2026-09-13 | `beff52a` | `parse hd`; common base of later research and development lines. |

Do not interpret every surviving pipeline capability as part of the final reported research. M6, M9, Longformer, MAGE, and external detector studies remained available while the canonical scope narrowed.

### Phase C: More focused representation and granularity study

The historical plans show a move away from optimizing a detector toward understanding overlapping and distinct representations of coherence. Classifier performance became supporting evidence rather than the sole endpoint.

The research branch was consolidated on 2026-09-24:

| Commit | Subject |
|---|---|
| `9eac565` | Exclude generated corpora and local research artifacts. |
| `d080c17` | Prepare aligned UniRST HC3 inputs. |
| `8ea59ae` | Add reproducible four-inventory analysis pipeline. |
| `2a631a1` | Preserve qualitative review tools. |
| `c5307ae` | Archive legacy scripts and side studies. |
| `4d55fd0` | Publish canonical results and workflow. |
| `21ca1ce` | Scope test discovery to the active suite. |
| `528c136` | Improve repository navigation and provenance. |

The research branch's November tip therefore does not mean all of its experiments were newly designed in November. Most of the organized analysis was committed in September.

### Phase D: Human judgments and an additional downstream task

The supplied notes refer to a discussion with Massimo on **22.10**, without explicitly writing a year. In context, October 22, 2026 is plausible, but the date should not be treated as independently established.

The notes and mails describe a need to broaden the paper beyond old machine-text detection data and consider argumentation prediction. One mail says processing of their annotations had completed, with shared-subset agreement of `0.42` and an intended emphasis on relaxed evaluation of UniRST outputs.

The mail excerpts are undated. Their order in `mails.txt` is not a reliable timestamped sequence, and there are no full headers. Do not derive a precise chronology from their order alone.

### Phase E: More data and faster parsing, October–November 2026

- `dev`, 2026-10-30, `0d49e08`: add StoryScope ingestion.
- Research branch, 2026-10-26, `f9f8d3f`: preparation updates.
- Research branch, 2026-11-10, `f06309b`: batch-processing support.
- `dev`, 2026-11-10, `009b72d`: batched UniRST parsing.
- `dev`, 2026-11-10, `e658d85`: batched segmentation.
- Feature branch, 2026-11-11, `67a1b55`: integrate Ettin DMRST with shared GUM segmentation.
- Feature branch, 2026-11-11, `696aa60`: retain whole documents when they fit the GUM encoder window.

These are processing developments. No new downstream accuracy, speed benchmark, or completed argumentation result follows simply from these commits.

## C04 — Terminology and distinctions needed to explain the project

| Term | Meaning in this project |
|---|---|
| RST | Rhetorical Structure Theory: analysis of how spans form a coherent discourse through relations and hierarchical organization. |
| Annotation scheme / inventory | A particular set of relation labels and associated annotation conventions. These are related but not identical concepts. |
| Granularity | How many distinctions an inventory preserves; for example, distinguishing several kinds of elaboration rather than using one broad label. |
| EDU | Elementary discourse unit: a small text span, often clause-like, used as a leaf in the discourse analysis. It is not necessarily a full sentence. |
| Segmentation | Deciding the EDU boundaries. Different schemes can have different conventions. |
| Tree topology / attachment | Which spans are grouped and which connect to which. Sharing EDUs does not force identical trees. |
| Nuclearity | Whether a span functions as a nucleus or satellite; multinuclear relations connect nuclei. |
| PCC | Potsdam Commentary Corpus / its annotation conventions; inventory key `deu.rst.pcc`. |
| NLDT | Dutch Discourse Treebank / the RST Website-derived relation scheme; key `nld.rst.nldt`. |
| eRST / GUM | The eRST scheme as used in GUM. Original UniRST condition: `eng.erst.gum`; full-inventory research condition: `eng.erst.gum.new`. |
| RST-DT | RST Discourse Treebank and its scheme. Original UniRST condition: `eng.rst.rstdt`; full-inventory research condition: `eng.rst.rstdt.new`. |
| UniRST | The parser used to predict different inventory-conditioned analyses from shared EDU inputs. |
| DMRST | The original parsing family/backend used in the inherited project; the new Ettin branch also exposes DMRST models through Dipo. |
| Dipo | The library used by the feature branch to load and run the Ettin checkpoints. |
| Motif | A recurring small directed graph pattern extracted after converting a discourse tree to the project's graph representation. |
| M3 | A three-node motif in the converted graph; not three sentences and not necessarily three EDUs. |
| M6 / M9 | The original workflow's larger compositions of M3 motifs. Treat the names as pipeline terminology rather than a promise about independent text spans. |
| MF | Motif frequency: the document's occurrence count for a motif divided by all counted M3 occurrences in the canonical representation. |
| WAD | Weighted average depth: a motif's location/depth summary, based on the scene-level graph calculation and occurrence-weighted document aggregation. |
| LR / RF / XGB | Logistic regression / random forest / XGBoost classifiers. |
| Silver annotation | Automatically predicted analysis treated as an experimental input, not independently validated gold. |
| Fixed-tree projection | Merging fine relation labels on already predicted trees, rather than reparsing with another model. |
| Common cohort | The documents retained in every inventory condition and ordered identically for a valid paired comparison. |
| Identity-clean sensitivity | An analysis removing training-overlap identities and consolidating repeated test identities under the implemented identity definition; not a guarantee against every kind of data dependence. |
| ARG | The argumentation annotation layer in ArgMicrotexts, distinct from its RST and SDRT layers. |

Important naming points:

- `deu` and `nld` in inventory keys refer to the inventory/treebank provenance. The canonical downstream texts are English HC3 texts; this is not a comparison of German HC3 against Dutch HC3.
- `unirst2`, `data/unirst2`, and `.new` are project naming conventions for aligned inputs and full-inventory conditions. Do not infer a separate released architecture called “UniRST v2” from the directory name alone.
- A richer relation inventory is only one aspect of an annotation scheme. The canonical experiment deliberately constrains segmentation and uses only the primary predicted eRST tree, not all eRST secondary edges or signal annotations.

## C05 — Starting hypothesis and the original larger proposal

The supplied `Towards_granular_corpora.md` begins with the argument that existing RST annotation schemes preserve different subtleties, while computational parsing often collapses those subtleties into coarse labels. The notes suggest that better parsers and relation-sensitive downstream tasks make it worthwhile to revisit that compromise.

Examples in the notes include:

- Distinguishing `list`, `conjunction`, and `joint`, potentially exposing enumeration patterns in generated answers.
- PCC's `elaboration` versus `e-elaboration`.
- NLDT's volitional versus non-volitional causal distinctions.
- eRST organization relations for formal text.

These examples motivate the study; their theoretical definitions and parser correctness were not independently validated here. In particular, do not treat every example of an inventory's purported distinction as a completed empirical finding.

The original proposed research had three parts:

1. **Downstream motivation:** extend *Threads of Subtlety* to test fine relations, favoring interpretable classifiers.
2. **Corpus enrichment:** map existing labels to candidate fine labels, ask an LLM to choose using codebook definitions, validate against humans, and convert a corpus.
3. **Optional parser improvement:** train on the new enriched corpus and evaluate downstream benefit again.

The conversion example was a coarse `elaboration` annotation narrowed to candidates such as `elaboration-additional`, `elaboration-general-specific`, and `elaboration-part-whole`. The LLM would choose among candidates rather than produce an unconstrained full discourse tree.

The notes considered several possible corpus/layer combinations and did not fix one consistently. They mention adding PCC-style labels to GUM/RST-DT, adding RST-DT/eRST-style labels to PCC, and an option involving “UNCS.” The identity and adoption of that last option were not resolved by this investigation. Do not invent a finalized target corpus.

The proposed validation included a small human sample, pairwise relation choices, and reconstruction/visual inspection of trees. An example size of ten trees is part of that proposal, not the later sixty-case human-review package.

### Status of the original proposal

The inspected branches contain no completed LLM conversion pipeline or released enriched corpus supporting the proposed contribution list. The later note says “slash parts 1 and 2” and evaluate existing parser outputs. That is evidence of scope revision, but cannot literally mean all HC3 work vanished: its implementation and results clearly remain in the research branch and paper.

The safest reconstruction is that the ambitious corpus-creation narrative was reduced or displaced by analysis of existing/predicted representations, while the document was not comprehensively rewritten.

Do not confuse two distinct activities:

- **Proposed:** using LLMs to create new fine-grained annotation layers.
- **Reported in the actual draft:** recovering original full label inventories in existing training data and fine-tuning parser checkpoints with a coarse-to-fine initialization recipe.

The latter does not demonstrate completion of the former.

## C06 — The implemented canonical experimental design

### Research questions recorded in the canonical tables

1. Do M3 motifs carry authorship information beyond document structure and marginal relation frequencies?
2. Does fine relation resolution preserve predictive information lost under fixed-tree coarse projection?
3. Are motif associations directionally stable across HC3 domains, and are presence, frequency, and depth distinguishable?
4. Which effects overlap across standards, and which depend on inventory-specific distinctions?

### Data and split alignment

The canonical experiment is restricted to HC3. It is not a reported combined HC3/MAGE, StoryScope, Human Detectors, or OOD study.

Reported matched cohort:

| Split | Documents |
|---|---:|
| Training | 43,322 |
| Validation | 968 |
| Test | 1,932 |
| Total | 46,222 |

The analysis aligns to released paper split identities, intersects successful inputs across four inventories, checks labels, and orders rows identically. The raw per-inventory success cohorts can be larger and differ slightly. Historical intermediate counts are not competing canonical totals.

Identity in `build_feature_matrices.py` is based on SHA-256 of canonical JSON containing `[text, source, label]`, plus a per-split occurrence suffix for `identity_id`. Thus:

- `identity_hash` and `identity_id` are different fields.
- Duplicated content with different source or label can fall outside this exact identity definition.
- An identity-clean sensitivity is useful, but should not be described as exhaustive prompt/template/semantic decontamination.

### Shared segmentation

The default UniRST preprocessing uses `deu.rst.pcc` for segmentation. EDUs are persisted and reused through `from_edus()` rather than recomputed independently for each inventory.

The canonical data selects a verified shared-segmentation view retaining established PCC/NLDT inputs and aligning the full eRST/RST-DT inputs to that target. The canonical tables report **49,360 recoverable scenes with exact EDU matches and no boundary mismatches** for the verification described there. Scenes and documents are different counting units.

Shared segmentation controls an important source of variation, but does not assert that PCC segmentation is theoretically neutral or universally optimal. It is a deliberate experimental choice.

### Two different kinds of comparison

| Comparison | Held fixed | Allowed to differ | What it addresses |
|---|---|---|---|
| Across inventories | Documents, labels, split identities, shared EDUs | Relation predictions and inventory-conditioned tree building | Consequences of different parser/inventory representations. |
| Fine versus coarse within one inventory | Documents, EDUs, existing trees, attachments, nuclearity handling, selected fine catalog coverage and motif occurrence mass | Relation labels and the resulting merged feature dimensions | Information lost by the specified label-merging operation. |

Never say “all four inventories use the same trees.” They use the same segmentation in the canonical comparison. The fixed-tree guarantee belongs to each within-inventory granularity ablation.

### Parser fine-tuning described in the paper

The draft says stock UniRST retained relatively fine PCC/NLDT labels but coarse GUM/RST-DT labels. To obtain full eRST and RST-DT predictions, the authors reportedly:

- Recovered original fine labels in the existing GUM/RST-DT training splits.
- Initialized fine classification heads from the trained coarse parent heads, rather than randomly.
- Fine-tuned separate checkpoints for ten epochs.
- Used the predicted outputs as silver annotations.

Reported parser F1 values in the draft:

| Condition | Full-inventory checkpoint | Corresponding coarse model |
|---|---:|---:|
| GUM/eRST | 0.422 | 0.484 |
| RST-DT | 0.462 | 0.552 |

These are parser-evaluation scores, not HC3 authorship F1 and not human-review agreement. The exact metric variant and complete checkpoint-training/release details need the missing appendix/training material; do not overinterpret this short draft description.

The experiment clearly expects `.new` full-inventory data, but the preparation adapter's generic defaults still name the older stock inventories and `tchewik/isanlp_rst_v3` / version `unirst`. Merely running that preparation command with defaults does not establish that the canonical fine-tuned conditions were recreated. The complete parser-training workflow was not found in this repository.

## C07 — Features and model construction

### Graph and motif representation

`ToSDataset.create_graph_from_const_format` converts the parser's span representation into a directed graph. It creates span nodes, structural links labeled `/`, and relation-labeled links between relevant spans. The draft characterizes this through a hypergraph/Levi-graph conversion.

The operative motif matching and hashing use directed edges and their `label_0` relation attributes. The inspected isomorphism helpers do **not** match the node nuclearity attributes as a separate criterion. The review code explicitly notes that motif identity pools nuclearity variants. Direction and graph shape can still encode information associated with nuclearity, so “all nuclearity information is absent” would also be too strong.

Structural `/` links matter when interpreting graph overlap. A high Jaccard over all exported links can reflect shared scaffolding and must not be presented as pure relation agreement or constituent agreement.

### Canonical representations

| Representation | Implementation meaning |
|---|---|
| Structural | Five features: text character count, count of valid parsed scenes, EDU count, graph node count, graph edge count. This is a deliberately small baseline and includes length information. |
| Relation frequency | Counts of relation-labeled edges normalized by all discourse-relation edges in the document; excludes `/` structural edges. Vocabulary is learned from training data. |
| M3 presence | Indicator that a selected motif occurs at least once across document scenes. |
| M3 frequency / MF | Sum of raw motif occurrences across scenes, normalized by all selected M3 occurrence counts for that document. |
| M3 depth / WAD | Occurrence-weighted scene-normalized motif depth aggregated over the document. Zero is used when absent; presence allows absence to be distinguished in the analysis. |
| MF+WAD | Per-motif frequency and depth columns interleaved for the dedicated LR/RF fine-versus-coarse comparisons. |

Catalogs are training-only and then fixed for validation/test. Do not re-extract a test-driven catalog and call the result the same experiment.

### Feature dimensions

| Inventory | Canonical key | Relation-frequency columns | M3 motifs |
|---|---|---:|---:|
| PCC | `deu.rst.pcc` | 26 | 70 |
| Full eRST | `eng.erst.gum.new` | 33 | 90 |
| Full RST-DT | `eng.rst.rstdt.new` | 61 | 161 |
| NLDT | `nld.rst.nldt` | 27 | 74 |

The total is 395 inventory-specific M3 motifs.

**Important counting detail:** `build_feature_matrices.py` appends an `__OTHER__` column to the training-derived relation vocabulary. Therefore 26/33/61/27 are feature dimensions, not directly the numbers of distinct theoretically defined or observed training relation labels. The represented training vocabularies have one fewer ordinary relation name in each case. An unseen validation/test label can contribute to `__OTHER__`.

The draft separately discusses theoretical inventories of 78 RST-DT, 32 NLDT, 31 PCC, and 32 eRST relations, as well as the label sets supported by particular parser resources. Those are different counts for different purposes. Do not conflate them.

### Model selection

General representation comparisons use LR and XGBoost. Dedicated granularity comparisons additionally use Kim-style RF.

Observed LR setup:

- `StandardScaler` followed by logistic regression.
- Balanced class weights, `max_iter=5000`.
- Candidate `C` values: `0.01`, `0.1`, `1.0`, `10.0`.
- Candidate selection by validation ROC-AUC, with average precision as tie-breaker and deterministic parameter ordering for exact ties.
- Decision threshold selected on validation **balanced accuracy**, not on test data and not directly on validation F1.

Observed XGBoost grid:

- `max_depth` in `{2, 4}` and `min_child_weight` in `{1, 5}`.
- 300 estimators, learning rate 0.05, row/column subsampling 0.8.
- Training-label-derived positive-class weighting, histogram tree method.
- Same validation model/threshold selection framework.

Observed dedicated RF granularity setup:

- `RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=...)`.
- MF+WAD features, fixed threshold 0.5.
- No validation-driven RF hyperparameter search in that dedicated implementation.

The primary reported classification metric is **binary F1 for label 1, the human class**. Label 0 is generated text. The code also calculates other metrics, but do not silently substitute macro-F1, generated-class F1, accuracy, or ROC-AUC for the reported F1.

### Difference from the original Longformer method

The canonical classifiers do not receive text embeddings. The original Longformer receives text and motif features together. In the inspected `tos/tos_models.py`, the motif vector is concatenated with the first-token hidden representation at the classification head. The notes' phrase “prepended discourse features” should not be treated as a literal description of prepended feature tokens.

The legacy Longformer default motif dimension is based on the original M3/M6/M9 catalog, not the four canonical M3-only catalogs. Old classifier weights are not valid for arbitrary new feature dimensions.

## C08 — Reported quantitative results and their proper interpretation

All tables in this section are **transcribed reported outcomes** from the canonical branch. The raw input bundle and full generated run were unavailable in the inspected checkout, so this investigation did not refit the models or independently regenerate these outcomes.

### Representation comparison: HC3 human-class test F1

| Inventory | Model | Structural | Relation frequency | M3 presence | M3 frequency | M3 depth |
|---|---|---:|---:|---:|---:|---:|
| PCC | LR | 0.678 | 0.624 | 0.723 | 0.768 | 0.703 |
| PCC | XGB | 0.699 | 0.774 | 0.748 | 0.796 | 0.756 |
| eRST | LR | 0.687 | 0.688 | 0.746 | 0.785 | 0.741 |
| eRST | XGB | 0.710 | 0.788 | 0.762 | 0.801 | 0.774 |
| RST-DT | LR | 0.674 | 0.686 | 0.742 | 0.748 | 0.709 |
| RST-DT | XGB | 0.691 | 0.761 | 0.746 | 0.771 | 0.770 |
| NLDT | LR | 0.664 | 0.666 | 0.719 | 0.753 | 0.714 |
| NLDT | XGB | 0.696 | 0.754 | 0.755 | 0.789 | 0.761 |

M3-frequency XGB F1 ranges from .771 to .801. In this table eRST, not PCC or NLDT, has the highest M3-frequency F1. Older prose singling out PCC/NLDT should not be converted into a universal cross-inventory ranking.

The headline comparison with structure is supported by these reported paired effects:

| Inventory | Model | M3-frequency minus structural F1 | Nominal 95% CI | Reported Holm-adjusted value |
|---|---|---:|---|---:|
| PCC | LR | +0.090 | [+0.062, +0.118] | 0.0070 |
| PCC | XGB | +0.096 | [+0.072, +0.123] | 0.0070 |
| eRST | LR | +0.098 | [+0.070, +0.126] | 0.0070 |
| eRST | XGB | +0.091 | [+0.066, +0.118] | 0.0070 |
| RST-DT | LR | +0.074 | [+0.045, +0.103] | 0.0070 |
| RST-DT | XGB | +0.080 | [+0.053, +0.106] | 0.0070 |
| NLDT | LR | +0.088 | [+0.059, +0.118] | 0.0070 |
| NLDT | XGB | +0.092 | [+0.068, +0.117] | 0.0070 |

M3 contains useful information beyond the small document-scale structural baseline. A claim of incremental value beyond *every possible structural/syntactic control* would exceed this evidence. Likewise, a numerical advantage over marginal relation frequency is not itself a significance test; consult the corresponding contrast before making that stronger claim.

### Fixed-tree granularity: logistic regression

| Inventory | Features | Fine F1 | Coarse F1 | Fine minus coarse | Nominal 95% CI | Holm-adjusted p |
|---|---|---:|---:|---:|---|---:|
| PCC | MF | 0.768 | 0.687 | +0.081 | [+0.061, +0.102] | 0.0040 |
| PCC | MF+WAD | 0.774 | 0.716 | +0.057 | [+0.040, +0.076] | 0.0040 |
| eRST | MF | 0.785 | 0.729 | +0.056 | [+0.035, +0.078] | 0.0040 |
| eRST | MF+WAD | 0.778 | 0.697 | +0.081 | [+0.057, +0.104] | 0.0040 |
| RST-DT | MF | 0.748 | 0.746 | +0.002 | [-0.014, +0.019] | 0.8226 |
| RST-DT | MF+WAD | 0.770 | 0.739 | +0.032 | [+0.013, +0.050] | 0.0040 |
| NLDT | MF | 0.753 | 0.696 | +0.057 | [+0.037, +0.076] | 0.0040 |
| NLDT | MF+WAD | 0.766 | 0.702 | +0.064 | [+0.046, +0.084] | 0.0040 |

Interpretation:

- All four MF+WAD fine/coarse LR contrasts have positive supported gains under the reported correction.
- MF alone does not support a clear gain for RST-DT.
- “More theoretical labels implies a larger gain” is contradicted by the relative patterns.
- A difference of .057 F1 is 5.7 points on a 0–100 F1 scale, not .057 percent.

### Fixed-tree granularity: Kim-style random forests with MF+WAD

| Inventory | Fine F1 | Coarse F1 | Fine minus coarse | Nominal 95% CI | Holm-adjusted p |
|---|---:|---:|---:|---|---:|
| PCC | 0.778 | 0.751 | +0.028 | [+0.011, +0.047] | 0.0160 |
| eRST | 0.795 | 0.775 | +0.021 | [+0.003, +0.039] | 0.0840 |
| RST-DT | 0.780 | 0.771 | +0.009 | [-0.008, +0.026] | 0.6457 |
| NLDT | 0.763 | 0.757 | +0.006 | [-0.011, +0.024] | 0.6457 |

Only PCC passes the reported Holm adjustment at .05. PCC and eRST have pointwise intervals excluding zero, but that does not make both adjusted comparisons significant.

The LR and RF columns use different model/threshold protocols. Their absolute F1 values do not isolate the effect of changing only the classifier family.

### Does depth change the fine/coarse LR gain?

These are paired differences between gains, not simply the effect of adding depth to a fine classifier:

| Inventory | (MF+WAD fine/coarse gain) minus (MF fine/coarse gain) | Nominal 95% CI | Holm-adjusted p |
|---|---:|---|---:|
| PCC | -0.024 | [-0.044, -0.003] | 0.0660 |
| eRST | +0.025 | [+0.004, +0.044] | 0.0660 |
| RST-DT | +0.029 | [+0.012, +0.049] | 0.0080 |
| NLDT | +0.007 | [-0.011, +0.027] | 0.4498 |

Only the RST-DT interaction survives the reported adjustment. Adding depth does not uniformly enlarge the granularity benefit.

### Stable motif associations

| Inventory | Stable presence associations | Stable frequency associations | Stable conditional-depth associations |
|---|---:|---:|---:|
| PCC | 3 | 1 | 0 |
| eRST | 4 | 3 | 0 |
| RST-DT | 2 | 1 | 0 |
| NLDT | 3 | 2 | 0 |
| Total | 12 | 7 | 0 |

These are **19 candidate–representation associations**, not necessarily nineteen distinct linguistic relations or independent discoveries. No stable conditional-depth effects under this screen does not prove that depth is universally useless; depth participates in other predictive comparisons.

The report's primary association table and identity-clean sensitivity figures are not identical cohorts. Its relaxed 3-of-5-domain visualization is a sensitivity view requiring robust same-direction effects in at least three of five domains with none robustly reversed. This “relaxed” stability screen is unrelated to the email's relaxed *parser evaluation*.

### Cross-inventory and domain findings

Reported summary:

- 395 inventory-specific M3 motifs.
- 287 pipeline-equivalent families.
- 14 relation-agnostic structural families.
- Training selected 80 domain–motif hypotheses.
- 48 replicated on test; 47 also survived the identity-clean sensitivity.
- Four exact-label motif families showed recurring robust domain interactions in multiple inventories.

Interpretation: the inventory outputs share structural templates, while much of the discriminative detail is inventory-specific. An exact shared hash, an empirically corresponding label, and a shared linguistic meaning are different notions.

### Statistical qualifications

The implemented paired analyses use 2,000 label-stratified identity-cluster bootstrap resamples, percentile intervals, and bootstrap-sign quantities with Holm adjustments. The intervals shown above are nominal pointwise intervals; do not describe them as simultaneous corrected confidence intervals.

Model and threshold selection use validation data in the implemented protocol. However, historical planning explicitly recognizes prior exploration of HC3 test outcomes. Distinguish “no programmatic test-set threshold tuning in this final protocol” from “a wholly untouched, prospectively confirmatory test set.”

More bootstrap draws improve Monte Carlo resolution; they do not add independent documents or create a new statistical replication. There is no basis here for a broad “all dependence removed” or “fully powered” claim.

Scores and differences are copied at the source's printed precision. A printed difference can differ by .001 from subtraction of separately rounded F1 values. Do not silently change source results to make rounded arithmetic agree.

## C09 — How the granularity ablation works, and what it proves

### Procedure

1. Start from the selected fine M3 catalog and the existing per-document fine features.
2. Replace each fine relation label with its specified coarse label.
3. Retain structural `/` edges and graph topology.
4. Merge motifs whose projected directed relation-labeled graphs are isomorphic under the operative matching rule.
5. Sum the merged fine frequencies into each coarse feature.
6. Compute coarse WAD as the frequency-weighted average of its component fine WAD values.
7. Fit/evaluate the coarse-feature classifier under the declared protocol.
8. Pair fine and coarse predictions on identical document identities and bootstrap resamples.

For a coarse group `G_c` in document `d`:

```text
MF(c,d)  = sum over m in G_c of MF(m,d)

WAD(c,d) = sum over m in G_c of [MF(m,d) * WAD(m,d)] / MF(c,d)
```

Absent coarse motifs receive WAD zero in classifier input. A simple unweighted average of fine depths would not reproduce the implemented projection.

### Mapping definitions

- PCC/NLDT use the mapping in `scripts/canonical/ablate_m3_granularity.py`, broadly toward the original coarse DMRST categories.
- Full eRST uses parent-label groupings in `scripts/canonical/ablate_full_inventory_granularity.py`.
- Full RST-DT uses `docs/eng.rst.rstdt.new_to_coarse.json`.

Examples from the mappings include `list`/`conjunction`/`joint` → `Joint`, `elaboration`/`e-elaboration` → `Elaboration`, and eRST `joint-*` → `joint`.

These are declared operational mappings, not a demonstration that all inventories share an identical semantic ontology. Historical material requests independent RST-informed validation of consequential mapping choices. A validator/form-preparation script is not evidence that the review was completed.

### Defensible conclusion

Merging these specified distinctions changes predictive information available to the classifier on fixed representations. This supports an inventory- and model-conditional granularity finding.

### Conclusions it does not establish

- That the fine parser is more accurate than a separately trained coarse parser.
- That every fine label is correctly predicted.
- That one annotation theory is more truthful than the others.
- That the original paper's exact selected coarse motif catalog has been reproduced.
- That an independently selected coarse-tree catalog would yield the same features or score.
- That the full eRST graph, including secondary edges and signals, was evaluated.
- That a practical system will improve by the same amount on newer LLMs or other genres.

The coarse catalog is projected from the selected fine catalog rather than independently rediscovered from all coarsened trees. That controlled coverage is useful for isolation, but limits the scope of the comparison.

## C10 — The shift toward complementary interpretations and human review

The historical `UNIRST2_COHERENCE_COMPLEMENTARITY_PLAN.md`, revised 2026-09-10, is especially useful for understanding the intellectual shift.

It separates:

1. Whether motifs carry authorship-label information.
2. Whether fusing inventories improves a detector.
3. Whether inventory-conditioned analyses expose different defensible aspects of coherence.

The plan explicitly says success on question 2 is neither necessary nor sufficient for question 3. This is a central conceptual correction: classifier improvements can reflect errors or superficial cues, while two defensible analyses may convey different distinctions without changing a binary prediction.

### Archived fusion experiment

Reported in that historical plan:

- None of fourteen selected-reference fusion contrasts met a chosen practical criterion of at least +.01 ROC-AUC improvement.
- Pair gains were approximately +.0003 to +.0057.
- All-inventory fusion versus validation-best PCC had an identity-clean AUC gain of +.00415 for LR, nominal 95% interval [+.00056, +.00791].
- Corresponding XGBoost gain: +.00260, interval [-.00056, +.00572].

The +.01 threshold was an operational stopping criterion, not a universal scientific boundary. Failing it does not prove equivalence or absence of linguistic complementarity.

These are archived side-study outcomes, not additional results in the canonical report. The plan documents inferential qualifications: bootstrap-sign adjustment families, nominal rather than simultaneous intervals, first-seed inference, and exploratory rather than independent replication from repeated deterministic/grouped fits.

### Review preparation recorded in the plan and code

The workflow includes cohort audits, representation alignment, measurement audits, contrast extraction, partially blinded review preparation, rendering, and free-text note extraction.

The historical plan reports:

- A measurement audit completed on 2026-09-10.
- 1,428 inventory-pair/category contrast summaries.
- A candidate pool of 104 passages.
- A prepared package of 60 unique cases covering all six inventory pairs and all five HC3 domains.
- Exclusion of three candidate identities in near-duplicate clusters.
- Deterministic sampling/random ordering and a separate answer key.

These are historical reported preparation counts. They are not a count of completed annotations and may refer to pre-canonical cohorts or paths. The actual completed package and judgments are not tracked in the inspected clone.

Selection was designed to avoid authorship labels, model scores, prediction correctness, and motif coefficient signs as selection criteria. It balanced recurring differences, renaming controls, and agreement controls. The sample was purposive, so it cannot estimate the population prevalence of valid alternative interpretations without further sampling justification.

Inventory identifiers and authorship information are hidden in the reviewer view, but native relation names can reveal the inventory. The documentation therefore calls the blinding partial.

### Codebook questions

The implemented codebook asks reviewers to judge:

- Textual support for each analysis: A, B, both, neither, or uncertain.
- Well-formedness and EDU alignment.
- Whether both analyses are defensible.
- Difference types: renaming, granularity, attachment, nuclearity, alternative interpretation, parser/alignment error, no meaningful difference, or uncertain.
- What aspect of coherence each analysis highlights.
- Confidence and a rationale.

It permits uncertainty and disagreement rather than requiring consensus. Native relation names alone are not sufficient evidence about their theoretical meanings.

### Why relaxed evaluation was attractive

Strict matching can count a plausible alternative label as fully incorrect. For example, one analysis might highlight temporal precedence and another a causal connection; closely related causal/explanatory labels may also differ without making one analysis nonsensical.

The intended human evaluation appears aimed at separating real parser errors from such defensible alternatives. This is a plausible and source-supported interpretation of the mails, not proof that most disagreements are legitimate.

### Caution about earlier graph-overlap figures

The plan initially mentions exported-edge Jaccard .970–.987 and shared-edge label agreement .723–.930, but explicitly calls for a metric audit because structural scaffolding contributes. Do not recycle those figures as clean constituent agreement or final discourse-only agreement. The audit code distinguishes node-span, all-link endpoint, and discourse-link endpoint metrics.

## C11 — What the mails establish, and what they leave open

The supplied `mails.txt` has three substantive snippets separated by markers. It has no full headers, dates, or named sender fields. The first two read as the colleague's messages; the third is the collaborator's reply. That attribution follows their contents and the user's framing.

### Snippet 1: annotation processing and relaxed evaluation

The colleague says:

- Processing of their annotations is complete.
- Overall agreement on a shared subset is `0.42`.
- Much of that agreement comes from cases with matching predictions.
- The desired result is that UniRST looks better under relaxed evaluation.
- A workshop paper may be an appropriate outlet.

What is **not** specified:

- Whether .42 is Cohen's kappa, another chance-corrected statistic, raw agreement, or something else.
- The number of annotators, number of shared cases, label scheme, confidence interval, or adjudication procedure.
- Whether this is exactly the sixty-case package recorded in the earlier plan.
- The actual relaxed score or how much it exceeds strict matching.

Do not call this “42% agreement” or “kappa .42.” Do not equate it with the draft's GUM parser F1 of .422.

Interpretation: it supports that some annotation work happened outside the committed results, but shared agreement concentrated on matching cases weakens the basis for a strong claim about the difficult disagreement cases. A finished analysis, coding rubric, and per-case data are needed to evaluate that claim.

### Snippet 2: broaden the paper and try argumentation

The colleague clarifies that he and Massimo do not want an exclusive machine-text-detection focus, especially because HC3 is dated. He asks for another downstream task where RST features are useful.

He identifies UniRST's slow parsing, attributed in the email to a CRF component, as a practical limit on dataset size. This is his account of the bottleneck; the investigation did not independently benchmark or profile it.

He proposes `peldszus/arg-microtexts-multilayer`, says he is in the middle of trying it, invites suggestions or independent exploration of other small tasks, and links `dev` as his working code.

That link identifies a development starting point, not the location of all completed research. The canonical analysis is on the other branch.

### Snippet 3: collaborator's reply

The reply says the collaborator has used ArgMicrotexts before, sees it as a suitable small testbed, but is unsure how the argumentative structures would be derived from RST trees. The reply interprets the proposal as different frameworks producing different analyses that might affect argumentation prediction or interpretation.

This is the ambiguity the receiving agent should help resolve: RST representations can be **features for a separate prediction task**, without there being a deterministic relation-by-relation conversion into an argumentation tree.

## C12 — Argumentation: intended experiment, preliminary observations, and gaps

### Dataset facts checked against its public README

[ArgMicrotexts-Multilayer](https://github.com/peldszus/arg-microtexts-multilayer/blob/master/README.md) describes 112 short argumentative texts with parallel RST, SDRT, and argumentation-structure layers. Its ARG annotation is segmented more finely into EDUs than the original corpus's coarser argumentative discourse units.

This makes it attractive as a small dataset with an existing argumentative target and a discourse layer for comparison. The README alone does not establish the precise training/evaluation protocol your colleague used.

### Most plausible intended experimental chain — inference

```text
Same short text
  -> one of several RST representations
  -> selected relation/tree-derived features
  -> model predicting gold argumentative properties
  -> compare conditions using the same argumentative target
```

Possible conditions suggested by the notes/fragments include gold RST versus predicted RST, different predicted inventories, and fine versus coarse features. No complete implemented condition matrix was found here.

The following are different tasks and cannot be silently interchanged:

- Predicting an argumentative relation label for a pair already known to be linked.
- Predicting whether an argumentative link exists.
- Predicting attachment targets.
- Reconstructing an entire argument graph/tree.
- Predicting roles or other document-level argumentative properties.

The repository does not settle which task he implemented, what candidate pairs he used, or whether attachment structure was given.

### Preliminary finding in the planning notes

The latest planning note says coarse features are **more predictive** of argumentation structure because their mapping to argumentative labels is cleaner. This is a reported preliminary observation with no accompanying scores, code, or protocol in the inspected material.

It is not inherently inconsistent with fine features helping HC3. Different targets require different distinctions, and small datasets can make fine categories harder to estimate. The latter is a general possible explanation, not a demonstrated mechanism in this experiment.

### Separate draft fragment: `latex/argumentative.tex`

The fragment discusses an observed advantage of gold/canonical ArgMicrotexts RST markup over UniRST-predicted trees. It proposes two contributing factors:

1. **Annotation quality and evaluation:** predicted trees may contain errors, while strict label matching may also penalize related or parallel readings.
2. **Annotation provenance:** the gold RST annotator was reportedly involved in creating the underlying corpus and may have known the argument structure, which could encourage rhetorical labels that align neatly with argumentative categories.

The second point is a hypothesis in the draft, not an established bias finding. The fragment says a clean-slate reannotation would be needed to estimate that influence.

Counts stated in the fragment:

| Gold RST relation / prediction | Reported instances |
|---|---:|
| `reason` in gold | 173 |
| `evidence` in gold | 12 |
| `cause` in gold | 14 |
| `contrast` in gold | 3 |
| `antithesis` in gold | 26 |
| `concession` in gold | 65 |
| `contrast` predicted by UniRST | 13 |

These counts were not independently recalculated. The fragment argues that frequent `reason`–`support` and `concession`–`undercut` associations would be easy for a downstream learner to capture. They are described corpus associations, not universal semantic equivalences.

The fragment has an empty result reference, an incomplete citation, no scores, and no complete setup. The main LaTeX file contains no input of this fragment in the inspected snapshot.

### What was not found

- An ArgMicrotexts loader, feature-building pipeline, experiment configuration, or trained model.
- A train/test split or cross-validation definition.
- Treatment of EDU/ADU alignment.
- A statement of whether the texts, gold EDUs, gold RST topology, or candidate links were supplied.
- A metrics table, uncertainty estimates, or leakage controls.
- The actual predicted and gold structures used in the described observations.

The available evidence therefore supports “proposed and apparently explored outside the available committed experiment,” not “nothing happened” and not “a validated second task is complete.”

## C13 — Engineering on `dev`: UniRST, batching, and additional corpora

### UniRST interface

The preprocessing adapter persists a shared EDU sequence and reuses it for each inventory through `from_edus()`. This avoids modifying the upstream parser package in-place and keeps downstream document/scene/graph formats compatible with the inherited pipeline.

Default UniRST inventory keys on `dev`:

```text
eng.erst.gum
eng.rst.rstdt
deu.rst.pcc
nld.rst.nldt
```

These generic defaults are not the canonical research condition's two `.new` keys.

### Output layouts and resumption

Typical development outputs:

```text
data/unirst/
  segments-deu.rst.pcc/
  rel-eng.erst.gum/
  rel-eng.rst.rstdt/
  rel-deu.rst.pcc/
  rel-nld.rst.nldt/
  paired/
```

The script supports separate per-inventory outputs, paired outputs holding all inventory predictions for a scene, or both; both is the default. Requested existing outputs can be skipped. Segmentation caches and final outputs have explicit force/rebuild controls.

The README describes validation, atomic cache writes, resume handling, per-scene statuses, and recovery from failed batches. A missing paired artifact can require reparsing all inventories so the paired record is complete.

### Batching

`dev` uses the upstream batched UniRST branch pinned in its requirements. The README states Python 3.10 or newer is required for that upstream version.

- Default `--batch-size 1` uses individual parsing.
- Larger sizes use `from_edus_batch` where supported.
- Learned segmentation encoders can also operate in batches without decoding full discourse trees.
- Failed batches are reported and retried individually; unavailable batch APIs or wrong result counts are treated as errors.
- Existing segmentation caches are reused across batch sizes.
- Larger batches change memory requirements; rounding can change borderline predictions.
- `--force-segmentation` and `--force-parsing` have distinct purposes.

This addresses throughput, but the repository commits alone do not provide a quantitative speedup or accuracy equivalence benchmark.

### Development motif workflow versus canonical scope

`dev` defaults to M3 and M6 distributions. M9 remains optional. It has support-based M6 pruning, selection manifests, incremental/atomic output handling, and CPU-worker controls.

M6 pruning can require minimum scene support, minimum shard support, and a cumulative coverage fraction. Defaults preserve unpruned behavior; the README example uses scene support 10, shard support 2, and coverage .95.

This is different from the research branch's training-only **M3-only** canonical experiment. Do not cite M6 or M9 development capability as evidence it was included in the paper's reported tables.

### Additional corpora

| Corpus | Role in available material |
|---|---|
| HC3 | Canonical quantitative testbed and original workflow input. |
| MAGE | Original/development parsing and detector workflow; omitted from canonical reported analysis. |
| Human Detectors | Added preparation and archived holdout/motif-distribution studies; excluded from canonical paper-facing outcomes. |
| StoryScope | Optional ingestion added on `dev`; no committed canonical evaluation on it. |

StoryScope details from the implementation/README:

- Source: `jjrussell10/storyscope`.
- Released train, validation, test, and dev splits are supported.
- Columns `story_gpt`, `story_deepseek`, `story_kimi`, `story_gemini`, and `story_claude` become separate machine-labeled documents when they meet the minimum length.
- Sources retain model-column and prompt ID information.
- Prompt sharding and output batches are supported; prompt-batch size is distinct from inference-batch size.
- The release used here does **not** include human story text, so it cannot by itself provide both classes for a human/generated classifier.

It is reasonable to infer interest in newer generated writing, but the presence of an ingestion path does not establish a new completed downstream study.

## C14 — The Dipo/Ettin feature branch

### Purpose and lineage

`feature/ettin-dmrst` starts from `dev` at `e658d85`. The supplied notes mention finding Dipo and expecting faster parsing; the feature branch is a concrete implementation of that direction.

The branch preserves the inherited dataset ingestion and downstream document, graph, motif, distribution, and split interfaces. It introduces `tos/ettin.py`, `requirements-ettin.txt`, `README_ETTIN.md`, and adapter tests, and changes preprocessing entry points to use the new backend.

### Models and identifiers

| Output inventory key | Checkpoint | Pinned model revision |
|---|---|---|
| `eng.erst.gum-12.1.0.ettin` | `larc-iu/dmrst-ettin-400m-gum-12.1.0` | `e33906a27f89b2703df8ab19e471aa82141d62cb` |
| `eng.rst.rstdt-coarse.ettin` | `larc-iu/dmrst-ettin-400m-rstdt-coarse` | `26594833545ef713d14e2ea6e9a7670c881c143a` |

Dipo revision: `4debc92e41140e5bcd9aadb987a297244c0fb14a`.

The branch documentation states GUM has 32 native relation labels and the RST-DT checkpoint has 18 coarse labels. Predicted NS/SN/NN nuclearity and native relation labels are retained. The integration adds no universal label projection, inventory matching, or new nuclearity pooling.

### Segmentation and scene boundaries

- Shared segmentation comes from the GUM Ettin checkpoint, replacing the PCC-based target used in the canonical UniRST study.
- Dipo's default paragraph/heading boundaries remain enforced.
- Both parsers receive the same cached EDU sequence.
- Documents fitting one GUM encoder window are kept intact.
- Longer documents fall back to the inherited XLM-R token-budget scene splitter.
- This changes tree/scene boundaries relative to the earlier pipeline, so outputs require a separate root.

### Runtime and provenance

- Separate Python 3.10+ environment; branch documentation recommends 3.12.
- `requirements-ettin.txt` uses Transformers 4.49.0 and PEFT 0.14.0, distinct from the legacy environment.
- No UniRST checkpoint weights are loaded by this backend, although the adapter reuses parts of the shared interface.
- The pinned Dipo implementation runs individual documents. Accepting `--batch-size` does not mean native multi-document GPU batching is implemented.
- The adapter validates that returned leaves preserve the supplied EDU text and order.
- `ettin_provenance.json` records models, revisions, segmentation, and scene policy and rejects incompatible existing output roots.
- New inventory keys prevent silent mixing with the original UniRST conditions.

The README's offline tests include a tiny randomly initialized ModernBERT exercising Dipo code. These are integration checks, not evaluation of released-checkpoint accuracy, throughput, or memory requirements. This handover investigation did not independently rerun those tests or download the models.

### Scientific interpretation

The branch README explicitly says the two parsers differ in training corpora and relation spaces. A difference in their downstream scores would not isolate inventory granularity.

The branch also changes the shared segmentation and scene policy relative to the canonical study. Therefore a new Ettin run is a different experimental condition. It cannot be spliced into the existing fine/coarse results as if only speed had changed.

No downstream results or measured speedup from this branch were found in the inspected commits. No PCC Ettin integration appears in its actual two-model configuration, even though the notes mention PCC as a possible additional option.

## C15 — Contradictions, stale prose, and how to reconcile them

| Apparent contradiction / confusion | Resolution supported by available evidence |
|---|---|
| The mail links `dev`, but the useful research reports are absent there. | The work diverged. `dev` is the development pipeline; `research/unirst2-canonical` holds the consolidated analysis. |
| `main` looks almost untouched by the new research. | It is the inherited baseline/default branch, not a summary of all colleague work. |
| The notes say eRST/RST-DT are only coarse in UniRST. | That describes the earlier stock-parser limitation. The draft later reports fine-tuning, and canonical inputs explicitly include full `.new` conditions. |
| The introduction promises corpus conversion and new-data parser training, but the body focuses on HC3 ablations. | The introduction retains the larger earlier proposal. No completed LLM conversion/enriched-corpus workflow was found. |
| `old.tex` claims successful LLM few-shot experiments. | It is an old prose fragment with no located supporting implementation/results. Treat its empirical-sounding claims as unverified draft statements. |
| Notes call the work a reproduction of *Threads of Subtlety*. | It reuses/extends the method. The final research question, parser, features, models, and scope differ from exact replication. |
| Original detector scores exceed the feature-only classifiers. | The original Longformer also uses text; task conditions and possibly metrics differ. The canonical goal is representation analysis, not directly improving those headline scores. |
| Notes say features were “prepended” to Longformer. | The inspected classification head concatenates motif vectors with the first-token text representation. |
| `dev` uses M3+M6 and has M9, but the draft says M3 only. | Development capabilities exceed the canonical experiment's deliberately narrower scope. |
| PCC/NLDT are described as especially good, but eRST has the highest M3-frequency F1. | Earlier observations and within-inventory fine/coarse gains are not universal cross-inventory rankings. Use the specific final table and question. |
| More granular RST-DT has a small gain. | Label count does not determine task usefulness. Rare, poorly learned, or irrelevant distinctions may add little. These are possible explanations, not a completed causal diagnosis. |
| Fine features help HC3 while coarse features reportedly help argumentation. | The targets and learning conditions differ. Granularity is task-dependent; the argumentation result also lacks a reviewable protocol here. |
| Gold RST predicts argumentation better, so predicted analyses must be linguistically wrong. | Gold quality can matter, but annotation provenance, label distribution, alternative readings, and modeling protocol are competing explanations. |
| “Shared segmentation” is taken to mean shared trees. | Only EDU boundaries are held shared across inventory parses. Trees are fixed in each within-inventory coarsening ablation. |
| All coarse conditions are assumed to be the same universal 18-label ontology. | The operational mappings differ: coarse DMRST-like groups, eRST parents, and the RST-DT mapping. |
| Relation dimensions are mistaken for complete inventory sizes. | Canonical relation features are training-derived and include an `__OTHER__` column. |
| Human agreement `.42` resembles parser F1 `.422`. | Different quantities from different sources; the agreement statistic is not specified. |
| A “relaxed” figure is assumed to report relaxed parser evaluation. | The 3-of-5-domain motif-stability figure and the mail's relaxed human evaluation are separate procedures. |
| The plan says human judgments were not collected, but a mail says annotation processing is complete. | They are different snapshots, and the mail is undated. The completed data/analysis is not present to connect them precisely. |
| Archived plans use 43,319/968/1,931 or larger per-inventory cohorts. | Those are historical/intermediate counts. Final reported matched cohort is 43,322/968/1,932. |
| Large graph overlap is treated as linguistic agreement. | Shared scaffolding and metric definitions matter; the project itself audited this issue. |
| A new Ettin backend is treated as a clean fine/coarse comparison. | It changes models, training provenance, segmentation, and scene policy. Its own README cautions against that interpretation. |
| A working inference pipeline implies the new study is complete. | Ingestion/integration is not a fitted model, score table, human validation, or completed paper. |

### State of the main manuscript

Title: **“More subtleties: making sense of RST granularity.”**

The inspected main LaTeX file:

- Has an extremely short abstract and placeholder author details.
- Contains substantive background, experimental setup, parser adaptation, cohort construction, projection equations, and an LR/RF result table.
- Repeats the final canonical MF+WAD LR/RF scores.
- Discusses rare RST-DT distinctions and enumeration/list cues.
- Still promises the older corpus-conversion programme in its introduction.
- Has TODO appendix/exclusion references and incomplete discussion/conclusion passages.
- Does not contain a complete annotation-analysis section or completed argumentation experiment.
- Does not input the separate `argumentative.tex` fragment.

It is therefore a useful but mixed-stage working draft. It should not be read as a complete, internally synchronized record of everything implemented.

## C16 — What is completed, reported, proposed, or missing

| Component | Best-supported status |
|---|---|
| Original DMRST/Longformer/motif workflow | Versioned inherited implementation. |
| UniRST replacement and shared EDU cache | Implemented, with tests and documentation. |
| M3/M6/M9 engineering and pruning | Implemented capabilities; larger motifs excluded from canonical scope. |
| Matched four-inventory HC3 analysis | Versioned analysis and reported result snapshots; full raw run unavailable here. |
| Fixed-tree fine/coarse LR/RF comparisons | Implemented with reported tables matching the draft. |
| Full eRST/RST-DT checkpoint adaptation | Reported in draft and expected by canonical inputs; complete training/release workflow unavailable in this repo. |
| Stable-association/domain/family analyses | Versioned analysis and reported outcomes. |
| Inventory fusion | Archived side study with a qualified small/negative practical result. |
| Human-review preparation | Implemented; historical package construction reported. |
| Completed human annotations | Processing said to be complete in email; raw judgments and final analysis not found. |
| Relaxed parser-evaluation improvement | Intended/claimed direction in email; no located score/protocol sufficient to verify it. |
| ArgMicrotexts task | Proposed in mail; preliminary observations in notes/fragment; no committed experiment found. |
| LLM corpus conversion | Earlier proposal; no completed implementation or enriched corpus found. |
| StoryScope | Implemented input preparation on `dev`; no canonical result and no human stories in the used release. |
| Ettin/Dipo | Implemented backend integration on feature branch; no downstream evaluation or benchmark found. |
| Submission-ready paper | Not supported by the supplied incomplete manuscript. |

### Most valuable missing handover material

To settle the largest open scientific questions, obtain these only if Max later asks to continue the research:

1. **Completed human-review analysis:** per-case judgments, annotators/shared cases, coding rubric, agreement statistic, handling of matching/disagreement cases, strict versus relaxed evaluation definition and scores, uncertainty, and adjudication.
2. **Argumentation experiment:** actual code location, exact target, feature conditions, corpus version, EDU/ADU alignment, given structures/candidate links, splits or document-level cross-validation, metrics, scores, and predictions.
3. **Canonical restoration bundle:** aligned shared-segmentation records, inventory-specific training catalogs and manifests, feature/model artifacts, provenance, and a storage/download/restoration record.
4. **Fine-tuned parser provenance:** checkpoint locations/revisions, full training commands/configurations, source annotation splits, head initialization, metrics, and evaluation protocol.
5. **Mapping review:** any completed independent assessment of the fine-to-coarse relation mappings and consequential merged groups.
6. **Newest manuscript:** whether the supplied Overleaf snapshot has since been revised to reconcile the contribution narrative and incorporate review/argumentation findings.
7. **New-backend evaluation:** any actual Ettin speed/memory measurements, parse-quality checks, or downstream results.

Do not stop a basic explanation to ask all these questions. This file already supports an informative account. These become necessary when evaluating claims, reproducing results, or implementing further work.

## C17 — Repository map for a fresh clone

### On `dev` in the inspected snapshot

| Paths | Purpose |
|---|---|
| `0_prepare_and_parse_datasets.py` | Original parsing entry point. |
| `0_prepare_and_parse_datasets_unirst.py` | UniRST input preparation and parsing. |
| `1_add_graphs_to_datasets.py`, `1_add_graphs_to_unirst_datasets.py` | Graph conversion. |
| `2_extract_single_triads.py` | M3 extraction. |
| `3_extract_double_triads.py` | M6 extraction and selection. |
| `4_extract_triple_triads.py` | Optional M9 extraction. |
| `5_add_motif_dists_to_datasets.py`, `5_add_motif_dists_to_unirst_datasets.py` | Motif distributions. |
| `6_split_hc3_dataset.py` | Split preparation. |
| `7_train_longformer.py`, `8_evaluate_longformer.py` | Inherited/adapted Longformer workflow. |
| `9_prepare_and_parse_ood_sets.py` | Additional evaluation preparation. |
| `tos/unirst.py` | Shared EDU adapter, scene handling, parsing/caching helpers. |
| `tos/tos_dataset.py` | Document/scene models, graphs, motifs, distributions. |
| `tos/tos_utils.py` | Shared motif/isomorphism/selection utilities. |
| `tos/tos_models.py` | Longformer text/motif model. |
| `tests/` | Adapter, batching, corpus-loader, and motif tests. |

The root's legacy `data/motifs/hc3-mage_*` files are not the missing four-inventory canonical catalogs.

### On `research/unirst2-canonical`

```text
README.md
UNIRST2_RESEARCH_REPORT.md
PAPER_RELEVANT_RESULTS_TABLES.md
REPOSITORY_REFACTOR_PLAN.md
configs/canonical.json
run_canonical.py
docs/
  REPOSITORY_GUIDE.md
  UNIRST2_PIPELINE.md
  eng.rst.rstdt.new_to_coarse.json
  figures/canonical/
  figures/relation_frequencies/
  tables/overall_relation_frequencies.csv
scripts/
  prepare/
  canonical/
  review/
tos/
  analysis/granularity.py
  ...
archive/
  README.md
  code/original_acl2024/
  code/superseded/
  code/side_studies/
  code/one_off_utilities/
  docs/
```

Important canonical modules:

| Module under `scripts/canonical/` | Role |
|---|---|
| `build_feature_matrices.py` | Build structural, relation, presence, frequency, and depth matrices. |
| `prepare_canonical_cohort.py` | Match inventory identities, labels, ordering, and input provenance. |
| `aggregate_relation_frequencies.py` | Pooled relation counts and figures. |
| `evaluate_feature_models.py` | LR/XGB representation models and predictions. |
| `compare_paired_performance.py` | Paired bootstrap contrasts. |
| `analyze_motif_associations.py` | Presence/frequency/conditional-depth associations. |
| `run_robustness_analysis.py` | Identity-clean sensitivities and stability checks. |
| `map_cross_standard_motifs.py` | Cross-inventory families and overlap. |
| `ablate_m3_granularity.py` | PCC/NLDT fine-to-coarse projection. |
| `ablate_full_inventory_granularity.py` | Full eRST/RST-DT projection. |
| `compare_rf_granularity.py` | Dedicated Kim-style RF MF+WAD comparison. |
| `compare_lr_wad_granularity.py` | LR MF/MF+WAD and paired gain interaction. |
| `compare_motif_distributions.py` | Distribution analysis. |
| `analyze_domain_motif_distinctions.py` | Domain-conditioned effects. |
| `visualize_results.py` | Research figures. |
| `synthesize_results.py` | Report/table generation. |

Preparation modules include parsing, alignment to paper splits, graph addition, M3 extraction, motif distributions, and legacy cache restoration. Review modules include `audit_cohort`, `align_representations`, `audit_measurements`, `extract_representation_contrasts`, `prepare_blinded_review`, `render_review_graphs`, `extract_filled_case_notes`, and `format_filled_case_notes`.

### Archive boundary

The canonical workflow does not read from `archive/`. The archive preserves:

- Original ACL 2024 preparation scripts.
- Superseded coarse/full comparisons and one-off alignment/plotting code.
- M6/M9, Longformer, fusion, OOD, and Human Detectors side studies.
- The old project-flow audit and coherence-complementarity plan.

Generated exploratory results are not necessarily versioned with those scripts. `archive/manifest.json` is an optional ignored local inventory, not a checked-in data-restoration manifest.

### On `feature/ettin-dmrst`

Look first at `README_ETTIN.md`, `tos/ettin.py`, `requirements-ettin.txt`, and `tests/ettin_adapter_test.py`. It retains the older numbered pipeline from `dev`; it does not automatically inherit `scripts/canonical/` from the research branch.

## C18 — Reproduction boundary and commands, if later requested

Explanation needs no new GPU run. The commands here describe the existing workflow; they were **not** all executed during this investigation.

### Canonical inputs expected but not supplied in Git

```text
data/unirst2/shared_segmentation/rel-<inventory>/*.jsonl
data/motifs2/shared_segmentation/<inventory>/*
```

Here `<inventory>` is one of:

```text
deu.rst.pcc
eng.erst.gum.new
eng.rst.rstdt.new
nld.rst.nldt
```

Final records use `.graph_added.m3_motif_dists.jsonl`, retaining text, scenes, graph data, and M3 distributions. The exact full filename pattern used by matrix construction is of the form:

`hc3_train.discourse_parsed.graph_added.m3_motif_dists.jsonl`.

The canonical guide says there is no in-repository data-download command. Inputs must be restored from the maintainer/team's research storage with the correct layout. Stock original catalogs are not a substitute.

### Canonical runner

After choosing the research branch/worktree, restoring inputs, and installing the branch-appropriate dependencies:

```bash
python run_canonical.py resume --config configs/canonical.json
python run_canonical.py verify --config configs/canonical.json
```

`build` forces a full rebuild/refit; `report` regenerates the Markdown reports from existing results. These are purposeful actions with compute/output consequences, not prerequisites for discussing this handover.

The runner starts from prepared M3 records. It does not recreate missing parser checkpoints or replace the entire raw-corpus preparation process.

Canonical configuration snapshot:

| Setting | Value |
|---|---|
| Data root | `data/unirst2/shared_segmentation` |
| Motif root | `data/motifs2/shared_segmentation` |
| Raw feature root | `artifacts/work/shared_segmentation_raw_feature_matrices` |
| Artifact root | `artifacts/canonical` |
| RST-DT mapping | `docs/eng.rst.rstdt.new_to_coarse.json` |
| Model seed | 42 |
| Bootstrap seed | 20260909 |
| Domain seed | 20260912 |
| Distribution seed | 20240910 |
| Bootstrap iterations | 2,000 |
| Bootstrap batch size | 250 |
| Jobs | 4 |
| Figure DPI | 180 |

Expected verification dimensions include 40 model-summary rows, 280 paired-summary rows, 144 stability rows, 200 robustness-model rows, 40 RF-granularity rows, and 80 LR-WAD-granularity rows. These are artifact-contract dimensions, not independent experiments or independent replications.

The verifier checks expected identities/counts, inventory membership, artifact sizes/hashes, report scope, and run provenance. It cannot establish linguistic validity or recreate missing raw results. Passing unit tests is not a substitute for verifying the restored canonical run.

### Expected generated artifact layout

```text
artifacts/canonical/
  feature_matrices/
  relation_frequencies/
  model_comparison/
  paired_comparison/
  motif_associations/
  robustness/
  cross_standard_mapping/
  granularity/
    pcc_nldt/
    erst/
    rstdt/
    rf_mf_wad/
    lr_mf_wad/
  motif_distributions/
  domain_distinctions/
  figures/
  report.md
  manifest.json
```

Optional review outputs live under `artifacts/review/`; historical generated packages may instead be archived under earlier names.

### Unit tests and dependency limits

The research branch recommends `python -m pytest -q` plus the canonical verifier. No model tests or full runs were executed as part of this explanatory investigation; its work was inspection and documentation.

The research branch's `requirements.txt` includes Transformers 4.46.3 and XGBoost 2.1.1, but also Git dependencies pointing at moving branches such as `master`. The environment is not a complete immutable lockfile. Provenance and restored artifacts matter for exact reproduction.

The Ettin environment is separate and uses its own pinned Dipo revision and model revisions. Do not overwrite a functioning legacy environment just to inspect the feature branch.

## C19 — Evidence ledger and unresolved claim assessment

| Claim | Assessment | Evidence / limitation |
|---|---|---|
| The project is coherent despite inconsistent documents. | Strong reconstruction. | Shared granularity question, canonical implementation, draft table match, explicit scope-change notes. |
| Fine labels can retain HC3 predictive information. | Implemented and reported, conditional. | Fixed-tree ablations; model/inventory-dependent support. |
| PCC is the best standard overall. | Not established. | Different feature/model tables rank conditions differently; linguistic correctness not tested by downstream F1. |
| All four fine conditions significantly beat coarse RF. | False for reported adjusted comparisons. | Only PCC RF MF+WAD survives Holm at .05. |
| All four fine MF+WAD LR conditions have supported gains. | Reported supported result. | Four adjusted p-values .0040. |
| RST-DT fine MF alone has a clear gain. | Not supported in final table. | Difference .002, CI crosses zero, adjusted p .8226. |
| Fine labels are correctly predicted because most appear in a histogram. | Not established. | Nonzero frequency shows label use, not correctness. The draft itself makes this distinction. |
| Enumeration/list distinctions help explain early generated style. | Plausible interpretation present in notes/draft. | No independently inspected localized causal attribution establishes it as the sole gain source. |
| Combining inventories offers major practical gains. | Not supported under archived fusion gate. | 0/14 selected-reference contrasts meet chosen +.01 AUC criterion. |
| Failed fusion proves no linguistic complementarity. | Invalid inference. | Historical plan explicitly rejects that stopping rule. |
| Human review preparation was implemented. | Observed. | Review scripts and codebook; historical reported package counts. |
| Completed annotation processing happened. | Reported in email. | Final bundle unavailable. |
| Agreement is Cohen's kappa .42. | Unknown. | Mail does not name the statistic. |
| UniRST is substantially better under relaxed evaluation. | Not verifiable here. | Desired conclusion in mail; score/protocol/data absent. |
| ArgMicrotexts is a small parallel-annotation testbed. | Checked dataset fact. | Public README describes 112 texts and RST/SDRT/ARG layers. |
| ArgMicrotexts coarse features outperform fine features under a known protocol. | Preliminary report only. | Note lacks code, scores, splits, target definition. |
| Gold RST's argumentation advantage proves annotation contamination. | Not established. | Draft proposes provenance as one hypothesis and requests clean-slate evidence. |
| LLM fine-grained reannotation was completed. | No supporting implementation/results located. | Proposal and old empirical-sounding prose are insufficient. |
| Full eRST/RST-DT models were fine-tuned. | Reported, with expected outputs in analysis. | Draft training recipe and `.new` conditions; training/release pipeline unavailable here. |
| StoryScope alone supports an authorship classifier. | Not with used release/loader. | All included stories are machine-labeled; human text absent. |
| Ettin yields a measured speedup or preserves old results. | Unknown. | Integration exists; no located benchmark/downstream evidence. |
| Everything is reproducible after a simple clone. | False at inspected snapshot. | Missing raw inputs, generated artifacts, parser-training provenance, and local paper/email files. |

## C20 — Source registry, stable links, and local-source provenance

Use this registry for targeted verification. The point of this document is that the receiving agent need not open every item again.

### Versioned research sources at the inspected commit

All paths in this list refer to `f06309b62cbc53dce0c98668cac225666dfb64d7` on the research branch unless otherwise noted.

| ID | Source | What it establishes |
|---|---|---|
| S01 | [Research README](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/README.md) | Scope, layout, common cohort, restoration boundary, workflow. |
| S02 | [Research report](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/UNIRST2_RESEARCH_REPORT.md) | Reported prediction, projection, association, and domain results. |
| S03 | [Paper-relevant tables](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/PAPER_RELEVANT_RESULTS_TABLES.md) | Final cohort/dimensions, numerical tables, scope and metric statements. |
| S04 | [Repository guide](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/docs/REPOSITORY_GUIDE.md) | Practical paths, missing input bundle, provenance, no data downloader. |
| S05 | [Pipeline guide](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/docs/UNIRST2_PIPELINE.md) | Stage dependencies and verification contract. |
| S06 | [Canonical config](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/configs/canonical.json) and [runner](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/run_canonical.py) | Exact configured paths, seeds, dimensions, orchestration. |
| S07 | [Granularity helpers](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/tos/analysis/granularity.py) | Mass/identity checks, weighted WAD, edge-labeled isomorphism. |
| S08 | [Feature matrices](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/canonical/build_feature_matrices.py) | Actual feature definitions, identity hashing, `__OTHER__` column. |
| S09 | [Model fitting](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/canonical/evaluate_feature_models.py) | LR/XGB grids, validation AUC selection, balanced-accuracy threshold. |
| S10 | [PCC/NLDT ablation](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/canonical/ablate_m3_granularity.py), [full-inventory ablation](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/canonical/ablate_full_inventory_granularity.py), [RST-DT map](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/docs/eng.rst.rstdt.new_to_coarse.json) | Projection definitions, fixed-tree interpretation. |
| S11 | [RF comparison](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/canonical/compare_rf_granularity.py), [LR comparison](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/canonical/compare_lr_wad_granularity.py) | MF+WAD construction, protocols, gain interaction. |
| S12 | [Complementarity plan](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/archive/docs/UNIRST2_COHERENCE_COMPLEMENTARITY_PLAN.md) | Scientific reframing, archived fusion result, sampling/review history, qualifications. |
| S13 | [Pre-refactor project flow](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/archive/docs/PROJECT_FLOW_PRE_REFACTOR.md) | Historical counts, confounded comparisons replaced by fixed-tree analysis, work lineage. |
| S14 | [Blinded review preparation](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/review/prepare_blinded_review.py) | Actual review questions, partial blinding, case presentation. |
| S15 | [Measurement audit](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/review/audit_measurements.py) | Structural versus discourse-only overlap metrics and provenance checks. |
| S16 | [Archive README](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/archive/README.md) and [refactor record](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/REPOSITORY_REFACTOR_PLAN.md) | Active/archive boundary and local-only generated artifacts. |

### Development, feature, and external sources

| ID | Source | What it establishes |
|---|---|---|
| S17 | [Development README at e658d85](https://github.com/ruthenian8/threads-of-subtlety/blob/e658d857b1b3a803f91873c3a18258850a9c4e65/README.md) | UniRST segmentation/batching, M3/M6 default, StoryScope, original headline table. |
| S18 | [Longformer model at e658d85](https://github.com/ruthenian8/threads-of-subtlety/blob/e658d857b1b3a803f91873c3a18258850a9c4e65/tos/tos_models.py) | Motif-vector concatenation with text representation. |
| S19 | [Ettin README at 696aa60](https://github.com/ruthenian8/threads-of-subtlety/blob/696aa60e70dc194303010844ab4e542b10729ffa/README_ETTIN.md) and [adapter](https://github.com/ruthenian8/threads-of-subtlety/blob/696aa60e70dc194303010844ab4e542b10729ffa/tos/ettin.py) | Backend design, model pins, changed segmentation/scene policy, limitations. |
| S20 | [ArgMicrotexts README](https://github.com/peldszus/arg-microtexts-multilayer/blob/master/README.md) | Public dataset size/layers and EDU/ADU distinction; checked 2026-11-11. This is a branch URL, not an immutable revision. |
| S21 | [Original paper](https://aclanthology.org/2024.acl-long.298/) | Original project citation. The investigation used the inherited code/README and supplied draft's account; it did not independently audit every experiment in this paper. |
| S22 | [Dipo repository](https://github.com/larc-iu/dipo) | Library identified in the notes and integrated by the feature branch. The inspected integration pins its own revision. |

### Supplied local sources not included in the inspected Git branches

Relative paths below refer to the supplied directory at the investigation machine. They may be absent after cloning; the substantive findings are preserved above.

| ID | Local source | Relevant content |
|---|---|---|
| L01 | `RST_relation_granularity/Towards_granular_corpora.md` | Original three-part proposal, claimed preliminary observations, scope revision, argumentation and Dipo additions. |
| L02 | `RST_relation_granularity/mails.txt` | Annotation processing and .42 agreement; relaxed evaluation; task expansion; collaborator uncertainty. |
| L03 | `RST_relation_granularity/latex/acl_latex.tex` | Main draft, title, parser adaptation, canonical cohort/projection/table, unfinished contribution narrative. |
| L04 | `RST_relation_granularity/latex/argumentative.tex` | Unintegrated argumentation fragment and reported relation counts/provenance hypothesis. |
| L05 | `RST_relation_granularity/latex/old.tex` | Older LLM few-shot prose; unsupported by located experimental artifacts. |

Approximate useful line locations in the inspected snapshots:

- L01: 31–56 original downstream proposal/observations; 58–95 corpus conversion and optional parser training; 97–111 later scope/argumentation/Dipo notes.
- L02: line 1 annotation result; lines 3–5 task expansion; line 9 collaborator reply.
- L03: title line 44; introduction's older promises around 163–168; setup 182–206; parser training 211–221; segmentation/cohort 223–232; projection 235–259; results table 267–287; discussion 307 onward.
- L04: essentially the entire short fragment.

SHA-256 values of those local files as inspected:

```text
09197b344883c6d0cb52117accd8f65e29c388e3ef8b07dad00e189fd99901b9  Towards_granular_corpora.md
e2a057c6b3a9f5932f0d0f111e64fa5d4b497691902105b191814d9fd7f89218  mails.txt
8d8c15ebf166432b6a4751491a1f9e9773e83fcd037258a331b69abe7d7b6059  latex/acl_latex.tex
b7fb0287bcd268b37c6cfac7e2ef01f0015f6992d03539e754d705729c70335b  latex/argumentative.tex
69cfa2b065a053b4a716dcde90eb4240578541ed0e7a7943c59aaddbd6405df5  latex/old.tex
```

L01 contains an Overleaf read link: [supplied project link](https://www.overleaf.com/read/fphsrcgdhchm#d3c78d). The investigation read the local snapshot, not a freshly retrieved live Overleaf version. No source commit/date was available for that snapshot. Template README/formatting files in the paper directory mostly describe ACL submission formatting and should not be mistaken for research notes.

## C21 — Ready-to-use short explanations for Max

These are suggested formulations, not additional evidence.

### “What was he trying to find out?”

He wanted to know whether the detailed labels in RST annotation schemes capture useful differences that ordinary coarse parsers throw away. Human-versus-generated writing was the first testbed. Later he also wanted to understand whether different parser analyses can be linguistically defensible, and whether the representation helps predict argumentation.

### “Why are there several branches?”

They are different lines of work. `main` is the original project, `dev` is the changing parsing pipeline, `research/unirst2-canonical` contains the controlled experiment and its results, and `feature/ettin-dmrst` introduces a newer backend. The latest development branch does not contain all the research code.

### “What is the cleanest experiment?”

Take an already predicted fine-grained tree and replace several detailed labels with one coarse label. Keep the text, discourse units, attachments, and occurrence counts unchanged. If the classifier loses performance, the merged distinctions contained useful information under that model and mapping. That is cleaner than comparing two parsers that may also build different trees.

### “Did fine labels help?”

On the reported HC3 experiment, yes in several conditions. All four inventories show supported gains for logistic regression with motif frequency and depth. Random forests show smaller gains, with only PCC passing the reported correction. The result is task- and model-dependent; it is not a universal ranking of RST schemes.

### “What is the .42 in the mail?”

It is an unspecified agreement statistic from their shared annotation subset. The mail says much of the agreement comes from cases where parser predictions already matched. The final judgments and calculation are missing here, so the value cannot establish how reliably they identified valid alternative analyses. It is separate from the draft's parser F1 of .422.

### “How would RST produce an argument structure?”

Most plausibly, RST features would be inputs to another model trained against the corpus's gold argumentation layer. That does not require a deterministic conversion between the two annotation systems. The available code does not tell us whether he predicted relation labels on given links or attempted to recover entire argument structures.

### “Why might coarse relations be better for argumentation?”

The notes report that preliminary observation, but the scores and setup are missing. The intended argument is that broad rhetorical categories may align more consistently with the argumentative target. A distinction useful for recognizing generated writing need not help predict support or undercut relations. That is a hypothesis to check against the actual experiment.

### “Is the new parser just a faster replacement?”

It is a new backend integrated with the old pipeline, but it also changes checkpoints, relation spaces, segmentation, and document splitting. Its outputs form a new experiment. No measured speedup or downstream result from that branch was found here.

### “What should I trust most?”

For the mechanics, use the relevant branch's code. For the recorded HC3 numbers, use the canonical report and tables, with the reproduction limitations stated here. For research intentions and work outside Git, use the notes and mails while preserving their uncertainty. The unfinished paper mixes these layers and needs reconciliation.
