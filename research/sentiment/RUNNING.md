# Reproduce the sentiment experiments

For a new machine, start with [setting up a fresh checkout](#set-up-a-fresh-checkout).
The [meeting notes](replication/MEETING_NOTES.md) summarize the SFU and
Amazon experiments. The [dependency-weighting pilot](DEPENDENCY_PILOT.md)
explains and reproduces the later CardiffNLP and frozen mDeBERTa comparison.
The [bottom-up Amazon experiment](#combine-sentiment-up-the-amazon-trees)
tests whether fine or coarse labels help when combining sentiment up a tree.
The [stock-coarse tree comparison](#compare-stock-coarse-and-mapped-trees)
reparses the same SFU and Amazon EDUs with the public coarse checkpoints.
The [fine/coarse diagnostics](#finecoarse-diagnostics-with-log-loss-and-scrambled-labels)
compare log-loss and test fine labels against labels scrambled within their
coarse groups.
The sections below also preserve the original SFU
stage commands and reference outcomes.

The CPU text, length, and prior controls were completed on 2026-11-18.
The four-inventory parser pilot is documented below.

## Environment

From the repository root, use the separate project and its lockfile:

```bash
uv sync --project research/sentiment --cache-dir /tmp/tos-uv-cache
```

The recorded run used Python 3.11.15. The precise package versions and source
hashes are in `results/sfu-text-controls-v1.json`. The root legacy parser
requirements are not dependencies of this CPU preparation stage.

## Data and frozen folds

Obtain the English raw SFU archive from its
[corpus owner page](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html), then run:

```bash
python -B -m research.sentiment.prepare_corpus \
  --archive /path/to/SFU_Review_Corpus_Raw.zip \
  --output-dir artifacts/sentiment/sfu-v1/prepared --seed 42
```

The baseline config checks the archive version/hash, 400 documents, eight
domains, 25 per domain/polarity, and the exact declared folds. Preparation
preserves paragraph text; whitespace normalization is separate identity
metadata. Existing compatible output is reused; inconsistent output is
rejected. Generated reviews/models/predictions are ignored by Git.

## Validation and fitting

```bash
uv run --project research/sentiment --locked --no-sync --cache-dir /tmp/tos-uv-cache \
  python -B -m unittest discover -s research/sentiment/tests -p 'test_*.py'

uv run --project research/sentiment --locked --no-sync --cache-dir /tmp/tos-uv-cache \
  python -B -m research.sentiment.baseline \
  --prepared-dir artifacts/sentiment/sfu-v1/prepared \
  --output-dir artifacts/sentiment/sfu-v1/text-controls-v1
```

The fitting command requires a fresh output directory to prevent silently
mixing runs. There are five outer folds, four training-only inner folds,
and a fixed LR C grid. 14 tests check data/encoding, duplicates, fold integrity,
vocabulary fitting boundaries, metadata exclusion, Unicode JSONL handling,
prepared-file inconsistencies, and output corruption.

Verify an existing complete run without refitting:

```bash
uv run --project research/sentiment --locked --no-sync --cache-dir /tmp/tos-uv-cache \
  python -B -m research.sentiment.baseline \
  --prepared-dir artifacts/sentiment/sfu-v1/prepared \
  --output-dir artifacts/sentiment/sfu-v1/text-controls-v1 --verify-only
```

Independently verify the saved artifacts and raw source text using only the
Python standard library, without importing the training or preparation code:

```bash
python -B -m research.sentiment.verify_artifacts \
  --prepared-dir artifacts/sentiment/sfu-v1/prepared \
  --run-dir artifacts/sentiment/sfu-v1/text-controls-v1 \
  --archive /path/to/SFU_Review_Corpus_Raw.zip
```

The archive is optional; omit it to check the prepared/run artifacts alone.
The [recorded independent report](results/sfu-text-controls-v1-independent-verification.json)
includes the raw-archive check. This validates saved data, splits, predictions,
hashes and metrics; it does not independently refit the classifiers.

## Recorded reference outcomes

| Control | Pooled held-out macro-F1 | Accuracy |
|---|---:|---:|
| Training class prior | 0.3333 | 0.5000 |
| Length-only logistic regression | 0.6166 | 0.6175 |
| TF-IDF unigram/bigram logistic regression | 0.8274 | 0.8275 |

These controls establish a reproducible starting point. In particular, review
length already carries sentiment information here, which justifies retaining
length/structural controls in the later representation comparisons. No RST
effect, superiority of a framework, statistical contrast, or novel sentiment
claim follows from these results.

Full aggregate metrics, chosen C values, input/source/artifact hashes, and
environment are preserved in [the portable results](results/sfu-text-controls-v1.json).
The ignored run directory holds 400 predictions per control and fifteen fitted
fold/control models. The verifier recomputes metrics and checks identity,
fold membership, configuration, input/source hashes, and artifact hashes.

## Four-inventory parsing pilot

Use a separate uv environment for the parser stack. The pilot used Python
3.10.12, PyTorch 2.2.2+cu118, NumPy 1.26.4, Transformers 4.46.3,
Hugging Face Hub 0.26.5 and SentSplit 1.0.8. These commands install the
matching stack on an internet-connected Pascal-compatible NVIDIA host.
For a CPU run, install the CPU Torch wheel and omit `--cuda-device`.

```bash
uv venv --python 3.10 /tmp/tos-unirst-venv
uv pip install --python /tmp/tos-unirst-venv/bin/python --torch-backend cu118 \
  torch==2.2.2 numpy==1.26.4
uv pip install --python /tmp/tos-unirst-venv/bin/python \
  transformers==4.46.3 huggingface_hub==0.26.5 sentsplit==1.0.8 \
  setuptools==80.9.0 sentencepiece \
  'isanlp @ git+https://github.com/iinemo/isanlp.git@2a102e59f9718acc7fe259dd8d83c66d5da39794' \
  'rstconverter @ git+https://github.com/rst-workbench/rst-converter-service.git@b7eaa16f58f471e8ccc40344c509b5169e589cd2' \
  'isanlp_rst @ git+https://github.com/ruthenian8/isanlp_rst.git@cb73639acef90163833c4b5043669dc1e7cf0316'
```

Setuptools 80.9.0 retains `pkg_resources`, which SentSplit imports. The
personal GPU host had no package-index DNS access. We used an isolated uv
interpreter with a staged uv Python 3.10 dependency overlay ahead of the
existing CUDA package directory from `/scratch/rst_similarity/.venv`.

Use stock UniRST snapshot `b99b1c0f39435971b9daaa450e8476ef7b1a554a`.
The fine loader also looks up its base at
`9407970f1d9d2435b5f875a0cd14293a16646304`. The model files are identical
at both revisions. They can be linked to one set of locally verified files
in an isolated HF cache for offline inference. The parser resolves
`xlm-roberta-large` to tokenizer/config files from revision
`c23d21b0620b635a76227c604d44e43a9f0ee389`. The scene splitter resolves
`xlm-roberta-base` to revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
When operating offline, set the `main` refs for those two model aliases
in the pilot cache. Checkpoint folders `erst/` and `rstdt/` must be under
`--checkpoints`, with their original configs, weights and tables.

From the repository root, prepare the samples locally. The prepare command
reads the declared maps directly from the pinned research-branch commit
without importing that branch's code.

```bash
python -B -m research.sentiment.parser_pilot prepare \
  --sample /tmp/tos-pilot-fixtures.json
python -B -m research.sentiment.parser_pilot prepare \
  --prepared artifacts/sentiment/sfu-v1/prepared --sample /tmp/tos-pilot-sfu.json
```

Run the fixtures first, then change the sample and output paths for the SFU
run. Both output directories must be fresh. Use the path to the pinned
UniRST snapshot from your local HF cache in `--stock-snapshot`.

```bash
/tmp/tos-unirst-venv/bin/python -B -u -m research.sentiment.parser_pilot run \
  --sample /tmp/tos-pilot-fixtures.json \
  --output artifacts/sentiment/unirst-parser-pilot/fixtures \
  --checkpoints HC3_project --stock-snapshot /path/to/pinned/unirst-snapshot \
  --cuda-device 0
```

The SFU sample selects the floor-indexed 10th, 50th, and 90th percentile
reviews within each polarity by whitespace-token count, with document ID
as the tie-break. The runner uses the `feature/sentiment` scene splitter
with SentSplit and its default 492-token target. It uses the existing
offset-based PCC segmenter once per scene, normalizes EDU whitespace and
supplies the same EDUs to each parser's `from_edus` call. It verifies
ordered leaf text, native relation/nuclearity labels and binary structure
in memory, then checks leaf text, node IDs, group types and parent
attachments in the RS3 export. Each successful scene retains the binary
tree as JSON and a separate RS3 file. The report records inventory-wide
and observed map gaps, and emits projected relation counts only for
inventories with complete coverage of the observed labels.

### Pilot outcomes

Both fixtures passed on all four parsers. All six SFU reviews yielded a
complete set of 60 validated trees and RS3 files across 15 scenes and 568
shared EDUs. The longest scene was 492 tokens using the scene-split tokenizer.
The SFU run took 8.1 seconds for PCC segmentation, 7.2–12.1 seconds per
inventory for parsing and export checks, and 41–53 seconds per parser load
on the GTX 1080 Ti. These are small-run timings and exclude checkpoint
hashing and transfer.

| Inventory | Native relation names | Predicted on SFU pilot | Pilot map coverage |
|---|---:|---:|---:|
| PCC | 25 | 23 | 553/553 occurrences |
| NLDT | 27 | 22 | 553/553 occurrences |
| eRST | 32 | 32 | 553/553 occurrences |
| RST-DT | 69 | 35 | 553/553 occurrences |

The research-branch maps match the supplied eRST and RST-DT HC3 result maps.
One allowed NLDT label (`volitional-cause`) and nine allowed RST-DT labels
are absent from the maps. None occurred in this pilot. Get the assignments
or a declared handling rule before the full fine/coarse comparison. Native
alphabet gaps, artifact hashes, and timing are in the
[portable report](results/unirst-parser-pilot.json). The RS3 files, trees,
shared scenes, and logs are under `artifacts/sentiment/unirst-parser-pilot/`.
The source runs are under `/scratch/threads-of-subtlety/parser-pilot/` on
the personal GPU host.

## Full SFU parsing and label audit

Reuse the four-inventory pilot environment and checkpoints. Prepare the
full input manifest from the frozen SFU artifacts, with the same research
maps.

```bash
python -B -m research.sentiment.parse_sfu prepare \
  --prepared artifacts/sentiment/sfu-v1/prepared --sample /tmp/tos-sfu-full-v1.json
```

Start PCC first so it can write the shared `scenes.jsonl` and
`segmentation.json` files. Once both exist, the other inventories can
run independently on separate GPUs. Run the command once per inventory
using `deu.rst.pcc`, `nld.rst.nldt`, `eng.erst.gum`, and
`eng.rst.rstdt` as the inventory keys.

```bash
/tmp/tos-unirst-venv/bin/python -B -u -m research.sentiment.parse_sfu run \
  --sample /tmp/tos-sfu-full-v1.json --output artifacts/sentiment/sfu-full-v1 \
  --checkpoints HC3_project --stock-snapshot /path/to/pinned/unirst-snapshot \
  --inventory deu.rst.pcc --cuda-device 0
python -B -m research.sentiment.parse_sfu verify \
  --sample /tmp/tos-sfu-full-v1.json --output artifacts/sentiment/sfu-full-v1 \
  --report /tmp/tos-sfu-full-report.json
```

The native binary trees are stored in one JSONL file per inventory. The
runner records split, segmentation and parse failures by scene. The
verifier compares scene identities and ordered EDU leaves, recounts each
inventory's relations, and reports complete review coverage across all
four inventories. The pilot's RS3 files remain available for inspection.

If the released converter fails to align rendered subword text back to
the supplied EDUs, copy `scenes.jsonl` and `segmentation.json` into a
fresh output directory. Run the same command for each inventory with
`--retry-from` pointing to the completed first-pass directory. This
checks three previously successful trees, copies the successful records
unchanged, and retries only known text-alignment errors. On retry the
decoder receives the same inputs, and the converter reads leaf text
directly from the original EDUs.

The completed SFU run produced 1,012 scenes and 36,850 EDUs, with 271
reviews split across multiple scenes. The first pass hit text-conversion
errors on the same 31 scenes in each inventory. The targeted converter
retry recovered all 124 parses. All 3,924 first-pass successful trees
remained exactly equal, and all 60 trees from the original six-review
pilot matched the full run. Across the 4,048 verified trees, the original
research maps cover all observed labels in each inventory.

The raw run directories are `artifacts/sentiment/sfu-full-v1-first-pass/`
and `artifacts/sentiment/sfu-full-v1/` locally, and `sfu-full-v1/` and
`sfu-full-v1-verified/` under the personal GPU-host pilot workspace. The
[portable summary](results/sfu-full-parser-audit.json) records both passes,
the ordered label tables, and the checksums.

### Export verified trees to RS3

Run this from the repository root with `isanlp` available in the Python
environment, as for the pilot. The exporter reconstructs the saved units
and uses `DiscourseUnit.to_rs3()` without running inference again.

```bash
python -B -m research.sentiment.export_rs3 \
  --corpus SFU --trees-dir artifacts/sentiment/sfu-full-v1 \
  --audit research/sentiment/results/sfu-full-parser-audit.json \
  --output-dir artifacts/sentiment/share/rs3
```

The layout is `SFU/<standard>/<fine|coarse>/<scene>.rs3`. The exporter
verifies the source checksums and checks every RS3 against its source
tree and EDU sequence. The scene index links filenames to document IDs.
When a relation label is used for both a nucleus–satellite and a
multinuclear attachment, the multinuclear use takes a `-multinuc`
suffix. The included `rs3-relation-aliases.json` reverses this spelling
for label comparisons. To export Amazon or UCI, use that corpus name and
the corresponding run's `parser-verified/` and `parser-audit.json`.

The SFU export has 8,096 files. All 60 fine pilot bodies match after
decoding aliases. The rstWeb reader imported all files with the expected
parent links and attachment types.

### Token length audit

The [length audit](results/sfu-token-lengths.json) records counts for the
400 canonical review texts and 36,850 cached EDUs. It uses Tokenizers
0.21.4, the pinned `tokenizer.json` files listed by model revision and
SHA-256 in the report, special tokens enabled, and padding and truncation
disabled. Review lengths come from the `text` field of the prepared
`reviews.jsonl`. EDU lengths come from each element of `edus` in the
verified `scenes.jsonl`. The 95th percentile uses linear interpolation
between adjacent sorted counts.

## Frozen SFU EDU scores

The scorer uses the quantized CardiffNLP ONNX export from the EDU check.
Run it from the repository root with the verified shared scene cache.
The script checks the three model files and scene file against their
recorded checksums before inference.

```bash
uv venv --python 3.12 /tmp/tos-sfu-scorer
uv pip install --python /tmp/tos-sfu-scorer/bin/python \
  numpy==2.2.6 onnxruntime==1.22.1 tokenizers==0.21.4
mkdir -p /tmp/tos-cardiff-sfu
scorer_url='https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest/resolve/f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445'
curl -L --fail "$scorer_url/onnx/model_quantized.onnx" -o /tmp/tos-cardiff-sfu/model.onnx
curl -L --fail "$scorer_url/tokenizer.json" -o /tmp/tos-cardiff-sfu/tokenizer.json
curl -L --fail "$scorer_url/config.json" -o /tmp/tos-cardiff-sfu/config.json
/tmp/tos-sfu-scorer/bin/python -B -m research.sentiment.score_sfu \
  --scenes artifacts/sentiment/sfu-full-v1/scenes.jsonl \
  --model-dir /tmp/tos-cardiff-sfu --output artifacts/sentiment/sfu-edu-scores-v1
/tmp/tos-sfu-scorer/bin/python -B -m research.sentiment.score_sfu \
  --scenes artifacts/sentiment/sfu-full-v1/scenes.jsonl \
  --output artifacts/sentiment/sfu-edu-scores-v1 --verify
```

`scores.jsonl` stores the review ID, scene key, and ordered rows of
`[negative, neutral, positive]` probabilities in the original EDU order.
All 36,850 vectors are finite and sum to one within `1e-6`. Inference
groups similar token lengths into batches of 64 and restores the original
order when saving. The full CPU inference pass took 81 seconds. A second
run on the same host with a fresh `--output` produced a byte-identical
`scores.jsonl`. Compared with the exploratory 32-EDU run, which used
different batches, one top label changed. Use this cache throughout the
comparison to hold the scores fixed. The cache is local and its checksum
and model provenance are in the [portable manifest](results/sfu-edu-scores-v1.json).

## Equal-weight EDU baseline

Use the locked sentiment environment and the frozen score cache. The
script checks the scores, prepared corpus, splits and LR config against
the recorded runs. Pass a fresh output directory.

```bash
uv run --project research/sentiment --locked --no-sync --cache-dir /tmp/tos-uv-cache \
  python -B -m research.sentiment.edu_baseline \
  --prepared-dir artifacts/sentiment/sfu-v1/prepared \
  --scores artifacts/sentiment/sfu-edu-scores-v1/scores.jsonl \
  --output-dir artifacts/sentiment/sfu-v1/edu-mean-v1
```

Each review supplies three means, one per scorer class. Every EDU has
equal weight, including across scenes of unequal length. Logistic
regression uses the same C grid and frozen folds as the text controls.
Scaling and C selection use only training data for each outer test fold.

The baseline scored 0.8174 pooled macro-F1 and 81.75% accuracy over 400
held-out reviews. An independent check rebuilt the review means, refit
all inner and outer models, and verified the saved probabilities and
metrics. A repeat on the same host produced identical predictions and
report. The [portable result](results/sfu-edu-mean-v1.json) records fold
metrics, C values, software versions, and hashes.

## Nuclearity control

Use the locked sentiment environment with the same prepared data, score
cache, and baseline predictions as above, plus the four verified tree
files. Pass a fresh output directory.

```bash
uv run --project research/sentiment --locked --no-sync --cache-dir /tmp/tos-uv-cache \
  python -B -m research.sentiment.nuclearity \
  --prepared-dir artifacts/sentiment/sfu-v1/prepared \
  --scores artifacts/sentiment/sfu-edu-scores-v1/scores.jsonl \
  --trees-dir artifacts/sentiment/sfu-full-v1 \
  --uniform-predictions artifacts/sentiment/sfu-v1/edu-mean-v1/predictions.jsonl \
  --output-dir artifacts/sentiment/sfu-v1/nuclearity-v1
```

An EDU is central if every branch from its scene root to the EDU is a
nucleus branch. An EDU below any satellite branch is supporting. A scene
with one EDU contributes that EDU to the central group. Pool each group
across a review's scenes and give logistic regression six inputs, the
three sentiment averages for each group. Use the review mean if the
supporting group is empty. Every SFU review had both groups in all four
inventories. Training and model selection reuse the baseline protocol.

| Grouping | Held-out macro-F1 | Reviews corrected | New errors |
|---|---:|---:|---:|
| Equal-weight baseline | 0.8174 | | |
| PCC | 0.8225 | 12 | 10 |
| NLDT | 0.8324 | 13 | 7 |
| GUM/eRST | 0.8250 | 12 | 9 |
| RST-DT | 0.8224 | 11 | 9 |

An independent check walked from each EDU to its scene root, rebuilt the
features, and refit all inner and outer models. All 1,600 held-out
predictions matched. Applying the research-branch coarse maps to every
tree left all six inputs identical for every review and inventory. A
repeat produced byte-identical outputs. The [portable report](results/sfu-nuclearity-v1.json)
records coverage, metrics, and checksums.

## Fine/coarse relation pooling

Reuse the verified trees, scorer cache, frozen folds, and previous
predictions. The command reads the research maps from the pinned Git
revision in the parser audit and requires a fresh output directory.

```bash
uv run --project research/sentiment --locked --no-sync --cache-dir /tmp/tos-uv-cache \
  python -B -m research.sentiment.relation_pooling \
  --prepared-dir artifacts/sentiment/sfu-v1/prepared \
  --scores artifacts/sentiment/sfu-edu-scores-v1/scores.jsonl \
  --trees-dir artifacts/sentiment/sfu-full-v1 \
  --uniform-predictions artifacts/sentiment/sfu-v1/edu-mean-v1/predictions.jsonl \
  --nuclearity-predictions artifacts/sentiment/sfu-v1/nuclearity-v1/predictions.jsonl \
  --output-dir artifacts/sentiment/sfu-v1/relation-pooling-v1
```

Take the three EDU sentiment means in each child span of each binary
node. For each review and relation name, average those left and right
vectors over all occurrences, across scenes. Append them to the six
nuclearity inputs. Columns come from the declared parser alphabets.
An absent relation contributes zeros, so the encoding also carries
relation-presence information. For coarse labels, merge the sums and
counts before averaging so each original node retains equal weight.
All 35,838 nodes per inventory contribute to both conditions.

| Inventory | Nuclearity macro-F1 | Fine | Coarse | Fine minus coarse | Fine / coarse input columns |
|---|---:|---:|---:|---:|---:|
| PCC | 0.8225 | 0.7900 | 0.7875 | +0.0025 | 156 / 90 |
| NLDT | 0.8324 | 0.7975 | 0.8000 | −0.0025 | 168 / 90 |
| GUM/eRST | 0.8250 | 0.7675 | 0.7800 | −0.0125 | 198 / 96 |
| RST-DT | 0.8224 | 0.7700 | 0.8000 | −0.0300 | 420 / 114 |

An independent check rebuilt features directly from child spans, applied
the coarse mapping occurrence by occurrence, and refit all models. All
3,200 held-out predictions matched. A repeat produced byte-identical
outputs. The [aggregate report](results/sfu-relation-pooling-v1.json)
includes per-fold results, paired errors, label counts, and checksums.
Interpret these first differences together with the change in input
dimensions and relation sparsity.

## Fine/coarse stability

```bash
uv run --project research/sentiment --locked --no-sync --cache-dir /tmp/tos-uv-cache \
  python -B -m research.sentiment.relation_stability \
  --predictions artifacts/sentiment/sfu-v1/relation-pooling-v1/predictions.jsonl \
  --manifest artifacts/sentiment/sfu-v1/prepared/manifest.jsonl \
  --output artifacts/sentiment/sfu-v1/relation-pooling-v1/stability.json
```

Each of 20,000 bootstrap draws resamples reviews within their original
fold, domain, and gold-label group (seed 42). Both conditions and all four
inventories use the same sampled reviews. Report the central 95% of
fine-minus-coarse differences, in macro-F1 points on a 0–100 scale.

| Inventory | Fine minus coarse | Paired resampling interval | Folds favoring fine / coarse / tied |
|---|---:|---:|---:|
| PCC | +0.25 | [−2.50, +3.01] | 2 / 3 / 0 |
| NLDT | −0.25 | [−3.24, +2.75] | 2 / 3 / 0 |
| GUM/eRST | −1.25 | [−4.25, +1.75] | 1 / 4 / 0 |
| RST-DT | −3.00 | [−6.24, +0.25] | 2 / 2 / 1 |

All intervals include zero. RST-DT favors coarse when any one fold is
omitted from the pooled predictions, with gaps of 1.23–4.10 points.
The intervals describe the sensitivity to which reviews are scored, with
the fitted models and split assignments held fixed. Training and split
variation fall outside this check. [Cross-validation variance](https://www.jmlr.org/papers/v5/grandvalet04a.html)

An independent confusion-count calculation reproduced the intervals,
and scikit-learn confirmed the fold results, domain results, and a sample
of bootstrap scores. A repeat produced a byte-identical report. The
[portable result](results/sfu-relation-stability-v1.json) includes all
fold, leave-one-fold-out, and domain gaps.

## Relation pooling with nucleus first

```bash
uv run --project research/sentiment --locked --no-sync --cache-dir /tmp/tos-uv-cache \
  python -B -m research.sentiment.relation_pooling \
  --prepared-dir artifacts/sentiment/sfu-v1/prepared \
  --scores artifacts/sentiment/sfu-edu-scores-v1/scores.jsonl \
  --trees-dir artifacts/sentiment/sfu-full-v1 \
  --uniform-predictions artifacts/sentiment/sfu-v1/edu-mean-v1/predictions.jsonl \
  --nuclearity-predictions artifacts/sentiment/sfu-v1/nuclearity-v1/predictions.jsonl \
  --nucleus-first artifacts/sentiment/sfu-v1/relation-pooling-v1/predictions.jsonl \
  --output-dir artifacts/sentiment/sfu-v1/relation-nucleus-first-v1
```

For nodes with a nucleus and a satellite, put the nucleus span mean
first and the satellite span mean second. This swaps the two triples at
each `SN` node before pooling by label. `NS` and two-nucleus (`NN`)
nodes keep their text order. A label shared by mononuclear and
multinuclear nodes can therefore still mix roles in its pooled slots.
The tree, labels, dimensions, six review-level nuclearity features,
EDU scores, folds, and model selection protocol are shared with the
text-order run.

| Inventory | Fine macro-F1 | Change vs. text order (points) | Coarse macro-F1 | Change vs. text order (points) |
|---|---:|---:|---:|---:|
| PCC | 0.7850 | −0.50 | 0.7850 | −0.25 |
| NLDT | 0.7850 | −1.25 | 0.7974 | −0.26 |
| GUM/eRST | 0.7875 | +2.00 | 0.7900 | +1.00 |
| RST-DT | 0.7650 | −0.50 | 0.7999 | −0.01 |

GUM/eRST gained with both label sets. Every nucleus-first run scored
below its nuclearity-only control. All eight paired 95% intervals for
the difference from text order and all four within-run fine/coarse
intervals include zero. The intervals use the same 20,000 stratified
resamples of fixed, held-out review predictions as above. The
[portable report](results/sfu-relation-nucleus-first-v1.json) records
the intervals, per-fold results, and input hashes.

An independent extraction using direct child-span slices and
occurrence-level coarsening reproduced every feature row and all 3,200
held-out decisions. A repeat produced byte-identical predictions and
report. The original text-order command still reproduces its saved
predictions byte for byte.

## HC3-style motifs on SFU

```bash
uv run --project research/sentiment --locked --no-sync --with networkx==3.1 \
  --cache-dir /tmp/tos-uv-cache python -B -m research.sentiment.motif_sfu \
  --prepared-dir artifacts/sentiment/sfu-v1/prepared \
  --trees-dir artifacts/sentiment/sfu-full-v1 \
  --output-dir artifacts/sentiment/sfu-v1/motif-v1
```

We convert the same scene trees into the directed, labeled HC3 graphs
and count connected three-node motifs (M3). As in the HC3 code, motif
frequency is relative to retained occurrences, and depth is the mean
root distance normalized by the scene graph diameter and weighted by
occurrences across a review. Symmetric matches count with their original
multiplicity. In each inner or outer fitting partition, the catalog
contains all fine motifs seen in its training reviews. The coarse version
merges this same catalog with the research maps, preserving retained
occurrence counts. This transfers the feature construction to SFU with
its own training catalogs, LR setup, and macro-F1 evaluation.

| Inventory | Fine macro-F1 | Coarse macro-F1 | Fine minus coarse (points) | Paired 95% interval (points) |
|---|---:|---:|---:|---:|
| PCC | .582 | .600 | −1.75 | [−6.22, +2.67] |
| NLDT | .602 | .557 | +4.51 | [−0.23, +9.27] |
| GUM/eRST | .607 | .600 | +0.77 | [−3.73, +5.19] |
| RST-DT | .600 | .582 | +1.78 | [−2.64, +6.25] |

All four fine/coarse and all twelve between-inventory intervals include
zero. Fine is numerically ahead in three inventories with motifs and in
one with sentiment pooling. Motif-only scores fall below the SFU
length-only control (.617). These are paired, pointwise intervals from the
same review draws as the sentiment-pooling stability check. They hold
the trained classifiers and splits fixed. [Aggregate results](results/sfu-motif-v1.json)

We checked the extractor against the saved HC3 motif features for six
answers under all four inventories. Frequencies and depths matched within
`1e-7`. An independent reconstruction of the SFU feature groups and
nested fits reproduced all 3,200 class decisions and all paired intervals.
The largest probability difference was .000271. The complete rerun
produced byte-identical predictions and reports.

## Recheck the saved HC3 granularity scores

The later [HC3 reproduction checks](HC3_REPRODUCTION.md) also refit LR and
RF from the saved features and regenerate reference features for six
original test documents from the answer text.

Use the locked CPU sentiment environment to recompute the LR MF+WAD test F1
from the supplied HC3 predictions and each run's validation-selected
threshold. The feature matrices supply an independent check of test
identities and gold labels. This verifies the saved evaluation outputs.

```bash
research/sentiment/.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from research.sentiment.parser_pilot import digest

root = Path("HC3_project/artifacts")
for standard in sorted((root / "granularity/lr_mf_wad/predictions").iterdir()):
    features = root / "feature_matrices" / standard.name / "hc3_test.features.npz"
    with np.load(features, allow_pickle=False) as matrix:
        ids, labels = matrix["identity_id"], matrix["y"]
    scores = {}
    for granularity in ("fine", "coarse"):
        directory = standard / granularity
        predictions = directory / "predictions.npz"
        metadata = json.loads((directory / "metrics.json").read_text())
        assert digest(predictions) == metadata["prediction_sha256"]
        with np.load(predictions, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["test_identity_id"], ids)
            np.testing.assert_array_equal(saved["test_y"], labels)
            threshold = float(saved["threshold"])
            assert threshold == metadata["selection"]["validation_selected_threshold"]
            score = f1_score(labels, saved["test_probability"] >= threshold)
        assert abs(score - metadata["test_metrics"]["f1"]) < 1e-12
        scores[granularity] = score
    print(standard.name, len(ids), scores, scores["fine"] - scores["coarse"])
PY
```


## Compare stock coarse and mapped trees

The sentiment experiments parse once using the fine checkpoint and then
rename its relations with the research map. To see what changes when using
the released coarse parser, run stock GUM/eRST and RST-DT on the exact same
PCC EDUs and scene boundaries. The runner uses the pinned stock UniRST
snapshot and the same parser environment as the original parsing pilot.
It retries text-alignment errors with the supplied-EDU converter and
validates the leaf text and binary tree for every scene. Use the converter
from the Amazon replay, which tokenizes the joined EDUs before aligning
punctuation. Apply its patch to an isolated copy of the parser source.

Set `sfu_trees` and `amazon_trees` to the original verified parse directories
and `stock_snapshot` to the released checkpoint described above. Use the
parser Python environment for both steps and a fresh `stock_output`.
Run one parser per GPU when two GPUs are available.

```bash
parser_python=/path/to/parser/venv/bin/python
stock_source="$(mktemp -d)"
cp --parents research/sentiment/{stock_coarse.py,parse_sfu.py,parser_pilot.py} "$stock_source"
cp --parents research/sentiment/{results/sfu-full-parser-audit.json,replication/results/amazon/parser-audit.json} "$stock_source"
amazon_patch="$PWD/research/sentiment/replication/amazon.patch"
(cd "$stock_source" && git apply --include=research/sentiment/parse_sfu.py "$amazon_patch")

# Use absolute paths for the checkpoint, tree directories, and output.
for inventory in eng.erst.gum eng.rst.rstdt; do
  (cd "$stock_source" && "$parser_python" -B -u -m research.sentiment.stock_coarse \
      --sfu-scenes "$sfu_trees/scenes.jsonl" --amazon-scenes "$amazon_trees/scenes.jsonl" \
      --stock-snapshot "$stock_snapshot" --inventory "$inventory" --cuda-device 0 \
      --output "$stock_output/$inventory")
done

"$parser_python" -B -m research.sentiment.compare_stock_coarse \
  --stock "$stock_output" --sfu-trees "$sfu_trees" --amazon-trees "$amazon_trees" \
  --report "$stock_output/comparison.json" --rs3-output "$stock_output/rs3"
```

The comparison matches nodes by their EDU spans and split points. For
scenes with at least two EDUs, it counts how many trees have the same
branching, how many also agree on nuclearity, and how many agree on all
coarse labels as well. Single-EDU scenes have no split and get their own
count. The RS3 output uses `<corpus>/<standard>/stock-coarse/<scene>.rs3`
and includes a scene index for joining the exports to the mapped trees.
If a parse run was interrupted, add `--resume-from` with the prior
inventory directory and use a fresh output directory. The runner checks
scene identities for the saved prefix and reparses three sample trees per
corpus before reusing it.

| Corpus | Standard | Scenes with 2+ EDUs | Branching changed | Branching, nuclearity and coarse labels all match |
|---|---|---:|---:|---:|
| Amazon | GUM/eRST | 1,999 | 667 (33.4%) | 450 |
| Amazon | RST-DT | 1,999 | 813 (40.7%) | 245 |
| SFU | GUM/eRST | 1,011 | 843 (83.4%) | 37 |
| SFU | RST-DT | 1,011 | 895 (88.5%) | 25 |

The stock and mapped trees used the same set of coarse relation names in
each comparison. For example, on the three-EDU Amazon scene `d1371-s00`,
stock RST-DT groups EDU 2 with EDU 3, and the mapped tree groups EDU 1
with EDU 2. Mapping after parsing isolates label granularity. Replacing
the fine parser with stock UniRST also changes the tree features.

The recovered run reused 1,012 SFU trees and 678 Amazon trees per stock
inventory after checking their scene identities and reproducing the
sample trees. The [report](results/stock-coarse-v1.json) records the
checkpoint, both stages' tree hashes, and the independent whole-tree
checks plus the 6,024 verified RS3 exports.

### Rerun sentiment with the stock trees

Both stock parsers take the saved PCC EDU boundaries. Stock relation names
are used as predicted. The reference uses the colleague's fine checkpoints
and maps their relation names after parsing. The review labels, folds,
and frozen EDU representations match across parser choices.

The table reports pooled held-out macro-F1 for the relation-aware version
of each method. Each cell shows mapped coarse → stock coarse.

| Corpus | Standard | Original LR pooling | Learned link weights with CardiffNLP | Bottom-up with CardiffNLP |
|---|---|---:|---:|---:|
| SFU | GUM/eRST | .7800 → .8050 | .8450 → .8375 | |
| SFU | RST-DT | .8000 → .7774 | .8300 → .8274 | |
| Amazon | GUM/eRST | .8775 → .8720 | .8855 → .8790 | .8805 → .8825 |
| Amazon | RST-DT | .8780 → .8885 | .8775 → .8825 | .8745 → .8855 |

All four paired intervals for original LR pooling and all four for
CardiffNLP link weights include zero. Original LR pooling stays below
nuclearity alone on SFU and above it on Amazon for both standards. The
bottom-up experiment covers Amazon. For RST-DT its stock-coarse run
gains 1.10 macro-F1 points over mapped coarse with a pointwise interval
of +0.10 to +2.10. The frozen mDeBERTa runs remain below the CardiffNLP
runs. [All conditions and paired intervals](results/stock-coarse-sentiment-v1.json)

Run the following from the repository root. Set `stock` to the verified
stock output above and `runs` to a fresh output directory. For each corpus
set `prepared`, `trees`, `scores`, and `embeddings` to the original
prepared set, parse directory, frozen CardiffNLP scores, and cached
mDeBERTa embeddings.

```bash
uv sync --project research/sentiment --locked
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
runner=(uv run --project research/sentiment --locked --no-sync python -B -m)
common=(--prepared "$prepared" --trees "$trees")
"${runner[@]}" research.sentiment.stock_pooling "${common[@]}" \
  --corpus "$corpus" --stock "$stock" --representation "$scores" --output "$runs/$corpus-pooling"
for arm in cardiff mdeberta; do
  representation="$scores"
  if [ "$arm" = mdeberta ]; then representation="$embeddings"; fi
  for source in mapped stock; do
    stock_args=()
    prefix=mapped-
    if [ "$source" = stock ]; then stock_args=(--stock "$stock"); prefix=; fi
    "${runner[@]}" research.sentiment.dependency_sentiment "${common[@]}" "${stock_args[@]}" \
      --corpus "$corpus" --arm "$arm" --representation "$representation" --output "$runs/$prefix$corpus-$arm"
  done
done
if [ "$corpus" = amazon ]; then
  "${runner[@]}" research.sentiment.recursive_sentiment "${common[@]}" \
    --representation "$scores" --output "$runs/mapped-amazon-recursive"
  "${runner[@]}" research.sentiment.recursive_sentiment "${common[@]}" --stock "$stock" \
    --representation "$scores" --output "$runs/amazon-recursive"
fi
```

After both corpora finish, generate the report using the two prepared sets.
The report generator validates each held-out row and checks that the
refitted link-weight and bottom-up predictions reproduce the original
published file hashes. Its paired intervals hold model fits fixed.

```bash
"${runner[@]}" research.sentiment.stock_sentiment_report --runs "$runs" \
  --sfu-prepared "$sfu_prepared" --amazon-prepared "$amazon_prepared" \
  --output "$runs/comparison.json"
```

## Combine sentiment up the Amazon trees

**Question.** When the model combines sentiment step by step up the RST
tree, does knowing the fine label help?

1. Use the 2,000 Amazon reviews, the same four sets of trees, and the same
   frozen CardiffNLP predictions. Start each EDU with one score, its
   positive probability minus its negative probability.
2. Combine the two sides of each split into a single score. Multiply each
   side's score by its main or supporting weight, add the two, and apply
   `tanh` to keep the result between −1 and +1. Use the main weight on both
   sides of a multinuclear split. Repeat until reaching the review root.
   This follows the composition rule in [Bhatia et al., §4](https://aclanthology.org/D15-1263.pdf#page=4).
3. Fit the weights and a logistic output on the root score using the review
   labels. The shared version learns one pair of weights for every split.
   Coarse learns one pair per coarse label, and fine learns one per fine
   label. All three use identical trees within a standard. All fits use
   binary log loss and L2 with fixed `C=1`, centered at 0.5 for the two
   composition weights, 1 for the output scale, and 0 for the bias.
4. Use the frozen five folds and a 0.5 threshold. Evaluate pooled held-out
   macro-F1. Paired 95% pointwise intervals use 20,000 resamples within
   fold, category, and gold label and keep the fitted models fixed.

| Standard | Shared rule | Coarse | Fine | Fine − coarse, F1 points (95% interval) |
|---|---:|---:|---:|---:|
| PCC | .8765 | .8835 | .8830 | −0.05 [−0.80, +0.65] |
| NLDT | .8745 | .8845 | .8820 | −0.25 [−0.85, +0.35] |
| GUM/eRST | .8715 | .8805 | .8790 | −0.15 [−0.95, +0.65] |
| RST-DT | .8670 | .8745 | .8755 | +0.10 [−0.75, +0.95] |

Both label-specific versions scored higher than the shared rule in all
four standards. Only the NLDT coarse gain had a pointwise interval that
excluded zero (+1.00 F1 points, interval +0.05 to +2.00). Fine and coarse
were within 0.25 points throughout, and all four fine–coarse intervals
included zero. This gives no clear granularity winner for bottom-up
composition on the Amazon cohort. Regularization and the supervised
mapping from local scores to a review prediction are specific to this
experiment.

After the [Amazon replay](#replay-sfu-amazon-and-uci-from-raw-data), point
`prepared` at the prepared reviews, `trees` at the verified parse folder,
and `scores` at the frozen CardiffNLP `scores.jsonl`. Run this from the repo root.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --project research/sentiment --locked --no-sync python -B \
  -m research.sentiment.recursive_sentiment \
  --prepared "$prepared" --trees "$trees" --representation "$scores" \
  --output artifacts/sentiment/amazon-recursive-v1
```

The [aggregate report](results/amazon-recursive-v1.json) records verified
input hashes, all paired comparisons, and the held-out prediction hash.

## Replay SFU Amazon and UCI from raw data

The aggregate reports and source manifests are under
[replication/results](replication/results). Amazon uses 2,000 reviews from
four categories, written from January 2022 through August 2023. The
preparation code fixes the length filter, cohort quotas, and star-to-polarity
mapping. UCI uses every released sentence and the authors' binary labels.
Its folds keep identical normalized texts together across categories and
both levels of cross-validation. Paired UCI intervals resample those
duplicate groups together.

Both runs use the published SFU scorer, parser checkpoints, and training
settings. The staging script starts from commit
25499ace90e23203da204abb92c32fb0ab8de41b. It gives the external cohorts their
own preparation checks, provenance, and outputs. The converter uses the
parser's joined-EDU tokenization for alignment retries. The original SFU
scripts and checksum references are preserved.

### Set up a fresh checkout

Use `sentiment2` on Max's fork. The [meeting notes](replication/MEETING_NOTES.md)
explain the model inputs and review-level findings. The
[fresh SFU audit](replication/results/sfu-reproduction.json) records what
matched on the RTX 4080 under WSL2. The result JSON files are portable.
Inference also requires the collaborator's original `erst/` and
`rstdt/` checkpoint folders, each with its weights, config, and relation
table.

From the repository root, configure the assets directory and verify that
the canonical mapping commit is available. If the check fails, fetch
`research/unirst2-canonical` from the fork.

```bash
scripts="$PWD/research/sentiment/replication"
assets="$PWD/artifacts/sentiment/assets"
export UNIRST_CHECKPOINTS="$PWD/HC3_project"
mkdir -p "$assets"
git cat-file -e f06309b62cbc53dce0c98668cac225666dfb64d7^{commit}
```

Create the three runtimes. For CPU parsing, use `--torch-backend cpu`
and set `CUDA_DEVICE=-1` when running the parser.

```bash
uv venv --python 3.10 "$assets/parser-venv"
uv pip install --python "$assets/parser-venv/bin/python" --torch-backend cu118 \
  torch==2.2.2 numpy==1.26.4
uv pip install --python "$assets/parser-venv/bin/python" \
  transformers==4.46.3 tokenizers==0.20.3 huggingface_hub==0.26.5 \
  sentsplit==1.0.8 setuptools==80.9.0 sentencepiece==0.2.3 \
  'isanlp @ git+https://github.com/iinemo/isanlp.git@2a102e59f9718acc7fe259dd8d83c66d5da39794' \
  'rstconverter @ git+https://github.com/rst-workbench/rst-converter-service.git@b7eaa16f58f471e8ccc40344c509b5169e589cd2' \
  'isanlp_rst @ git+https://github.com/ruthenian8/isanlp_rst.git@cb73639acef90163833c4b5043669dc1e7cf0316'

uv venv --python 3.12 "$assets/scorer-venv"
uv pip install --python "$assets/scorer-venv/bin/python" \
  numpy==2.2.6 onnxruntime==1.22.1 tokenizers==0.21.4

UV_PROJECT_ENVIRONMENT="$assets/classifier-exact-venv" \
  uv sync --project research/sentiment --locked --python 3.11.15
uv pip install --python "$assets/classifier-exact-venv/bin/python" networkx==3.1
```

The [package lists](replication/runtime/) record the environment actually
used under WSL2 on the RTX 4080. If your uv release lacks the 3.11.15
interpreter, install the archived build for your platform and pass its
Python executable to `uv sync --python`. [Python 3.11.15 build used for the replay](https://github.com/astral-sh/python-build-standalone/releases/tag/20260324)

Stage the pinned public weights and tokenizers. The helper verifies the
local fine checkpoints and the stock UniRST and CardiffNLP files before
inference. It also sets the pinned tokenizer aliases for offline parsing.

```bash
"$assets/parser-venv/bin/python" -B "$scripts/prepare-model-assets.py" \
  --assets "$assets" --checkpoints "$UNIRST_CHECKPOINTS"
```

### Raw inputs

Keep raw files and generated per-review artifacts under the ignored
`artifacts/sentiment/` directory. Fetch the TweetEval training file for
both external corpora. The published cohort reports record the expected
data and model hashes.

```bash
tweeteval_revision=3eb5e91dd5070b5a5931df4e70cbab5978e3c50c
tweeteval_train="$assets/tweeteval-sentiment-train.txt"
curl -fL "https://raw.githubusercontent.com/cardiffnlp/tweeteval/$tweeteval_revision/datasets/sentiment/train_text.txt" \
  -o "$tweeteval_train"
```

For SFU, use the official English raw ZIP as `sfu_zip`. For UCI, use the
official Sentiment Labelled Sentences ZIP as `uci_zip`. [SFU source](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html) · [UCI source](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences)

```bash
sfu_zip="$assets/SFU_Review_Corpus_Raw.zip"
uci_zip="$assets/uci-sentiment-labelled-sentences.zip"
curl -fL https://www.sfu.ca/~mtaboada/docs/research/SFU_Review_Corpus_Raw.zip -o "$sfu_zip"
curl -fL 'https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip' -o "$uci_zip"
```

For Amazon, create a fresh run directory and download its four source
archives and the language ID model there. The expected archive hashes
are in the [Amazon cohort manifest](replication/results/amazon/cohort.json).

```bash
run="$(mktemp -d "$PWD/artifacts/sentiment/amazon23-short-v1.XXXXXX")"
amazon_source=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories
for category in All_Beauty Appliances Musical_Instruments Software; do
  curl -fL "$amazon_source/$category.jsonl.gz" -o "$run/$category.jsonl.gz"
done
curl -fL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -o "$run/lid.176.ftz"
cp "$tweeteval_train" "$run/tweeteval-sentiment-train.txt"
printf '%s\n' "$tweeteval_revision" > "$run/tweeteval-revision.txt"
uv venv --python 3.10 "$assets/amazon-prep-venv"
prep_python="$assets/amazon-prep-venv/bin/python"
uv pip install --python "$prep_python" fasttext-wheel==0.9.2 numpy==1.26.4 tokenizers==0.20.3
```

### Prepare and run

Use a fresh directory beneath the checkout for each run. The SFU command
builds the full comparison-compatible layout and uses the published
checksum references for its scorer and downstream inputs.

```bash
sfu_run="$(mktemp -d "$PWD/artifacts/sentiment/sfu-repro.XXXXXX")"
bash "$scripts/run-sfu.sh" "$sfu_run" "$assets" "$sfu_zip"
```

The scorer and downstream SFU stages check the historical SHA-256 values.
If a different machine produces a mismatch, inspect the preceding stage
report to locate the first difference. The WSL2 replay matched the EDU
and discourse predictions byte for byte. Its text-control metrics matched
with different TF–IDF artifact hashes. [Fresh SFU audit](replication/results/sfu-reproduction.json)

For Amazon, use the run directory populated above. For UCI, create a
fresh directory. Each preparation regenerates the corpus and folds
before any fitting.

```bash
bash "$scripts/stage.sh" "$run" amazon
(
  cd "$run/source"
  "$prep_python" -B -m research.sentiment.prepare_amazon \
    --run-dir "$run" \
    --tokenizer "$assets/hf/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/tokenizer.json"
)
cmp "$run/baseline_config.json" "$scripts/results/amazon/baseline-config.json"
python -B "$scripts/verify-amazon.py" --run-dir "$run" > "$run/cohort-verification.json"
```

```bash
run="$(mktemp -d "$PWD/artifacts/sentiment/uci-sls-v1.XXXXXX")"
bash "$scripts/stage.sh" "$run" uci
(
  cd "$run/source"
  "$assets/classifier-exact-venv/bin/python" -B -m research.sentiment.prepare_uci \
    --run-dir "$run" --archive "$uci_zip" --tweeteval-train "$tweeteval_train"
)
cmp "$run/baseline_config.json" "$scripts/results/uci/baseline-config.json"
```

Then fit the text controls, parse all four inventories, and run the
remaining classifier and uncertainty comparisons.

```bash
(
  cd "$run/source"
  cp "$run/baseline_config.json" research/sentiment/baseline_config.json
  "$assets/classifier-exact-venv/bin/python" -B -m research.sentiment.baseline \
    --prepared-dir "$run/prepared" --output-dir "$run/text-controls"
)
bash "$scripts/run-parser.sh" "$run" "$assets"
bash "$scripts/run-classifiers.sh" "$run" "$assets"
```

The parser runner defaults to one process on GPU 0. Set
`UNIRST_JOBS=2` to share the GPU between two inventory processes after
PCC segmentation when it has enough memory. Set `CUDA_DEVICE=-1` with
CPU Torch to run sequentially on CPU.

### Read and verify the comparisons

The [three-corpus report](replication/results/uci/comparison.json) contains
all 32 macro-F1 and accuracy comparisons, parser coverage, and paired
intervals. The [Amazon report](replication/results/amazon/comparison.json)
also records its temporal and TweetEval checks.

The 2026-11-21 handover check regenerated the prepared inputs, folds, and
parser samples from the raw archives for all three corpora. They matched
the previous runs byte for byte. Offline model staging verified the
pinned files. SFU scoring and classifier fitting from the verified trees
again produced identical downstream predictions and text-order intervals.

UCI has 3,000 labelled sentences, with 500 per polarity in each of its
three domains. The complete release produced 5,208 EDUs and 12,000
verified trees. Each sentence occupied one scene, and 1,544 sentences
contained one EDU. The verifier independently checked all 3,000 raw
labels and texts and recomputed the metrics for 96,000 held-out
predictions. Full fine and coarse results are in the report.

Use `sfu_run` from the SFU command above. Run the Amazon comparison
from the branch checkout. Run the three-corpus comparison with the
staged UCI source so the comparison uses its grouped bootstrap.

```bash
PYTHONPATH="$PWD" "$assets/classifier-exact-venv/bin/python" -B \
  "$scripts/compare-to-sfu.py" --sfu-run "$sfu_run" --amazon-run "$run" \
  > "$run/comparison.json"
```

```bash
PYTHONPATH="$run/source" "$assets/classifier-exact-venv/bin/python" -B \
  "$scripts/compare-to-sfu.py" --sfu-run "$sfu_run" --uci-run "$run" \
  --uci-archive "$uci_zip" > "$run/comparison.json"
```

## Fine/coarse diagnostics with log-loss and scrambled labels

**Question.** Do fine labels carry sentiment information that their coarse
group loses, beyond simply adding parameters? Exploratory, 2026-11-25, on the
same SFU and Amazon cohorts that were used to develop the models.

1. **Log-loss reanalysis** (`granularity_diagnostics loss`). Macro-F1 changes
   only when a review crosses the 0.5 threshold. Held-out log-loss uses every
   predicted probability. The command compares fine with mapped coarse, and
   labelled with unlabeled models, for relation pooling (refitted; every
   macro-F1 must equal the published value), the relation-gate model and the
   Amazon recursive model (published predictions, checked by hash). Intervals
   are paired bootstrap percentiles over 20,000 draws within fold, category
   and gold label. Holm adjustment is applied within each family and corpus.
2. **Scrambled-label control** (`granularity_diagnostics scramble`). The fine
   gate model is refitted 1,000 times per standard after shuffling fine labels
   among the links of each coarse group across the corpus. Each fine label
   keeps its link count and coarse parent. If real fine labels beat the
   scrambled ones, the distinctions inside coarse groups carry information.
   The command also compares the spread of learned gates among sibling fine
   labels, as a link-weighted variance of log gates, with the scrambled fits.
   The refit with real labels must match the published fine, coarse and
   unlabeled predictions; all 24 matched exactly.

```bash
A=artifacts/sentiment   # verified inputs from the earlier runs
out=$A/granularity-diagnostics-v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run="uv run --project research/sentiment --locked --no-sync python -B -m research.sentiment.granularity_diagnostics"
sfu=(--corpus sfu --prepared $A/sfu-v1/prepared --trees $A/sfu-full-v1
     --scores $A/sfu-edu-scores-v1/scores.jsonl
     --gate-predictions $A/sentiment3-v1.OzbMlX/sfu-cardiff/predictions.jsonl)
amazon=(--corpus amazon --prepared $A/amazon23-short-v1.4YbtTy/prepared
        --trees $A/amazon23-short-v1.4YbtTy/parser-verified
        --scores $A/amazon23-short-v1.4YbtTy/edu-scores/scores.jsonl
        --gate-predictions $A/sentiment3-v1.OzbMlX/amazon-cardiff/predictions.jsonl)
$run loss "${sfu[@]}" --output $out/sfu-loss.json
$run loss "${amazon[@]}" --recursive-predictions $A/amazon-recursive-v1/predictions.jsonl \
  --output $out/amazon-loss.json
$run scramble "${sfu[@]}" --permutations 1000 --output $out/sfu-scramble
$run scramble "${amazon[@]}" --permutations 1000 --output $out/amazon-scramble
python -B -m research.sentiment.granularity_summary $out/*-loss.json $out/*-scramble/report.json
```

The reports are copied to `results/granularity-{loss,scramble}-{sfu,amazon}-v1.json`.
The 1,000 permutation rows per standard stay with the artifacts. Each report
records their SHA-256. On one CPU thread, the SFU scramble took 9 minutes and
the Amazon scramble 22 minutes.

### What the diagnostics found

**Labels help on Amazon.** In the gate model, fine and coarse labels both
lowered Amazon log-loss against unlabeled links by 0.010–0.018 nats. Seven of
the eight contrasts survive Holm adjustment; RST-DT coarse gives Holm p = .085.
On SFU the labelled models also had lower log-loss, but no contrast survives
the adjustment.

**Fine against coarse depends on the model.**

| Model | Amazon fine − coarse log-loss (×10⁻³) | Reading |
|---|---|---|
| Relation pooling, text order and nucleus first | +26 to +41 in all 8; all Holm p ≤ .036 | Fine is reliably *worse* |
| Relation gates | −1.0 to −3.3 in all 4; all intervals include 0 | Fine leans better |
| Recursive composition | −0.9 to +6.0; NLDT +5.3 [+0.8, +9.6], Holm p = .151; others include 0 | No reliable difference |

Pooling gives each fine label six of its own feature columns, and most are
zero in most reviews. Its fine models therefore become overconfident, even
when their macro-F1 barely moves. The gate model needs one extra number per
label, so it can use fine labels without that cost.

**Scrambled labels (gate model, 1,000 permutations).**

| Corpus | Standard | Fine | Coarse | Scrambled fine, mean [95% range] | p fine < scrambled | p sibling gate spread > scrambled |
|---|---|---:|---:|---:|---:|---:|
| Amazon | PCC | .2842 | .2867 | .2889 [.2859, .2917] | .004 | .005 |
| Amazon | NLDT | .2861 | .2880 | .2900 [.2867, .2929] | .010 | .018 |
| Amazon | GUM/eRST | .2864 | .2873 | .2899 [.2865, .2933] | .024 | .036 |
| Amazon | RST-DT | .2871 | .2905 | .2934 [.2896, .2970] | .002 | .001 |
| SFU | PCC | .3886 | .3930 | .3896 [.3682, .4143] | .476 | .202 |
| SFU | NLDT | .3734 | .3741 | .3810 [.3642, .3971] | .180 | .007 |
| SFU | GUM/eRST | .3770 | .3798 | .3843 [.3702, .3993] | .156 | .012 |
| SFU | RST-DT | .3962 | .3932 | .4055 [.3900, .4207] | .112 | .038 |

On Amazon, real fine labels beat scrambled ones in all four standards. All
four permutation p-values remain below .05 after Holm adjustment. Scrambled
fine labels score *worse* than coarse labels, so the extra parameters cost
accuracy. Real fine labels recover that cost and slightly improve on coarse.
Sibling labels receive more distinct weights than scrambled siblings, and the
pattern repeats across standards. For example, *contrast* gets more weight
than *concession* everywhere: NLDT 1.20 against 0.28, PCC 1.01 against 0.40,
RST-DT `contrast-mn` 1.08 against 0.23, and GUM 0.78 against 0.43. *Evaluation*
and *summary* also get more weight than their siblings. On SFU, three of four
standards also learn more distinct sibling weights than chance at
uncorrected p < .05, and two remain below .05 after Holm adjustment. The 400
reviews give too little held-out signal to show better predictions.

**What this does and does not establish.** Fine distinctions inside coarse
groups carry some sentiment-relevant information when a model can use them
cheaply. The direct fine-versus-coarse gain in the gate model is about
0.001–0.003 nats on Amazon, and each interval includes zero. These analyses
are exploratory and reuse the development data. The pre-registered
[confirmation plan](GRANULARITY_CONFIRMATION.md) tests the contrast on a new
16,000-review cohort.
