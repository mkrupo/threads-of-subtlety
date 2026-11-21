# Reproduce the sentiment controls and parser pilot

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


## Amazon Reviews '23 and UCI Sentiment Labelled Sentences

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

### Runtime and raw inputs

Use the parser and scorer environments and pinned model files from the
earlier sections. Place or link them in an assets directory with the layout
below. Use the locked sentiment project for the classifier environment
and install NetworkX 3.1 in it for motifs. Run from the repository root
with the canonical relation-map revision available to Git.

```text
assets/
  parser-venv/
  scorer-venv/
  classifier-exact-venv/
  hf/hub/
  stock/
  cardiff/
```

Create a fresh run directory beneath this checkout, such as
artifacts/sentiment/amazon23-short-v1 or artifacts/sentiment/uci-sls-v1.
Keep the released raw files and full per-review outputs in that ignored
directory.

For Amazon, download the four raw category archives, lid.176.ftz, and
the TweetEval sentiment training file. Save the TweetEval Git revision
in tweeteval-revision.txt. Use the named files and revision in the
[Amazon cohort manifest](replication/results/amazon/cohort.json) and an
isolated Python 3.10 preparation environment with fasttext-wheel 0.9.2,
NumPy 1.26.4, and Tokenizers 0.20.3.
[Amazon source archives](https://amazon-reviews-2023.github.io/) ·
[FastText language identification](https://fasttext.cc/docs/en/language-identification.html) ·
[TweetEval training file](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment)

For UCI, use the official raw archive and the same TweetEval training
file. UCI uses the locked classifier environment for preparation.
[UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences)

### Prepare and run

Set run and assets to absolute paths, with scripts pointing to the
replication directory in this checkout. Use the preparation command for
the chosen corpus.

```bash
scripts="$PWD/research/sentiment/replication"
bash "$scripts/stage.sh" "$run" amazon
(
  cd "$run/source"
  "$prep_python" -B -m research.sentiment.prepare_amazon \
    --run-dir "$run" \
    --tokenizer "$assets/hf/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/tokenizer.json"
)
```

```bash
bash "$scripts/stage.sh" "$run" uci
(
  cd "$run/source"
  "$assets/classifier-exact-venv/bin/python" -B -m research.sentiment.prepare_uci \
    --run-dir "$run" --archive "$uci_zip" --tweeteval-train "$tweeteval_train"
)
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

The parser runner defaults to GPU 0 with two concurrent processes once
the shared PCC segmentation has been written. Set CUDA_DEVICE=-1 with
the CPU Torch environment for sequential CPU inference. Set
UNIRST_CHECKPOINTS when the two fine checkpoints live outside
HC3_project in the run's checkout.
