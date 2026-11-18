# Reproduce the CPU sentiment controls

The first run was completed on 2026-11-18. These are text/length/prior controls,
not an RST comparison. No colleague parser weights are needed for this stage.

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
