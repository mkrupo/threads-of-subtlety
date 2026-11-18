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
