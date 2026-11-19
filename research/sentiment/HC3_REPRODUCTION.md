# Reproduction checks for the colleague’s experiments

The SFU run in [RUNNING.md](RUNNING.md) checks parser execution on the
sentiment corpus. The checks here use the colleague’s HC3 data and the public
ArgMicrotexts gold annotations.

## HC3 classifiers

The supplied feature artifacts contain the aligned train, validation, and
test matrices for LR and RF MF+WAD. All 24 fine/coarse split matrix hashes
and the other relative input hashes match the supplied run manifests.

I refit the four fine and four coarse models for each classifier from the
saved features. The run used the canonical branch at
`f06309b62cbc53dce0c98668cac225666dfb64d7`, Python 3.11.15, and the saved
versions of NumPy (1.24.4), SciPy (1.10.1), scikit-learn (1.3.2), NetworkX
(3.1), and XGBoost (2.1.1). The saved manifest records Python 3.8.18 and
different source hashes for the two comparison entry points. The available
LR evaluator and paired comparison modules match their recorded hashes.

| Inventory | LR fine/coarse F1 from refit | RF fine/coarse F1 from refit |
|---|---:|---:|
| PCC | .773362 / .716370 | .778485 / .750577 |
| eRST | .777262 / .696894 | .795415 / .774713 |
| RST-DT | .770465 / .738846 | .779738 / .771071 |
| NLDT | .766396 / .702401 | .763098 / .756602 |

RF matched all eight F1 scores, every thresholded test prediction, and
all four primary paired F1 rows, including the intervals and adjusted
p-values. LR matched the chosen C values and changed three test decisions
across the eight fits. All F1 differences from the saved results are below
.001 and the four primary Holm-adjusted p-values agree. The fine-over-coarse
direction holds for all four inventories in both models. This reproduces the
HC3 downstream comparison starting from the saved features. The parsing
checks below start from the original answer text.

Run these commands from the sentiment checkout to reproduce the refits.
The full outputs are in `artifacts/sentiment/hc3-feature-refit/`.

```bash
repo_dir="$PWD"
hc3_dir="$repo_dir/HC3_project/artifacts"
refit_out="$repo_dir/artifacts/sentiment/hc3-feature-refit-rerun"
refit_src="$(mktemp -d /tmp/tos-hc3-source.XXXXXX)"
refit_env="$(mktemp -d /tmp/tos-hc3-env.XXXXXX)"
git archive f06309b62cbc53dce0c98668cac225666dfb64d7 scripts/canonical tos |
  tar -x -C "$refit_src"
uv venv --python 3.11 "$refit_env"
uv pip install --python "$refit_env/bin/python" \
  numpy==1.24.4 scipy==1.10.1 scikit-learn==1.3.2 networkx==3.1 xgboost==2.1.1 tqdm
cd "$refit_src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$refit_env/bin/python" -B -m scripts.canonical.compare_lr_wad_granularity \
  --feature-root "$hc3_dir/feature_matrices" --granularity-root "$hc3_dir/granularity" \
  --mapping-file "$hc3_dir/granularity/pcc_nldt/motif_mapping.csv" \
  --prediction-root "$hc3_dir/model_comparison" --output-dir "$refit_out/lr_mf_wad"
"$refit_env/bin/python" -B -m scripts.canonical.compare_rf_granularity \
  --feature-root "$hc3_dir/feature_matrices" --granularity-root "$hc3_dir/granularity" \
  --mapping-file "$hc3_dir/granularity/pcc_nldt/motif_mapping.csv" \
  --output-dir "$refit_out/rf_mf_wad"
```

## HC3 fresh inference

The canonical alignment script identifies the original HC3 test release with
SHA-256 `43fe588ecd25a7c3c2188e7a96deb19cc5207f0d96b8cd4906142396c99b9608`.
The downloaded file matches. It contains 1,927 unique identity hashes
represented in the 1,932 canonical test rows. The identity key is SHA-256
of the compact UTF-8 JSON array `[text, source, label]`. Only those identity
fields were read from the original release. Its embedded parses belong to
the 2024 DMRST experiment. [Pinned test release](https://huggingface.co/datasets/zaemyung/ToS-Dataset/blob/f94c7f19a7ff99ba6996533910903a63e952ce1d/hc3/hc3_test.discourse_parsed.graph_added.motif_dists.jsonl).

For each authorship label, choose the earliest identity ID in lexicographic
order among the canonical rows with at least four reference EDUs in each of
three strata: 150–500 characters and one scene, 650–1,400 characters and one
scene, or 1,400–3,500 characters and multiple scenes. The zero-based
canonical test indices in run order are 1928, 516, 20, 95, 1809, and 1005.
Use the pilot’s stock PCC segmentation, shared EDU input and identical
four checkpoints. Compare structural fields, integer relation-edge counts
and the selected M3 frequency and depth vectors against the supplied
canonical matrices. The comparison uses NetworkX 3.1 and the supplied M3
hash columns to match induced triads.

All six documents matched the supplied structural fields, integer relation
counts and M3 frequency/depth vectors in all four inventories, for 24
matched document–inventory pairs. M3 vectors were checked at absolute
tolerance `1e-7`. All 32 scene trees and RS3 exports passed the pilot
validator, using the same eight scenes and 170 PCC EDUs. The model weight,
config and table hashes match the SFU run. This checks the historical
feature representation from fresh inference. Exact historical tree replay
would also compare EDU boundaries, constituent spans, attachments,
nuclearity and labels against the colleague's original UniRST trees.
Gold treebank parses and the original scoring setup are needed to
reproduce the paper's parser-accuracy scores.

The local reference trees and run report are in
`artifacts/sentiment/unirst-parser-pilot/hc3-reference-v1/`. The source
run and log are under `/scratch/threads-of-subtlety/parser-pilot/` on the
personal GPU host. The local selection and comparison scripts are in
`artifacts/sentiment/hc3-feature-refit/`. The
[portable results](results/hc3-reproduction.json) contain the selected IDs,
scores and feature comparisons.

### HC3 answer length and scene cuts

The canonical loader splits answers into roughly 492 XLM-R tokens per
scene. It packs paragraphs and uses sentence boundaries inside long
paragraphs. The checked-in command defaults to Punkt and also accepts
SentSplit. The supplied feature matrices establish the scene counts,
while the exact historical sentence-splitter flag is unavailable.
[Canonical loader](https://github.com/ruthenian8/threads-of-subtlety/blob/f06309b62cbc53dce0c98668cac225666dfb64d7/scripts/prepare/prepare_and_parse_datasets_unirst.py)

Count `scene_count > 1` in `X_structural` using `structural_names` in
the supplied feature matrices. The counts match row for row across all
four inventories.

| HC3 split | Answer rows | Answer rows with multiple scenes | Total scenes |
|---|---:|---:|---:|
| Training | 43,322 | 1,632 | 45,417 |
| Validation | 968 | 36 | 1,011 |
| Test | 1,932 | 53 | 1,993 |

For a length comparison with SFU, join the canonical test matrix's
`identity_hash` to the released answer texts using the identity key
above. Tokenize each of the 1,932 rows with the BERT tokenizer pinned
in the [SFU length audit](results/sfu-token-lengths.json), including
special tokens and disabling padding and truncation. HC3 test answers
have a median of 186 BERT tokens, a maximum of 1,871, and 29 rows over
512 tokens. BERT token counts provide a common yardstick. The actual
scene cuts use XLM-R tokens.

## ArgMicrotexts

I counted satellite relations by attachment and multinuclear relations
once per parent and relation in all 112 English gold RST files at commit
`ceb01da17d8c2a82cb713d8f16e290f42eee0eab`. The gold distribution supports
the draft’s observation that `reason` is much more common than `evidence`
or `cause`. Some counts differ under this counting rule.

| Relation | Draft count | Public gold RS3 count |
|---|---:|---:|
| reason | 173 | 181 |
| evidence | 12 | 13 |
| cause | 14 | 20 |
| antithesis | 26 | 32 |
| concession | 65 | 65 |
| contrast | 3 | 3 |

The public corpus supplies all three annotation layers. Reproducing the
colleague’s comparison of gold and predicted RST for argument prediction
requires his version and document subset, EDU-to-argument alignment and
conversion rules, and model/evaluation configuration or saved predictions.
The local draft includes neither scores nor a complete protocol for that
comparison. [Public gold RS3 files](https://github.com/peldszus/arg-microtexts-multilayer/tree/ceb01da17d8c2a82cb713d8f16e290f42eee0eab/corpus/rst),
[draft argumentation section](../../RST_relation_granularity/latex/argumentative.tex).

## Label coverage

The checkpoint table lists every possible decoder output as a relation and
nuclearity pair. The coarse maps operate on just the relation name. The
HC3 relation matrices used the vocabulary observed in the training data
plus an `__OTHER__` column. The supplied canonical frequencies show zero
edge counts in that column on train, validation and test for both NLDT and
RST-DT. The missing native names therefore did not contribute edges to the
supplied HC3 comparison.

The released NLDT table contains both `Volitional-cause_NS` and
`Volitional-causal_NS`/`_SN`. Only `volitional-causal` was mapped to
`Cause` in the research branch. RST-DT has 110 decoder classes representing
69 native names, of which 60 appeared in the canonical HC3 matrices. The
nine absent names are `analogy-mn`, `comment-topic-mn`, `conclusion`,
`elaboration-process-step`, `interpretation-mn`, `otherwise`,
`proportion-mn`, `reason-mn`, and `topic-comment-mn`. These are coverage
gaps for other corpora. The historical `__OTHER__` feature column does not
specify their linguistic coarse classes. Preserve the original maps and
confirm a policy for additional emitted labels before the full sentiment
comparison.
