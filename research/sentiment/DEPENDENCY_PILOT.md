# Can RST relation labels help weight sentiment evidence?

The earlier SFU and Amazon experiment averaged sentiment separately for
each relation label. A fine label with few occurrences therefore has its own
set of sparsely observed feature columns. This pilot gives every RST link
the same scoring function and lets its label change how much that score
contributes to the review. The fine and coarse conditions use the same links.

## What we borrowed

Chistova (2023) encodes each discourse unit with mDeBERTa and scores pairs
of units to predict argument links such as support and attack. RST relations
modify those pair scores by multiplication after encoding. The experiment
also tests whether the presence of an RST link is enough, using unlabeled
adjacency. [Paper, §3.2](https://aclanthology.org/2023.findings-acl.209.pdf#page=3)

Here the target is one positive or negative label per review. For example,
if “the camera works well” is the nucleus of a concession and “it is heavy”
is its satellite, the same two EDU vectors enter the pair score in the fine,
coarse, and unlabeled runs. The trained classifier decides the weight of
that pair from its fine label or from the coarse group it belongs to. In the
unlabeled run every link receives equal weight.

## What each classifier sees

- **EDU mean** averages the frozen vectors for the entire review.
- **Nuclearity** takes separate means for EDUs on all-nucleus paths and for
  supporting EDUs across the review's scenes. If there are no supporting
  EDUs, their mean is set to the central mean.
- **Unlabeled links** adds the mean score of pairs of linked EDUs to the
  nuclearity features. A single bilinear function scores each pair by
  learning a coefficient for every product of a dependent-vector coordinate
  and a head-vector coordinate.
- **Fine and coarse links** fit the same pair scoring function and a positive
  weight for each fine label or coarse group, respectively. The extra score
  is the weighted sum of pair scores divided by the sum of link weights.

Each binary tree split adds one link from the dependent head to the nucleus
head. At a two-nucleus split the left head becomes the parent. Single-EDU
scenes add zero links, and links from all scenes in a review contribute to
the same mean. A review with no links has a zero pair feature vector before
scaling. Fine and coarse use the same trained-parser trees with the recorded
research mappings. The four inventories can produce different trees.

The two arms start with either three frozen CardiffNLP probabilities or the
frozen mDeBERTa-base CLS vector per EDU. For this pilot we retain eight PCA
components of the mDeBERTa vectors. Both arms train with binary log loss and
fixed L2 regularization (`C=1`). Each gate starts at the same value. The
pair score, gates, and coefficients for the nuclearity features are fitted
jointly within a training fold. PCA and numeric scaling use only training
reviews. Each review contributes one held-out prediction in the existing
five folds with a 0.5 threshold. We report pooled macro-F1 and paired 95%
pointwise percentile intervals from 20,000 resamples within fold, category,
and gold label. Resampling keeps the fits fixed and reports a separate
interval for each contrast.

## What we found

All scores are pooled held-out macro-F1. CardiffNLP's EDU mean scored
0.8200 on SFU (400 reviews) and 0.8629 on Amazon (2,000 reviews).

| Corpus | Standard | Nuclearity | Unlabeled | Fine | Coarse |
|---|---|---:|---:|---:|---:|
| SFU | PCC | .8250 | .8200 | .8399 | .8374 |
| SFU | NLDT | .8300 | .8275 | .8349 | .8450 |
| SFU | GUM/eRST | .8250 | .8300 | .8325 | .8450 |
| SFU | RST-DT | .8250 | .8150 | .8349 | .8300 |
| Amazon | PCC | .8655 | .8710 | .8895 | .8845 |
| Amazon | NLDT | .8640 | .8690 | .8925 | .8845 |
| Amazon | GUM/eRST | .8695 | .8760 | .8890 | .8855 |
| Amazon | RST-DT | .8680 | .8759 | .8770 | .8775 |

With CardiffNLP on Amazon, both fine and coarse weights beat the unlabeled
control for PCC, NLDT, and GUM/eRST, with pointwise intervals above zero.
On SFU, all label-versus-unlabeled intervals include zero. The largest
Amazon fine advantage is for NLDT (+0.80 points, interval +0.25 to +1.40).
SFU GUM/eRST favors coarse by 1.25 points (interval +0.25 to +2.50).
These are individual comparisons. The direction of the fine/coarse
difference depends on the corpus and standard under this feature design.

In the frozen mDeBERTa arm, EDU mean scores 0.5697 on SFU and 0.6095 on
Amazon. Those means exceed every tree condition on Amazon and all but one
on SFU. The result speaks to these frozen, compressed
EDU vectors and this small classifier. For a workshop paper, the CardiffNLP
arm gives the clearer test of whether relation weights help when the local
features already carry sentiment information.

The [aggregate report](results/sentiment3-dependency-v1.json) has all
17 conditions per corpus and encoder, with hashes and paired intervals.
An independent check recomputed macro-F1 and accuracy from the saved
predictions and verified their gold labels and held-out folds.

## Reproduce

Use the [verified SFU artifacts and Amazon replay](RUNNING.md) for the
prepared sets, frozen CardiffNLP scores, and four tree files per corpus.
The evaluator checks the recorded hashes before fitting. From the repo
root, set `corpus` to `sfu` or `amazon` and point `prepared`, `trees` and
`cardiff` to the replay artifacts. `trees` contains `scenes.jsonl` and the
four inventory JSONL files. `cardiff` points to `scores.jsonl`.

To cache the mDeBERTa features, use a separate Python 3.10 environment with
PyTorch 2.2.2+cu118, Transformers 4.46.3, Hugging Face Hub 0.26.5,
SentencePiece 0.2.3 and NumPy 1.26.4. Use the SFU audit at
`research/sentiment/results/sfu-full-parser-audit.json` or the Amazon audit
at `research/sentiment/replication/results/amazon/parser-audit.json`. The
encoder pins the [Microsoft checkpoint revision](https://huggingface.co/microsoft/mdeberta-v3-base/tree/a0484667b22365f84929a935b5e50a51f71f159d)
and stores FP32 CLS vectors in scene and EDU order. The run used batch size
64 on one GTX 1080 Ti per corpus. Both encoder jobs finished in under two
minutes after switching to the SentencePiece tokenizer to resolve a
protobuf version mismatch in the existing parser environment. An active
sample showed about 1.5 GiB in use per card. The longest EDU was 103 tokens
on SFU and 55 on Amazon, including special tokens.

```bash
encoder_python=/path/to/encoder/venv/bin/python
output=artifacts/sentiment/dependency-pilot

CUDA_VISIBLE_DEVICES=0 "$encoder_python" -B -m research.sentiment.encode_mdeberta \
  --scenes "$trees/scenes.jsonl" --audit "$audit" \
  --device cuda:0 --batch-size 64 --output "$output/$corpus-embeddings"

uv sync --project research/sentiment --locked
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for arm in cardiff mdeberta; do
  representation="$cardiff"
  if [ "$arm" = mdeberta ]; then representation="$output/$corpus-embeddings"; fi
  uv run --project research/sentiment --locked --no-sync python -B \
    -m research.sentiment.dependency_sentiment \
    --corpus "$corpus" --arm "$arm" --prepared "$prepared" --trees "$trees" \
    --representation "$representation" --output "$output/$corpus-$arm"
done
```

Each fresh run records fold predictions, input checksums, runtime versions,
metrics, paired differences, and fitted weights. The linked comparisons
are exploratory and use the same corpora as the earlier sentiment runs.
