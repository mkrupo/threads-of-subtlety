# Fine versus coarse RST labels for review sentiment: confirmation plan

**Outcome (2026-11-26):** fine labels beat mapped coarse labels in all four
standards. Holm p ≤ .014, and log-loss is lower by 0.0014–0.0040 nats.
The advantage grows with training data. [Results](#results-added-2026-11-26-the-plan-above-is-unchanged)

Status: **frozen on 2026-11-25** by the commit that added this line, tagged
`granularity-confirmation-plan-v1`. No sentiment model had been fitted on the
confirmation cohort. Its reviews had been selected and its parse was running.
Later changes go in the deviation log at the end, with the date and reason.

Cohort records: [selection provenance](results/amazon-confirm-provenance-v1.json) ·
[independent verification](results/amazon-confirm-cohort-verification-v1.json).
The parser `sample.json` has SHA-256 `a1c27bcb…`. The staged parser code hash,
computed over all staged `.py` files, is `7eb3dd6c…`. That is the Amazon
exploration parser code plus `prepare_amazon_confirm.py`.

## Question

Inside a coarse relation group, do the fine RST labels carry information about
review sentiment that the coarse label loses? Example: NLDT *contrast* and
*concession* both map to the coarse group *Contrast*.

All comparisons use **mapped coarse labels**: the same predicted trees with
fine labels renamed through the canonical maps. They therefore test label
granularity, not tree structure or a different parser.

## Why a confirmation cohort

The SFU (400 reviews) and Amazon (2,000 reviews) results were used to develop
the models, so they are exploration data. Their fine-versus-coarse macro-F1
gaps are mostly within ±1 point, and their direction changes with corpus,
standard and model. See [what the exploration found](#exploration-evidence).
A new, larger and disjoint sample can test one pre-specified contrast with
enough precision to detect or rule out a gap of the size seen so far.

## Confirmation cohort

- Source and rules: identical to the exploration Amazon cohort
  ([prepare_amazon.py](replication/prepare_amazon.py)). Amazon Reviews '23,
  categories All_Beauty, Appliances, Musical_Instruments and Software; written
  2022-01-01 or later; stars 1–2 negative, 4–5 positive; 40–160 XLM-R content
  tokens; fastText English p ≥ 0.80; no normalized match to the TweetEval
  sentiment training texts; at most one review per normalized text, reviewer
  and product. Candidates are visited in the same seeded hash order.
- Disjointness: every normalized text, reviewer and product of the exploration
  cohort is excluded. [prepare_amazon_confirm.py](replication/prepare_amazon_confirm.py)
- Size: 2,000 reviews per category and polarity, 16,000 in total. A quota of
  2,500 could not be filled for positive Software reviews (2,122 eligible), so
  the largest multiple of 250 that fills every stratum was used. Selection uses
  ratings and text filters only, never model outputs.
- Folds: five outer folds and four inner folds per outer fold, stratified by
  category and polarity, seed 42 (`corpus.build_splits`).
- Parsing and scoring: the staged parser code, checkpoints and environment of
  the exploration Amazon run (source hash recorded), and the frozen CardiffNLP
  scorer. The rebuilt scorer reproduced the exploration Amazon scores byte for
  byte before use.

## Primary analysis (fixed)

- **Model.** The relation-gate model (Experiment 3,
  [relation_gate.py](relation_gate.py)) with CardiffNLP EDU probabilities,
  nuclearity base features, one learned gate per label and fixed L2 with
  `C = 1`. No refitting choices are made on this cohort.
- **Contrast.** Fine minus mapped coarse, for each of the four standards
  (PCC, NLDT, GUM/eRST, RST-DT).
- **Outcome.** Pooled held-out binary log-loss over the five outer folds.
  Log-loss responds to every change in predicted probability, whereas macro-F1
  only changes when a review crosses 0.5.
- **Inference.** Paired percentile bootstrap with 20,000 draws, resampling
  reviews within outer fold × category × gold label. Two-sided, α = 0.05,
  Holm-adjusted across the four standards.
- **Smallest effect of interest.** 0.002 nats of log-loss, roughly the
  largest fine-versus-coarse gap in the Amazon exploration data. Each verdict
  uses the 95% interval:
  - *fine better / coarse better*: Holm-adjusted p < 0.05. If the 95%
    interval also lies inside (−0.002, +0.002), the verdict becomes *fine/coarse
    better, but within ±0.002*. The difference is then reliable but smaller than
    the margin.
  - *equivalent within ±0.002*: not significant, and the 95% interval lies inside
    (−0.002, +0.002);
  - otherwise *inconclusive*.
- Macro-F1 differences are reported alongside, but are not the primary outcome.
- On 2026-11-25, Max chose the 0.002 margin and a gate-only primary test.
  Relation pooling is secondary. This choice was made after the
  exploration results and before any confirmation-cohort model fit.

## Secondary analyses (fixed; Holm within each family)

1. **Labels versus unlabeled links.** Fine minus unlabeled and coarse minus
   unlabeled gate models, log-loss.
2. **Scrambled-label control.** 200 refits of the fine gate model after
   shuffling fine labels among the links of each coarse group (link counts and
   coarse parents kept). Report the permutation p-values for (a) real fine
   log-loss below the scrambled ones and (b) the link-weighted spread of log
   gates among sibling fine labels above the scrambled ones. Holm-adjust each
   test across the four standards. Run `scramble` on the same host,
   environment and thread settings as `primary`, because its real-label refit
   must reproduce the primary predictions within 1e-9.
3. **Learning curve.** Within each outer training fold, draw nested subsets
   of 1,000, 2,000, 4,000 and 8,000 reviews, stratified by category and
   polarity, five seeded draws per size, plus the full training fold
   (12,800). Fit unlabeled, coarse and fine gate models and score the full
   test fold. Report fine minus coarse log-loss at each size, averaged over the
   draws. If sparsity limits fine labels, the gap should become more favourable
   to fine as training size grows. Test this with the gap at 1,000 minus the gap
   at full size, using a bootstrap interval and Holm adjustment across the four
   standards. The interval resamples test reviews with the training draws
   fixed. It omits variation between draws and is therefore somewhat narrow at
   small sizes.
4. **Other models.** Fine minus coarse log-loss for text-order relation pooling
   (Experiment 2, nested choice of C) and for recursive composition. In the
   exploration data, fine pooling had reliably *worse* log-loss than coarse on
   Amazon in all eight conditions, which suggests overfitting of the sparse
   fine columns.

## Exploration evidence

These results come from the 2,000-review Amazon cohort and the 400 SFU
reviews, which were also used to build the models. Commits `d634183` and
`4dc67fc` record them. [Diagnostics](RUNNING.md#finecoarse-diagnostics-with-log-loss-and-scrambled-labels) ·
[Split ablation](RUNNING.md#split-tree-ablation-on-amazon)

- **Gate model, fine minus coarse log-loss, Amazon:** −2.50, −1.90, −0.97 and
  −3.31 ×10⁻³ for PCC, NLDT, GUM/eRST and RST-DT. Every 95% interval
  includes zero, with half-widths of about 0.004–0.006.
  The expected interval half-width with 16,000 reviews is about
  0.0015–0.002. The design can therefore detect gaps of the explored size,
  and it can show equivalence if the true gap is near zero.
- **Scrambled labels, Amazon:** real fine labels beat 1,000 within-group
  shuffles in all four standards (permutation p = .002–.024). Sibling fine
  labels received more distinct gates than shuffled ones (p = .001–.036).
  SFU showed the gate pattern in three standards, but not the predictive gain.
- **Labels against unlabeled links, Amazon gate model:** 0.010–0.018 nats
  lower log-loss. Seven of eight contrasts survive Holm adjustment.
- **Relation pooling, Amazon:** fine had reliably worse log-loss than coarse
  in all eight conditions, by +0.026 to +0.041 nats. Macro-F1 differed by at
  most 1.3 points.
- **Recursive composition, Amazon:** fine and coarse differed by at most
  0.006 nats, and no difference survived Holm adjustment.
- **Learning curve:** not run on the exploration data.

## Not part of this plan

Split-tree effects (the [split ablation](split_ablation.py)) are measured on
the exploration cohort and reported separately. Star-level (1–5) prediction,
mixed-sentiment subsets and SFU repair are possible follow-ups.

## Execution

Commands, in order, after the parse and scores are verified. Use one host and
one locked environment with `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.

```bash
cohort=(--corpus cohort --prepared $RUN/prepared --trees $RUN/parser-verified
        --scores $RUN/edu-scores/scores.jsonl --audit $RUN/parser-audit.json
        --scores-manifest $RUN/edu-scores/manifest.json --sample $RUN/sample.json)
python -B -m research.sentiment.confirm_granularity primary "${cohort[@]}" --output $RUN/confirm/primary
python -B -m research.sentiment.granularity_diagnostics scramble "${cohort[@]}" --permutations 200 \
  --reference-report $RUN/confirm/primary/report.json \
  --reference-predictions $RUN/confirm/primary/predictions.jsonl --output $RUN/confirm/scramble
python -B -m research.sentiment.confirm_granularity curve "${cohort[@]}" --output $RUN/confirm/curve
python -B -m research.sentiment.confirm_granularity others "${cohort[@]}" \
  --baseline-config research/sentiment/replication/results/amazon/baseline-config.json \
  --output $RUN/confirm/others
```

A fresh agent reviewed the analysis code against this plan before the freeze.
It found no leakage, alignment or sign errors. Its points led to the combined
verdict, the Holm adjustments for the scramble and curve families, and extra
checks on the scorer, configuration and selection inputs.

## Results (added 2026-11-26; the plan above is unchanged)

On 2026-11-25 the GTX 1080 Ti host ran the four commands with code archived
from the frozen tag. The logged source hashes match the tag, and the analyses
finished at 23:58 UTC. The parse covered all 16,000 reviews, with one scene
and four verified trees per review. The scramble's real-label refits
reproduced the primary predictions exactly.
Reports: [primary](results/amazon-confirm-primary-report-v1.json) ·
[scramble](results/amazon-confirm-scramble-report-v1.json) ·
[curve](results/amazon-confirm-curve-report-v1.json) ·
[other models](results/amazon-confirm-others-report-v1.json).

### Primary: fine minus mapped coarse, relation-gate model

| Standard | Fine log-loss | Coarse log-loss | Δ (×10⁻³) | 95% interval | Holm p | Δ macro-F1 (points) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| PCC | .2850 | .2886 | −3.61 | [−5.07, −2.15] | < .001 | +0.24 | fine better |
| NLDT | .2873 | .2887 | −1.37 | [−2.48, −0.25] | .014 | +0.09 | fine better |
| GUM/eRST | .2849 | .2880 | −3.11 | [−4.54, −1.68] | < .001 | +0.19 | fine better |
| RST-DT | .2943 | .2983 | −3.97 | [−5.53, −2.40] | < .001 | +0.27 | fine better |

**Fine labels beat mapped coarse labels in all four standards.** For PCC and
RST-DT, the whole interval lies beyond the 0.002 margin. For GUM/eRST and NLDT,
the interval reaches inside the margin. The gain shows mainly in calibrated
probabilities: macro-F1 rises by 0.1–0.3 points, and fine gate models reach
.880–.887.

### Secondary analyses

1. **Labels against unlabeled links:** coarse lowers log-loss by 0.013–0.016
   nats and fine by 0.017–0.019. All eight contrasts have Holm p < .001.
   Macro-F1 rises by 0.8–1.3 points.
2. **Scrambled labels, 200 permutations:** in every standard, real fine labels
   beat all 200 within-group shuffles. Each permutation p is .005, the
   smallest possible, and the Holm-adjusted p is .020. Shuffled fine labels score
   almost exactly like coarse labels. For PCC, for example, real fine scores
   .2850, scrambled .2890 [.2886, .2894] and coarse .2886. The sibling gate
   spread is 3–14 times the scrambled mean, with Holm p = .020 in every
   standard. Several sibling contrasts repeat the exploration pattern. *Contrast*
   receives 1.05–1.20 against *concession* 0.32–0.48 in all four standards. GUM
   *causal-result* receives 1.36 against *causal-cause* 0.50, and PCC *result*
   receives 1.02 against *cause* 0.52.
3. **Learning curve:** fine minus coarse log-loss ×10⁻³ by training size,
   averaged over five nested draws per size.

   | Standard | 1,000 | 2,000 | 4,000 | 8,000 | 12,800 | Gap(1,000) − gap(12,800), Holm p |
   |---|---:|---:|---:|---:|---:|---|
   | PCC | −0.51 | −2.21 | −3.05 | −3.30 | −3.61 | +3.10 [+2.34, +3.85], < .001 |
   | NLDT | +1.60 | +0.21 | −0.87 | −1.16 | −1.37 | +2.97 [+2.27, +3.67], < .001 |
   | GUM/eRST | +0.02 | −1.79 | −2.44 | −2.81 | −3.11 | +3.13 [+2.32, +3.94], < .001 |
   | RST-DT | +0.28 | −1.81 | −2.94 | −3.76 | −3.97 | +4.25 [+3.40, +5.09], < .001 |

   With 1,000 training reviews, fine and coarse tie, and NLDT is worse with fine
   labels. The fine advantage then grows steadily with data, as the sparsity
   hypothesis predicts. This explains why the exploration cohort, with 1,600
   training reviews per fold, could not separate fine from coarse, and why
   SFU, with 320, could not either.
4. **Other models, fine minus coarse log-loss:**
   - **Text-order relation pooling:** fine was better for PCC (−4.5 ×10⁻³) and
     GUM/eRST (−5.3), both with Holm p < .001. NLDT showed no reliable
     difference (−1.5, Holm p = .30). RST-DT fine was worse (+7.6, Holm
     p = .002); its 69 fine labels give 420 feature columns.
   - **Recursive composition:** fine was better for PCC (−1.7, Holm p = .033).
     The other three standards showed no reliable difference.
   - **Comparison with exploration:** in the 2,000-review exploration
     cohort, fine pooling was worse in all eight conditions. With 8× the
     data, fine pooling helps in two of four standards.

### Reading

Within a coarse RST group, fine relation labels carry sentiment-relevant
information. On 16,000 held-out reviews, a model that spends one weight per
label turns that information into better review-level predictions in all
four standards. The gain is reliable, but small in accuracy terms. It
needs a few thousand training reviews, which the earlier and smaller
samples lacked. Models that spend many parameters per label need more
data still, and the 69-label RST-DT inventory remains too sparse for
relation pooling.

## Deviation log

- 2026-11-25: Quota reduced from the intended 2,500 to 2,000 per stratum
  before any parsing, because positive Software reviews ran out at 2,122.
- 2026-11-26: No analysis deviations. The analysis launcher was started
  over an SSH command that did not detach. The job ran in its own session,
  PID 1072242, recorded in `analysis-launch/`. It ran to completion and was
  unaffected.
