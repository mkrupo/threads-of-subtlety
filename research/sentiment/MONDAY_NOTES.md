# Monday meeting notes — RST and review sentiment

## What I tested

- I used sentiment as a downstream task. I compared four RST standards (PCC, NLDT, GUM/eRST, RST-DT) and fine versus coarse relation labels within each one.
- SFU has 400 English product reviews, evenly split between recommend and do not recommend. Long reviews are split into shorter chunks. All four parsers get the same chunks and the same clause-sized units (EDUs).
- I used the released UniRST checkpoints for PCC and NLDT and my colleague’s fine-tuned checkpoints for GUM/eRST and RST-DT. Within each standard, fine keeps the predicted relation names and coarse merges them using my colleague’s mapping. The two versions share the same tree.

## What the classifier sees

- Frozen CardiffNLP gives each EDU three probabilities in order `[negative, neutral, positive]`. They sum to one. We reuse the same EDU scores across conditions. Logistic regression uses the following numbers to predict whether the reviewer recommends the product.

| Condition | Input for one review |
|---|---|
| Equal weight | **3 numbers.** Average the probabilities over all EDUs. This baseline uses no tree features. |
| Nuclearity | **6 numbers.** Average the central EDUs, then the supporting EDUs. |
| Fine relations | **6 nuclearity numbers plus 6 per fine label.** Each label gets three averages for the left side and three for the right. |
| Coarse relations | **6 nuclearity numbers plus 6 per coarse label.** Fine labels that map to the same coarse label share a block. |

- **Central** means every branch from the scene root to the EDU is a nucleus branch. A central path can split into two nuclei. An EDU below any satellite branch is supporting. We collect individual EDUs from all scenes before averaging the two groups.
- **Relation blocks** use left and right in text order. At a node, average the EDU scores on each side. Then average those six numbers across all occurrences of the label in the review. A missing label gets six zeros. Logistic regression learns a weight for each column. If a nucleus switches sides, both orders still feed the same left and right columns.
- **Input size** is `6 + 6 × number of labels`. Fine → coarse label counts are PCC 25 → 14, NLDT 27 → 14, GUM/eRST 32 → 15, and RST-DT 69 → 18. For example, RST-DT goes from 420 to 114 inputs. The trees and the first six numbers stay the same within each standard.

**Toy review.** “The screen looks great, but the battery dies by noon.” Suppose the parser gives us a concession with the battery complaint as the nucleus. Illustrative sentiment scores are screen `[0.10, 0.10, 0.80]` and battery `[0.80, 0.10, 0.10]`.

- **Equal weight** gives `[0.45, 0.10, 0.45]`.
- **Nuclearity** gives `[0.80, 0.10, 0.10 | 0.10, 0.10, 0.80]` (battery first as the nucleus).
- **Fine** starts with the nuclearity numbers and puts `[0.10, 0.10, 0.80 | 0.80, 0.10, 0.10]` into the concession block (screen first in the text). GUM/eRST calls this block `adversative-concession`.
- **Coarse** starts with the same nuclearity numbers. In GUM/eRST, the `adversative` block averages concession and contrast occurrences along with any other occurrences mapped to `adversative`. In this toy review it contains the concession’s six numbers.

## How we evaluated

- **Five-fold cross-validation.** All conditions use the same splits, balanced by product category and recommendation. Each round fits on 320 reviews and tests on 80. Each review is tested once. Inside the 320 training reviews, four smaller rounds pick how strongly to penalize large logistic regression weights.
- **Macro-F1.** Pool the 400 held-out predictions. Compute F1 separately for recommend and do not recommend, then average the two scores.
- **Uncertainty.** Draw 20,000 lists of 400 reviews from the original reviews, allowing repeats. Keep the same mix of test folds, product categories, and true labels. Use both models’ saved predictions on each list to recompute the fine-minus-coarse gap. The middle 95% of gaps gives the interval for that comparison.

## Results

| Standard | Nuclearity | Fine relations | Coarse relations |
|---|---:|---:|---:|
| PCC | 0.8225 | 0.7900 | 0.7875 |
| NLDT | 0.8324 | 0.7975 | 0.8000 |
| GUM/eRST | 0.8250 | 0.7675 | 0.7800 |
| RST-DT | 0.8224 | 0.7700 | 0.8000 |

- Held-out macro-F1 for the shared controls was **0.817** for equal-weight EDU pooling, **0.827** for word-based TF–IDF, and **0.617** for review length.
- Nuclearity scored slightly higher than equal weight for all four parsers. Adding relation features lowered the scores relative to nuclearity. Coarse scored higher in three of four standards and every fine-minus-coarse interval included zero. The direction of the granularity effect remains uncertain on these reviews.

## Names to remember

- **Data.** [SFU Review Corpus](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html)
- **Idea.** Use discourse structure to combine local sentiment. [Bhatia, Ji, and Eisenstein (2015)](https://aclanthology.org/D15-1263/)
- **EDU scorer.** [CardiffNLP Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Parser family.** [UniRST](https://arxiv.org/abs/2510.06427)
