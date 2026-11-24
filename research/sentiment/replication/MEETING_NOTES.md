# Meeting notes on RST granularity in review sentiment

## What we tested

- **Our starting point was HC3.** The earlier work found that keeping detailed RST relation labels helped detect machine-written text when the classifier used small patterns from discourse trees. We extended that setup to predicting whether a review is positive or negative.
- **We ran the same sentiment experiment on two corpora.** SFU has 400 reviews from eight categories, labeled by whether the reviewer recommends the product. Our Amazon sample has 2,000 short English reviews from four categories. We used one- and two-star reviews as negative and four- and five-star reviews as positive. The model saw the review body. The star rating supplied the target label.
- **Amazon gives us short reviews with room for discourse.** The median review is 48 words and has eight clause-sized units, called **EDUs**. Every Amazon review fits in one scene. SFU reviews have a median of 573 words, and 271 had to be split into scenes before parsing.
- **We compared four RST inventories.** PCC, NLDT, GUM/eRST, and RST-DT all received the same EDU boundaries. Each inventory predicted its own trees. Within each inventory, the fine and coarse conditions used the same trees. Fine keeps labels such as *concession* and *contrast*. Coarse can group them under a broader label such as *adversative*.

## What each model saw

- **The sentiment scorer saw one EDU at a time.** Frozen CardiffNLP RoBERTa gave each EDU three probabilities, one each for negative, neutral, and positive. We reused those scores across the discourse conditions.
- **Equal weight saw three numbers per review.** Average each of the three probabilities across all EDUs. Every EDU counts equally.
- **Nuclearity saw six numbers.** Average the probabilities for the main material and the supporting material separately. The tree determines which EDUs belong in each group.
- **Relation pooling saw the same six nuclearity numbers plus six per relation label.** For each occurrence of a relation, average the sentiment on the left side and on the right side. Then average those values across occurrences of that label in the review. An absent label gets six zeros. The coarse version pools occurrences whose fine labels map to the same broader label.
- **Nucleus-first pooling changed the order of those inputs.** For a main–supporting pair, place the main side first, even when it appears second in the text. Pairs with two main sides keep their text order.
- **The motif-only model saw tree patterns.** It got the frequency and occurrence-weighted average depth of small three-node patterns, using the same feature idea as HC3. The text controls used word and word-pair TF–IDF, or just the log-transformed word count.

In “The screen looks great, but the battery dies by noon,” equal weighting averages praise and complaint. If the tree marks the battery complaint as the main point, nuclearity gives the classifier separate scores for the complaint and the praise. Text-order relation pooling also records “positive on the left, negative on the right” under *concession*. Coarse pooling puts those scores under *adversative*.

## How we evaluated it

- Within each corpus, all conditions used the same five test folds, balanced by category and polarity. Each review was tested once. Four inner folds chose the logistic regression regularization.
- We pooled the held-out predictions and reported **macro-F1**, the mean of positive-class F1 and negative-class F1. For each comparison, we resampled the same held-out reviews for both models 20,000 times to estimate a paired 95% interval.

## What we found

| Inventory | SFU nuclearity | SFU + fine / coarse relations | Amazon nuclearity | Amazon + fine / coarse relations |
|---|---:|---:|---:|---:|
| PCC | .8225 | .7900 / .7875 | .8655 | .8745 / .8795 |
| NLDT | .8324 | .7975 / .8000 | .8640 | .8710 / **.8840** |
| GUM/eRST | .8250 | .7675 / .7800 | .8690 | .8705 / .8775 |
| RST-DT | .8224 | .7700 / .8000 | .8670 | .8755 / .8780 |

- **Adding relation features reversed direction.** All eight text-order conditions scored below their nuclearity controls on SFU. All eight scored above them on Amazon. Coarse NLDT went from **3.25 points below** nuclearity on SFU to **2.00 points above** it on Amazon. The paired 95% interval for the Amazon gain was **+0.85 to +3.15 points**.
- **Coarse labels worked well for sentiment pooling.** Coarse scored higher in three of four inventories on SFU and all four on Amazon. The clearest Amazon gap was NLDT, where coarse beat fine by **1.30 points** with a paired interval of **+0.40 to +2.20**.
- **The way we use the tree matters too.** With motif features alone, fine scored higher in three of four inventories on both corpora. All paired intervals for the change from text order to nucleus-first pooling included zero.
- **The simple controls put the gains in context.** Equal-weight EDU pooling scored **.8174 on SFU** and **.8629 on Amazon**. TF–IDF scored **.8274** and **.8330**, respectively. Length alone scored **.6166** on SFU and **.5013** on Amazon.

**Paper takeaway.** The usefulness of finer RST labels depends on the corpus and on what the classifier reads from the tree. Amazon shows that adding relation features can help on this sample of short, intact reviews. Review length, sample size, product categories, and label definitions changed together, so the comparison leaves the cause of the reversal open. [Full SFU–Amazon comparison](results/amazon/comparison.json)

## Experiment 3 — Weighting connections in the tree

| Corpus | Standard | Unlabeled links | Fine | Mapped coarse |
|---|---|---:|---:|---:|
| SFU | PCC | .8200 | **.8399** | .8374 |
| SFU | NLDT | .8275 | .8349 | **.8450** |
| SFU | GUM/eRST | .8300 | .8325 | **.8450** |
| SFU | RST-DT | .8150 | **.8349** | .8300 |
| Amazon | PCC | .8710 | **.8895** | .8845 |
| Amazon | NLDT | .8690 | **.8925** | .8845 |
| Amazon | GUM/eRST | .8760 | **.8890** | .8855 |
| Amazon | RST-DT | .8759 | .8770 | **.8775** |

Held-out macro-F1 from the same five folds. Bold marks the higher labeled
score in each row.

- **What changed.** We kept the same trees, CardiffNLP EDU scores, and nuclearity averages. The model also scores each pair of EDUs connected by the tree. Unlabeled links all get equal weight. Fine and coarse learn a weight for each label. [Inspiration from Chistova, §3.2](https://aclanthology.org/2023.findings-acl.209.pdf#page=3)
- **Do labels help?** On Amazon, both fine and coarse beat unlabeled links for PCC, NLDT, and GUM/eRST, with paired intervals above zero. On SFU, all intervals for adding labels include zero.
- **Does finer detail help?** Amazon NLDT flipped from a 1.30-point coarse lead with the original averages to a 0.80-point fine lead with link weighting. SFU GUM/eRST favored coarse by 1.25 points. [Results and paired intervals](../DEPENDENCY_PILOT.md#what-we-found)

## Other follow-ups

- **Combine sentiment up the tree (Bhatia-inspired).** Combine two pieces at a time, starting at the bottom. Test whether the model learns one rule for all relations, one per coarse label, or one per fine label. On Amazon the fine and coarse scores were within 0.25 macro-F1 points for all four standards. [Results](../RUNNING.md#combine-sentiment-up-the-amazon-trees)
- **Check the public coarse parsers.** Stock GUM/eRST and RST-DT used the same PCC EDUs and built different trees. Original pooling still helped over nuclearity alone on Amazon and hurt on SFU. Stock RST-DT improved Amazon bottom-up pooling from .875 to .886. This comparison measures a change of parser as well as a change in how coarse relations are produced. [Results](../RUNNING.md#rerun-sentiment-with-the-stock-trees)
