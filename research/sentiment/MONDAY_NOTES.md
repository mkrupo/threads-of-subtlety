# Monday meeting notes — RST and review sentiment

## What I tested

- I used sentiment as a downstream task. I compared four RST standards (PCC, NLDT, GUM/eRST, RST-DT) and fine versus coarse relation labels within each one.
- SFU has 400 English product reviews, evenly split between recommend and do not recommend. Long reviews are split into shorter chunks. All four parsers get the same chunks and the same clause-sized units (EDUs).
- I used the released UniRST checkpoints for PCC and NLDT and my colleague’s fine-tuned checkpoints for GUM/eRST and RST-DT. Within each standard, fine keeps the predicted relation names and coarse merges them using my colleague’s mapping. The two versions share the same tree.

## What the classifier sees

- Frozen CardiffNLP gives each EDU three probabilities in order `[negative, neutral, positive]`. Together they sum to one. Every comparison based on EDU sentiment uses these same scores.
- **Equal weight — 3 numbers per review.** Average each sentiment probability across all EDUs in the review. This gives us a baseline without tree features.
- **Nuclearity — 6 numbers per review.** The central average comes first, then the supporting average. Both use the order `[negative, neutral, positive]`.
  - Follow the path from the scene’s tree root to each EDU. A path through nucleus branches only is central. Any satellite branch makes the EDU supporting, even if it is a nucleus farther down the tree. A central path can branch into two nuclei, making several EDUs central.
  - Collect the individual EDUs from every scene into the same two review-wide groups, then average each group. Fine and coarse give identical nuclearity inputs within each standard because they share the same branches.
- Logistic regression learns from these numbers and the training reviews’ recommendation labels. In five-fold cross-validation, each review is held out once. Equal weight scored **0.817 macro-F1** and nuclearity scored **0.822–0.832**, depending on the parser.

## Names to remember

- **Data.** [SFU Review Corpus](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html)
- **Idea.** Use discourse structure to combine local sentiment. [Bhatia, Ji, and Eisenstein (2015)](https://aclanthology.org/D15-1263/)
- **EDU scorer.** [CardiffNLP Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Parser family.** [UniRST](https://arxiv.org/abs/2510.06427)
