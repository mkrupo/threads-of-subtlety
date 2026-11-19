# Monday meeting notes — RST and review sentiment

## What I tested

- I used sentiment as a downstream task. I compared four RST standards (PCC, NLDT, GUM/eRST, RST-DT) and fine versus coarse relation labels within each one.
- SFU has 400 English product reviews, evenly split between recommend and do not recommend. Long reviews are split into shorter chunks. All four parsers get the same chunks and the same clause-sized units (EDUs).
- I used the released UniRST checkpoints for PCC and NLDT and my colleague’s fine-tuned checkpoints for GUM/eRST and RST-DT. Within each standard, fine keeps the predicted relation names and coarse merges them using my colleague’s mapping. The two versions share the same tree.

## What the classifier sees

- Frozen CardiffNLP scores each EDU for negative, neutral, and positive sentiment. Every comparison based on EDU sentiment uses these same scores.
- **Equal weight — 3 numbers per review.** Average each sentiment probability across all EDUs in the review. This gives us a baseline without tree features.
- **Nuclearity — 6 numbers per review.** Average the three probabilities for all central EDUs and separately for all supporting EDUs, pooling across chunks. A central EDU is reached from its chunk’s tree root through nucleus branches only. Several EDUs in one tree can qualify.
- Logistic regression learns from these numbers and the training reviews’ recommendation labels. In five-fold cross-validation, each review is held out once. Equal weight scored **0.817 macro-F1** and nuclearity scored **0.822–0.832**, depending on the parser.

## Names to remember

- **Data.** [SFU Review Corpus](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html)
- **Idea.** Use discourse structure to combine local sentiment. [Bhatia, Ji, and Eisenstein (2015)](https://aclanthology.org/D15-1263/)
- **EDU scorer.** [CardiffNLP Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Parser family.** [UniRST](https://arxiv.org/abs/2510.06427)
