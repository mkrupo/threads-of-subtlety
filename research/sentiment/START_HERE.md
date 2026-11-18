# RST and sentiment in one page

**Our question** is whether the choice of discourse standard, or the detail
of its relation labels, changes how well we can predict a review’s
sentiment. This adds an English sentiment task alongside the paper’s
authorship and argumentation work.

## How the sentiment model would work

A regular sentiment classifier can read a review and return a positive or
negative score. SFU supplies those two review labels. In our proposed setup,
a shared sentiment scorer reads each
elementary discourse unit (EDU), usually a clause or short span. It can be a
neural model or a simpler scorer. RST provides a tree that tells us how to
combine those local scores. The model learns against the review’s overall
sentiment.

Take “The screen looks great, but the battery dies by noon.” Suppose the
two EDU scores are +0.8 and −0.6. Their average is +0.1. If a tree marks
the battery complaint as central and a learned rule gives it three times
the weight of the praise, the combined score becomes −0.25. These numbers
illustrate the idea. In the experiment, the scorer gives the same two
inputs to every condition. We measure whether the way we combine them helps
on held-out reviews. [R2N2 method](https://aclanthology.org/D15-1263/)

A typical EDU is 11 BERT tokens. A typical whole review is 727 tokens, and
256 of 400 reviews exceed the 512-token input limit. This makes the scorer's
context length a separate consideration for whole-review classification.
[Length audit](results/sfu-token-lengths.json)

## What changes in each comparison

| Comparison | What changes |
|---|---|
| Uniform versus tree aggregation | We compare averaging the EDU scores with using the tree. A tree model that ignores relation names gives us a structural control. |
| PCC, NLDT, GUM/eRST, RST-DT | Each parser builds its own tree on the same English EDUs. Their attachments, nuclearity, and relations can differ. |
| Fine versus coarse within one inventory | We take its existing tree and merge relation labels using the research-branch map. For example, eRST concession and contrast both map to `adversative`. The cuts, attachments, and nuclearity stay fixed. |

## What we have done

- **Checked the colleague’s setup on his data.** We regenerated the saved
  discourse features for six HC3 documents across all four inventories.
  Fresh classifier fits recovered the HC3 fine/coarse findings. [HC3 check](HC3_REPRODUCTION.md)
- **Prepared the sentiment dataset.** SFU has 400 English reviews, balanced
  by positive and negative recommendation across eight domains. We froze
  the review splits and ran text-only controls. [SFU controls](RUNNING.md)
- **Parsed all 400 SFU reviews under all four inventories.** We verified
  4,048 trees across 1,012 scenes and 36,850 shared EDUs. The research
  maps cover every relation observed. A review can have multiple scene
  trees, so the sentiment model will need to combine their scores.
  [Full parse audit](results/sfu-full-parser-audit.json)

**What we can say now** is that the parser setup reproduces the colleague’s
features on the HC3 sample and gives us all four trees for every SFU scene.
The sentiment comparison still needs a choice of local scorer, a rule for
combining scenes, and the held-out experiment.
