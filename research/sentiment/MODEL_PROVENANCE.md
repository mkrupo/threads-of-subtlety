# What we can recover, and what to obtain — 2026-11-17

This audit supplements [PLAN.md](PLAN.md). Goal: reuse the colleagues' representation conditions reproducibly, with sentiment as a testbed. Novel sentiment methods and state-of-the-art scores are not requirements.

## Immediate answer

We can begin corpus preparation, controls, and local sentiment modeling independently. We can also implement and test parser adapters with small fixtures. For a four-standard run matching the colleague's full-inventory conditions, obtain the actual fine-tuned artifacts and their run metadata. A code recipe or output directory called `.new` does not identify a trained model.

## 1. Relation sources already recoverable

The draft explicitly identifies RST-DT, the RST Website-derived NLDT scheme, PCC, and eRST/GUM, citing their corpus/framework publications at `latex/acl_latex.tex:147`. Its resource-level counts and shared PCC segmentation are at lines 214–225. Manuals and original annotated training corpora are not embedded in this repository.

Operational coarse maps are available on the research branch `f06309b62cbc53dce0c98668cac225666dfb64d7`:

| Condition | Definition in code | Coverage in that version |
|---|---|---|
| PCC/NLDT | `scripts/canonical/ablate_m3_granularity.py::RELATION_MAP` | 31 source label strings, 14 occupied coarse targets; a shared union map, not each inventory's complete alphabet. |
| Full GUM/eRST | `scripts/canonical/ablate_full_inventory_granularity.py::FINE_TO_PARENT` | 32 native names plus `__OTHER__`, mapping to 15 native parents plus `__OTHER__`. |
| Full RST-DT | `docs/eng.rst.rstdt.new_to_coarse.json` | 60 native names plus `__OTHER__`, mapping to 18 native targets plus `__OTHER__`. |

They are different operational coarsenings, not one universal 18-label taxonomy. eRST here means the primary predicted tree; secondary edges and signals are excluded. NLDT/RST-classic means the specific NLDT implementation, not every historical RST relation taxonomy.

## 2. Additional recovery in the pinned dependency

The top-level development requirements pin **`ruthenian8/isanlp_rst` at `c51480a10a6636ae621fcc4676b12c0099f7a0f9`**. Inspecting that exact dependency revealed training/label information not present in this repository itself. Earlier handover statements that the complete recipe was not found *in this repo* should be read alongside this new recovery.

Pinned source base: [upstream fork](https://github.com/ruthenian8/isanlp_rst/tree/c51480a10a6636ae621fcc4676b12c0099f7a0f9).

| Source | Recovered decision |
|---|---|
| [`universal_parser/finetune_gum_relations.py`](https://github.com/ruthenian8/isanlp_rst/blob/c51480a10a6636ae621fcc4676b12c0099f7a0f9/isanlp_rst/universal_parser/finetune_gum_relations.py) | Base `tchewik/isanlp_rst_v3`, immutable revision `9407970f1d9d2435b5f875a0cd14293a16646304`; GUM 11.1.0; fine head; warm initialization; default relation-head-only training, optional full-model training. |
| [`universal_parser/finetune_rstdt_relations.py`](https://github.com/ruthenian8/isanlp_rst/blob/c51480a10a6636ae621fcc4676b12c0099f7a0f9/isanlp_rst/universal_parser/finetune_rstdt_relations.py) | Same pinned base; converted suffix-merged RST-DT labels; DISRPT 2025 partition provenance; analogous training modes. |
| [`universal_parser/relation_finetuning.py`](https://github.com/ruthenian8/isanlp_rst/blob/c51480a10a6636ae621fcc4676b12c0099f7a0f9/isanlp_rst/universal_parser/relation_finetuning.py) | Explicit fine-label-to-released-parent maps, head replacement/initialization and freezing helpers. |
| [`universal_parser/src/parser/data.py`](https://github.com/ruthenian8/isanlp_rst/blob/c51480a10a6636ae621fcc4676b12c0099f7a0f9/isanlp_rst/universal_parser/src/parser/data.py) | Fine label tables: GUM 50 relation/nuclearity classes covering 32 relation names; RST-DT 110 classes covering 69 suffix-merged relation names. Counts include nuclearity variants. |
| [`parser.py`](https://github.com/ruthenian8/isanlp_rst/blob/c51480a10a6636ae621fcc4676b12c0099f7a0f9/isanlp_rst/parser.py) and [`universal_parser/predictor.py`](https://github.com/ruthenian8/isanlp_rst/blob/c51480a10a6636ae621fcc4676b12c0099f7a0f9/isanlp_rst/universal_parser/predictor.py) | `Parser(relation_head_dir=...)` delegates to an artifact loader that checks base revision, task, training scope, weights, and ordered label inventory. |

This is a recovered **recipe and artifact contract**, not proof these exact defaults produced the historical HC3 results. The draft says ten epochs; these scripts default to twenty and allow head-only or full-model training. Actual saved configuration must resolve that difference. A later iteration is a possible explanation; do not treat either source as automatically mistaken.

The top-level `UniRSTAdapter` only passes generic model name/version and inventory arguments. It does not expose `relation_head_dir`, and its `.new` analysis keys are not a demonstrated loader configuration for these artifacts. Connecting actual fine artifacts requires an explicit adapter/condition registry.

Stock public-parser `hf_model_version` selects among named variants such as `unirst`; the internal predictor also uses it as a Hugging Face revision. Passing an arbitrary hash through the public wrapper's variant selector would fail its dispatch. Pin the effective snapshot in a deliberate loader implementation rather than assuming a CLI substitution is sufficient.

## 3. Individual Ettin models are specified more precisely

Feature branch `62b6fbc1f7cc3f6011e455f4134af3f3cf138309`, `tos/ettin.py`, pins:

| Model | Revision |
|---|---|
| `larc-iu/dmrst-ettin-400m-gum-12.1.0` | `e33906a27f89b2703df8ab19e471aa82141d62cb` |
| `larc-iu/dmrst-ettin-400m-rstdt-coarse` | `26594833545ef713d14e2ea6e9a7670c881c143a` |

Dipo itself is pinned at `4debc92e41140e5bcd9aadb987a297244c0fb14a`. Shared segmentation is GUM-based. This is GUM 12.1.0 versus the recovered UniRST fine recipe's GUM 11.1.0. PCC/NLDT individual checkpoints are not configured. Treat these as separate named conditions, not replacements with only a speed difference.

## 4. Static mapping checks: substantive points to resolve

[Audit output](model-provenance-audit.json) and [audit script](audit_model_provenance.py) record a comparison of pinned upstream tables with the canonical RST-DT projection. No model or training corpus was loaded.

- The canonical map covers 60 of the upstream fine table's 69 native names; nine possible names are absent. A new corpus could trigger them. Expand the mapping by an explicit reviewed rule or fail visibly; do not silently drop their occurrences.
- After normalizing only `cause`/`causal` and `contrast`/`adversative` category names, three grouping differences remain:

| Fine relation | Canonical downstream projection | Upstream warm-initialization parent |
|---|---|---|
| `preference` | `adversative` | `comparison` |
| `reason` | `causal` | `explanation` |
| `restatement` | `elaboration` | `summary` |

These maps serve different purposes and may intentionally differ. The check does not establish a bug or the linguistically correct assignment. It identifies choices that affect a granularity experiment and need an explanation, source, or declared sensitivity. Historical mapping-review tooling exists, but completed review judgments were not located.

## 5. Narrow request for the colleague

> To align the sentiment experiment, could you share the actual full-GUM and full-RST-DT run directories/checkpoints used for the reported parses, including their saved configs and ordered relation tables? Please identify the exact versions for PCC/NLDT as well, confirm the segmentation/scene policy the new task should share, and say whether the committed coarse maps are final—particularly the RST-DT grouping choices and any additional labels. We recovered the training scripts, so the main need is identifying the actual artifacts and decisions used, rather than reconstructing the whole method.

For artifacts matching the recovered upstream loader, the minimal bundle is:

- `config.json`;
- `relation_table_eng.erst.gum.txt` or `relation_table_eng.rst.rstdt.txt`;
- `best_relation_head.pt` for a head-only run, or `best_weights.pt` for a full-model run;
- preferably the training command and selection/test metrics as provenance.

If the historical models use another artifact format, request its actual loader and model files instead. Hosted IDs with immutable revisions can substitute for local files. ArgMicrotexts code remains useful for task alignment, but is not necessary to implement the sentiment baseline.

## 6. Repeating the static check

Obtain the two source files `relation_finetuning.py` and `src/parser/data.py` from the pinned upstream revision, then run:

```bash
python research/sentiment/audit_model_provenance.py \
  --fine-tuning-source /path/to/relation_finetuning.py \
  --relation-tables-source /path/to/data.py \
  --output /tmp/model-provenance-audit.json
```

The output records source hashes, table sizes, missing map coverage, and different grouping assignments. It is static verification, not validation of a learned parser or confirmation of the historical model identity.
