"""Measure what review sentiment loses when one RST tree is cut into two scene trees.

SFU reviews longer than the parser window are split into scenes whose trees are
never connected. This ablation reproduces that cut on intact Amazon reviews
without changing their text units. Each review that has a sentence-final EDU
boundary is cut at the boundary that best balances its words. Both halves keep
the original PCC EDUs, so the CardiffNLP EDU scores are unchanged. Only the
tree links across the cut disappear.

``prepare``  Write the split scene file for the existing parser runner.
``analyze``  Refit the sentiment models on intact and split trees and compare
             them on the same held-out reviews.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .parser_pilot import digest, write_json


SENTENCE_END = re.compile(r"[.!?][\"'’”)\]]*$")
RULE = ("Cut before EDU i when EDU i-1 ends with . ! or ? (optionally followed by closing quotes or "
        "brackets). Choose the cut minimizing |words before - words after|, the earlier cut on ties. "
        "Reviews without such a boundary stay as one scene.")


def choose_cut(edus):
    """Return the index of the first EDU in the second scene, or None."""
    words = [len(edu.split()) for edu in edus]
    total, before, best = sum(words), 0, None
    for index in range(1, len(edus)):
        before += words[index - 1]
        if SENTENCE_END.search(edus[index - 1].rstrip()):
            gap = abs(2 * before - total)
            if best is None or gap < best[0]:
                best = (gap, index)
    return None if best is None else best[1]


def split_scenes(scenes, count_tokens):
    """Split one-scene reviews at their chosen cut and record each decision."""
    rows, decisions = [], []
    for scene in scenes:
        if scene["status"] != "ok" or not scene["key"].endswith("-s00"):
            raise ValueError(f"Expected one verified scene per review, got {scene['key']}")
        edus = scene["edus"]
        cut = choose_cut(edus)
        decisions.append({"document_id": scene["document_id"], "key": scene["key"], "edus": len(edus), "cut": cut,
                          "words_before": sum(len(e.split()) for e in edus[:cut]) if cut else None,
                          "words_after": sum(len(e.split()) for e in edus[cut:]) if cut else None})
        if cut is None:
            rows.append(scene)
            continue
        prefix = scene["key"][:-len("-s00")]
        for index, part in enumerate((edus[:cut], edus[cut:])):
            text = " ".join(part)
            rows.append({"document_id": scene["document_id"], "key": f"{prefix}-s{index:02}", "text": text,
                         "tokens": count_tokens(text), "edus": part, "status": "ok"})
    joined = lambda items: "".join("".join(edu.split()) for row in items for edu in row["edus"])
    if joined(rows) != joined(scenes):
        raise ValueError("Splitting changed the EDU sequence")
    return rows, decisions


def prepare(args):
    from tokenizers import Tokenizer

    audit = json.loads(args.intact_audit.read_text())
    scenes_path = args.intact_trees / "scenes.jsonl"
    if digest(scenes_path) != audit["segmentation"]["scenes_sha256"] or digest(args.sample) != audit["sample_sha256"]:
        raise ValueError("Intact scenes or parser sample differ from the verified Amazon parse")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    with scenes_path.open(encoding="utf-8") as stream:
        scenes = [json.loads(line) for line in stream]
    rows, decisions = split_scenes(
        scenes, lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
    args.output.mkdir(parents=True)
    with (args.output / "scenes.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    split = [row for row in decisions if row["cut"] is not None]
    write_json(args.output / "segmentation.json", {
        "scene_token_target": audit["segmentation"]["scene_token_target"],
        "seconds": 0.0,
        "scene_status": dict(Counter(row["status"] for row in rows)),
        "scenes_sha256": digest(args.output / "scenes.jsonl"),
        "derived_from_scenes_sha256": audit["segmentation"]["scenes_sha256"],
        "split_rule": RULE,
    })
    write_json(args.output / "split-manifest.json", {
        "protocol": "amazon-split-ablation-v1",
        "rule": RULE,
        "intact_scenes_sha256": audit["segmentation"]["scenes_sha256"],
        "split_scenes_sha256": digest(args.output / "scenes.jsonl"),
        "sample_sha256": audit["sample_sha256"],
        "tokenizer_sha256": digest(args.tokenizer),
        "source_sha256": digest(Path(__file__)),
        "reviews": len(decisions), "split_reviews": len(split), "scenes": len(rows),
        "edus": sum(len(row["edus"]) for row in rows),
        "decisions": decisions,
    })
    print(f"Split {len(split)} of {len(decisions)} reviews into {len(rows)} scenes", flush=True)


def split_data(intact, split_trees, split_audit_path, manifest_path, sample_path, intact_scores):
    """Inputs for the split trees: same reviews, folds and EDU vectors, new scene trees."""
    from types import SimpleNamespace

    from .baseline import read_jsonl
    from .parser_pilot import INVENTORIES

    audit = json.loads(split_audit_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    scenes = read_jsonl(split_trees / "scenes.jsonl")
    if (digest(split_trees / "scenes.jsonl") != audit["segmentation"]["scenes_sha256"]
            or manifest["split_scenes_sha256"] != audit["segmentation"]["scenes_sha256"]
            or digest(sample_path) != audit["sample_sha256"]
            or audit["complete_documents_across_inventories"] != audit["documents"]
            or any(info["unmapped_observed"] for info in audit["inventory_reports"].values())):
        raise ValueError("Split scenes, sample or parser coverage differ from the split audit")
    for name in INVENTORIES:
        if digest(split_trees / f"{name}.jsonl") != audit["inventory_reports"][name]["trees_sha256"]:
            raise ValueError(f"Split tree checksum mismatch for {name}")
    flatten = lambda rows: [edu for row in rows for edu in row["edus"]]
    if flatten(scenes) != flatten(intact.scenes):
        raise ValueError("Split scenes must keep the intact EDU sequence")
    # Re-chunk the verified intact EDU scores into the split scenes.
    flat_scores = [p for row in read_jsonl(intact_scores) for p in row["probabilities"]]
    score_rows, offset = [], 0
    for scene in scenes:
        score_rows.append({"key": scene["key"], "document_id": scene["document_id"],
                           "probabilities": flat_scores[offset:offset + len(scene["edus"])]})
        offset += len(scene["edus"])
    maps = json.loads(sample_path.read_text())["maps"]
    specs = {name: (split_trees / f"{name}.jsonl", audit["inventory_reports"][name]["ordered_relation_table"],
                    maps[name]) for name in INVENTORIES}
    hashes = {**intact.hashes, "split/scenes.jsonl": audit["segmentation"]["scenes_sha256"],
              "split/parser-audit.json": digest(split_audit_path),
              **{f"split/{name}.jsonl": audit["inventory_reports"][name]["trees_sha256"] for name in INVENTORIES}}
    data = SimpleNamespace(**{**vars(intact), "scenes": scenes, "tree_specs": specs, "hashes": hashes})
    return data, score_rows, manifest


def analyze(args):
    import numpy as np

    from .baseline import read_jsonl
    from .granularity_diagnostics import (
        CONFIGS, GateData, bootstrap_p, gate_run, load_inputs, log_loss, pooling_probabilities,
        published_predictions, resampled_means, runtime, source_hashes, summarize_fit,
    )
    from .parser_pilot import INVENTORIES
    from .relation_stability import ITERATIONS, SEED, bootstrap_weights

    intact = load_inputs("amazon", args.prepared, args.intact_trees, args.scores)
    split, split_scores, manifest = split_data(intact, args.split_trees, args.split_audit, args.split_manifest,
                                               args.sample, args.scores)
    published = published_predictions(args.gate_predictions, gate_run("amazon")["predictions_sha256"], intact.metadata)
    probabilities = {}
    for condition, data, rows, check in (("intact", intact, read_jsonl(args.scores), True),
                                         ("split", split, split_scores, False)):
        pooled, _ = pooling_probabilities(data, rows, "amazon", check_published=check)
        for name, values in pooled.items():
            if name.split("/")[-1] in ("nuclearity", "fine", "coarse") and "nucleus-first" not in name:
                probabilities[f"{condition}/{name}"] = values
        for inventory in INVENTORIES:
            gate = GateData(data, inventory)
            for mode, labels, count in (("unlabeled", np.zeros_like(gate.inv.fine_ids), 1),
                                        ("coarse", gate.inv.coarse_ids, len(gate.inv.coarse_labels)),
                                        ("fine", gate.inv.fine_ids, len(gate.inv.fine_labels))):
                probability, _ = gate.cross_validate(labels, count, use_labels=mode != "unlabeled")
                if check and np.max(np.abs(probability - published[f"{inventory}/{mode}"])) > 1e-9:
                    raise ValueError(f"Intact gate refit differs from the published {inventory}/{mode}")
                probabilities[f"{condition}/gate/{inventory}/{mode}"] = probability
        print(f"Fitted {condition} trees", flush=True)

    truth, metadata = intact.truth, intact.metadata
    split_ids = {row["document_id"] for row in manifest["decisions"] if row["cut"] is not None}
    subsets = {"all_reviews": np.arange(len(metadata)),
               "split_reviews": np.array([i for i, row in enumerate(metadata) if row["document_id"] in split_ids])}
    report = {}
    for subset, indices in subsets.items():
        weights = bootstrap_weights([metadata[i] for i in indices]).astype(np.float64)
        rows = {}
        for name in sorted(key.split("/", 1)[1] for key in probabilities if key.startswith("intact/")):
            before, after = probabilities[f"intact/{name}"][indices], probabilities[f"split/{name}"][indices]
            loss_before, loss_after = log_loss(truth[indices], before), log_loss(truth[indices], after)
            difference = resampled_means(weights, loss_after - loss_before)
            rows[name] = {
                "intact": summarize_fit(truth[indices], before), "split": summarize_fit(truth[indices], after),
                "split_minus_intact_log_loss": float((loss_after - loss_before).mean()),
                "interval_95": np.quantile(difference, [0.025, 0.975]).tolist(),
                "bootstrap_p": bootstrap_p(difference),
                "changed_predictions": int(((before >= 0.5) != (after >= 0.5)).sum()),
            }
        report[subset] = {"reviews": len(indices), "models": rows}
    args.output.mkdir(parents=True)
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for condition, values in sorted(probabilities.items()):
            for row, value in zip(metadata, values, strict=True):
                stream.write(json.dumps({"document_id": row["document_id"], "outer_fold": row["outer_fold"],
                                         "label": row["label"], "condition": condition,
                                         "probability": float(value)}) + "\n")
    write_json(args.output / "report.json", {
        "protocol": "amazon-split-ablation-v1", "rule": RULE,
        "split_reviews": len(split_ids), "reviews": len(metadata),
        "comparison": "split minus intact held-out log-loss on the same reviews; positive means splitting hurts",
        "resampling": {"draws": ITERATIONS, "seed": SEED, "unit": "review", "paired": True,
                       "strata": ["outer_fold", "domain", "label"]},
        "input_hashes": {**split.hashes, CONFIGS["amazon"].name: digest(CONFIGS["amazon"]),
                         "split-manifest.json": digest(args.split_manifest)},
        "source_hashes": source_hashes(extra=("split_ablation.py",)), "runtime": runtime(),
        "subsets": report, "predictions_sha256": digest(args.output / "predictions.jsonl"),
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--intact-trees", type=Path, required=True, help="Verified Amazon parse folder")
    prep.add_argument("--intact-audit", type=Path, required=True, help="Published Amazon parser-audit.json")
    prep.add_argument("--sample", type=Path, required=True, help="The Amazon parser sample.json")
    prep.add_argument("--tokenizer", type=Path, required=True, help="Pinned XLM-R tokenizer.json")
    prep.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("analyze")
    fit.add_argument("--prepared", type=Path, required=True, help="Amazon prepared/ folder")
    fit.add_argument("--intact-trees", type=Path, required=True, help="Verified intact Amazon parse folder")
    fit.add_argument("--scores", type=Path, required=True, help="Verified intact CardiffNLP scores.jsonl")
    fit.add_argument("--gate-predictions", type=Path, required=True, help="Published Amazon gate predictions")
    fit.add_argument("--split-trees", type=Path, required=True, help="Verified split parse folder")
    fit.add_argument("--split-audit", type=Path, required=True, help="The split parse's parser-audit.json")
    fit.add_argument("--split-manifest", type=Path, required=True, help="split-manifest.json from prepare")
    fit.add_argument("--sample", type=Path, required=True, help="The Amazon parser sample.json (relation maps)")
    fit.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "prepare":
        prepare(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
