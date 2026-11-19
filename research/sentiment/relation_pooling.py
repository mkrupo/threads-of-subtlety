"""Compare fine and coarse relation-conditioned sentiment on fixed SFU trees."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
import sklearn

from .baseline import HERE, INPUT_FILES, digest, metrics, read_jsonl, validate_prepared, write_json
from .edu_baseline import RESULTS
from .nuclearity import evaluate, pool_nuclearity
from .parser_pilot import RESEARCH_REVISION, canonical_maps
from .relation_stability import ITERATIONS, SEED, bootstrap_weights, macro_f1, pair_predictions


def scene_relation_scores(tree, probabilities, *, nucleus_first=False):
    """Take the EDU mean on each side of every internal node."""
    nodes = tree["nodes"]
    leaves = iter(np.asarray(probabilities, dtype=np.float64))
    occurrences = []

    def visit(key):
        node = nodes[str(key)]
        if "left" not in node:
            return next(leaves), 1
        left, left_size = visit(node["left"])
        right, right_size = visit(node["right"])
        spans = (left / left_size, right / right_size)
        if nucleus_first and node["nuclearity"] == "SN":
            spans = spans[::-1]
        occurrences.append((node["relation"].lower(), np.concatenate(spans)))
        return left + right, left_size + right_size

    _, count = visit(tree["root"])
    if count != len(probabilities) or len(occurrences) != count - 1:
        raise ValueError(f"Relation and EDU counts disagree for scene {tree['key']}")
    return occurrences


def accumulate(groups, label, count, total):
    if label in groups:
        prior_count, prior_total = groups[label]
        count, total = count + prior_count, total + prior_total
    groups[label] = count, total


def pool_relations(scores, trees, *, nucleus_first=False):
    pooled = {}
    for score, tree in zip(scores, trees, strict=True):
        if (tree["key"], tree["document_id"], tree["status"]) != (score["key"], score["document_id"], "ok"):
            raise ValueError(f"Tree and scores disagree for scene {score['key']}")
        groups = pooled.setdefault(score["document_id"], {})
        for label, sides in scene_relation_scores(tree, score["probabilities"], nucleus_first=nucleus_first):
            accumulate(groups, label, 1, sides)
    return pooled


def coarsen(pooled, mapping):
    coarse = {}
    for document, groups in pooled.items():
        merged = {}
        for label, (count, total) in groups.items():
            if label not in mapping:
                raise ValueError(f"No canonical coarse label for observed relation {label!r}")
            accumulate(merged, mapping[label], count, total)
        coarse[document] = merged
    return coarse


def feature_rows(pooled, nuclearity, alphabet):
    columns = {label: i for i, label in enumerate(alphabet)}
    features = {}
    for document, groups in pooled.items():
        by_label = np.zeros((len(alphabet), 6), dtype=np.float64)
        for label, (count, total) in groups.items():
            by_label[columns[label]] = total / count
        features[document] = np.concatenate((nuclearity[document], by_label.ravel()))
    return features


def label_counts(pooled):
    counts = Counter()
    for groups in pooled.values():
        counts.update({label: count for label, (count, _) in groups.items()})
    return counts


def paired_change(current, reference):
    if set(row["document_id"] for row in current) != reference.keys() or len(current) != len(reference):
        raise ValueError("Paired conditions cover different reviews")
    previous = [reference[row["document_id"]] for row in current]
    if any((row["fold"], row["label"]) != (old["fold"], old["label"])
           for row, old in zip(current, previous, strict=True)):
        raise ValueError("Paired conditions have different folds or labels")
    labels = [row["label"] for row in current]
    updated = [row["prediction"] for row in current]
    original = [row["prediction"] for row in previous]
    paired = list(zip(labels, updated, original, strict=True))
    return {
        "macro_f1_delta": metrics(labels, updated)["macro_f1"] - metrics(labels, original)["macro_f1"],
        "corrected_reviews": sum(new == y and old != y for y, new, old in paired),
        "new_errors": sum(new != y and old == y for y, new, old in paired),
    }


def compare_pooling_orders(inventories, predictions, text_order, reference, metadata):
    truth = np.array([row["label"] for row in metadata], dtype=np.int16)
    weights = bootstrap_weights(metadata)
    for inventory, conditions in inventories.items():
        current = pair_predictions(predictions, metadata, inventory)
        previous = pair_predictions(text_order, metadata, inventory)
        current_draws = [macro_f1(weights, truth, row) for row in current]
        conditions["paired_fine_vs_coarse"]["bootstrap_95_percentile_interval"] = np.quantile(
            current_draws[0] - current_draws[1], [0.025, 0.975],
        ).tolist()
        for index, name in enumerate(("fine", "coarse")):
            original_score = reference["inventories"][inventory][name]["pooled_out_of_fold"]["macro_f1"]
            differences = current_draws[index] - macro_f1(weights, truth, previous[index])
            conditions[name]["paired_vs_text_order"] = {
                "macro_f1_delta": conditions[name]["pooled_out_of_fold"]["macro_f1"] - original_score,
                "bootstrap_95_percentile_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
            }


def run(prepared, scores, trees, uniform_path, nuclearity_path, output, *, nucleus_first_against=None):
    audit = json.loads((RESULTS / "sfu-full-parser-audit.json").read_text())
    reference = json.loads((RESULTS / "sfu-nuclearity-v1.json").read_text())
    config_path = HERE / "baseline_config.json"
    inputs = {name: digest(prepared / name) for name in INPUT_FILES}
    inputs.update({"baseline_config.json": digest(config_path), "scores.jsonl": digest(scores),
                   "uniform_predictions.jsonl": digest(uniform_path)})
    inputs.update({f"{name}.jsonl": digest(trees / f"{name}.jsonl") for name in audit["inventory_reports"]})
    if inputs != reference["input_hashes"]:
        raise ValueError("Use the verified inputs from the nuclearity control")
    inputs["nuclearity_predictions.jsonl"] = digest(nuclearity_path)
    if inputs["nuclearity_predictions.jsonl"] != reference["predictions_sha256"]:
        raise ValueError("Use the verified nuclearity predictions")
    if digest(HERE / "nuclearity.py") != reference["source_hashes"]["nuclearity.py"]:
        raise ValueError("Nuclearity extraction differs from the recorded control")
    if RESEARCH_REVISION != audit["research_revision"]:
        raise ValueError("Relation maps differ from the parser audit")

    nucleus_first = nucleus_first_against is not None
    if nucleus_first:
        text_order_reference = json.loads((RESULTS / "sfu-relation-pooling-v1.json").read_text())
        inputs["text_order_predictions.jsonl"] = digest(nucleus_first_against)
        if (inputs["text_order_predictions.jsonl"] != text_order_reference["predictions_sha256"]
                or any(inputs[name] != checksum for name, checksum in text_order_reference["input_hashes"].items())):
            raise ValueError("Use the verified text-order predictions and their original inputs")
        text_order_predictions = read_jsonl(nucleus_first_against)

    config = json.loads(config_path.read_text())
    records, splits = validate_prepared(prepared, config)
    by_id = {row["document_id"]: row for row in records}
    uniform = {row["document_id"]: row for row in read_jsonl(uniform_path)}
    nuclearity_predictions = read_jsonl(nuclearity_path)
    score_rows = read_jsonl(scores)
    maps = canonical_maps()
    predictions, inventories = [], {}
    for inventory, parser in audit["inventory_reports"].items():
        tree_rows = read_jsonl(trees / f"{inventory}.jsonl")
        mapping = maps[inventory]
        nuclearity, coverage = pool_nuclearity(score_rows, tree_rows)
        if coverage != reference["inventories"][inventory]["coverage"]:
            raise ValueError(f"Nuclearity coverage differs from the control for {inventory}")
        fine = pool_relations(score_rows, tree_rows, nucleus_first=nucleus_first)
        if fine.keys() != by_id.keys() or label_counts(fine) != parser["observed_relations"]:
            raise ValueError(f"Relation or review coverage differs from the audit for {inventory}")
        coarse = coarsen(fine, mapping)
        for document, groups in fine.items():
            if sum(count for count, _ in groups.values()) != sum(count for count, _ in coarse[document].values()):
                raise ValueError(f"Projection changed occurrence coverage for {inventory}/{document}")

        alphabet = sorted({label.rsplit("_", 1)[0].lower() for label in parser["ordered_relation_table"]})
        coarse_alphabet = sorted({mapping[label] for label in alphabet if label in mapping})
        matched_nuclearity = {row["document_id"]: row for row in nuclearity_predictions if row["inventory"] == inventory}
        conditions = {}
        condition_predictions = {}
        for name, groups, columns in (("fine", fine, alphabet), ("coarse", coarse, coarse_alphabet)):
            features = feature_rows(groups, nuclearity, columns)
            rows, result = evaluate(features, by_id, splits, config, uniform, inventory)
            predictions.extend({**row, "granularity": name} for row in rows)
            condition_predictions[name] = rows
            conditions[name] = {
                **result,
                "paired_vs_nuclearity": paired_change(rows, matched_nuclearity),
                "feature_columns": 6 + 6 * len(columns),
                "relation_columns_in_order": columns,
                "label_occurrences": dict(label_counts(groups)),
                "mean_distinct_labels_per_review": sum(map(len, groups.values())) / len(groups),
            }
        inventories[inventory] = {
            **conditions,
            "paired_fine_vs_coarse": paired_change(condition_predictions["fine"], {
                row["document_id"]: row for row in condition_predictions["coarse"]
            }),
            "unmapped_declared_labels": sorted(set(alphabet) - mapping.keys()),
            "map_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
        }
        if nucleus_first:
            inventories[inventory]["reordered_nodes"] = sum(
                node["nuclearity"] == "SN" for tree in tree_rows for node in tree["nodes"].values() if "left" in node
            )

    if nucleus_first:
        metadata = sorted(read_jsonl(prepared / "manifest.jsonl"), key=lambda row: row["document_id"])
        compare_pooling_orders(inventories, predictions, text_order_predictions, text_order_reference, metadata)

    output.mkdir(parents=True)
    path = output / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    report = {
        "protocol": "sfu-relation-pooling-v1",
        "primary_metric": reference["primary_metric"],
        "research_revision": RESEARCH_REVISION,
        "features": {
            "base": reference["features"],
            "per_node": "arithmetic mean across the EDUs in each child span",
            "score_order_per_side": reference["features"]["scores_per_group"],
            "side_order": ["left", "right"],
            "per_label": "mean across occurrences within a review, after mapping relation names",
            "missing_label": "six zeros",
        },
        "input_hashes": inputs,
        "source_hashes": {name: digest(HERE / name) for name in (
            "relation_pooling.py", "nuclearity.py", "edu_baseline.py", "baseline.py", "parser_pilot.py", "uv.lock",
        )},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "predictions_sha256": digest(path),
        "inventories": inventories,
    }
    if nucleus_first:
        report["protocol"] = "sfu-relation-nucleus-first-v1"
        report["features"]["side_order"] = ["nucleus for NS/SN, left for NN", "satellite for NS/SN, right for NN"]
        report["source_hashes"]["relation_stability.py"] = digest(HERE / "relation_stability.py")
        report["resampling"] = {
            "seed": SEED, "draws": ITERATIONS, "unit": "review", "paired": True,
            "strata": ["outer_fold", "domain", "gold_label"],
        }
    write_json(output / "report.json", report)
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prepared-dir", type=Path, required=True)
    cli.add_argument("--scores", type=Path, required=True)
    cli.add_argument("--trees-dir", type=Path, required=True)
    cli.add_argument("--uniform-predictions", type=Path, required=True)
    cli.add_argument("--nuclearity-predictions", type=Path, required=True)
    cli.add_argument("--nucleus-first", type=Path, metavar="TEXT_ORDER_PREDICTIONS",
                     help="Order mononuclear relations by role and compare against verified text-order predictions")
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    report = run(args.prepared_dir, args.scores, args.trees_dir, args.uniform_predictions,
                 args.nuclearity_predictions, args.output_dir, nucleus_first_against=args.nucleus_first)
    for inventory, result in report["inventories"].items():
        fine = result["fine"]["pooled_out_of_fold"]["macro_f1"]
        coarse = result["coarse"]["pooled_out_of_fold"]["macro_f1"]
        print(f"{inventory}: fine {fine:.4f}, coarse {coarse:.4f}, delta {fine - coarse:+.4f}")


if __name__ == "__main__":
    main()
