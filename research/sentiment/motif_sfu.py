"""Test HC3-style M3 frequency and depth on the frozen SFU review folds."""

import argparse
from collections import Counter
from functools import cache
from itertools import combinations, permutations, product
import json
from pathlib import Path
import re
import sys
import warnings

import networkx as nx
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import HERE, INPUT_FILES, digest, metrics, read_jsonl, validate_prepared, write_json
from .edu_baseline import RESULTS
from .parser_pilot import RESEARCH_REVISION, canonical_maps
from .relation_stability import ITERATIONS, SEED, bootstrap_weights, macro_f1, pair_predictions


PERMUTED_EDGES = tuple(tuple(a * 3 + b for a, b in product(order, repeat=2))
                       for order in permutations(range(3)))


@cache
def motif_signature(edges):
    """Canonical relation-labeled adjacency and number of symmetric matches."""
    forms = Counter(tuple(edges[i] for i in order) for order in PERMUTED_EDGES)
    key = min(forms)
    return key, forms[key]


def discourse_graph(tree):
    graph = nx.DiGraph()
    for parent, node in tree["nodes"].items():
        if "left" not in node:
            continue
        left, right = str(node["left"]), str(node["right"])
        graph.add_edges_from(((left, parent), (right, parent)), label_0="/")
        relation = re.sub(r"[\s,:]+", "-", node["relation"].strip())
        roles = node["nuclearity"]
        if roles in ("SN", "NN"):
            graph.add_edge(left, right, label_0=relation)
        if roles in ("NS", "NN"):
            graph.add_edge(right, left, label_0=relation)
    return graph


def scene_motifs(tree):
    graph = discourse_graph(tree)
    counts, depth_totals = Counter(), Counter()
    if not graph:
        return counts, depth_totals
    undirected = graph.to_undirected(as_view=True)
    lengths = nx.single_source_shortest_path_length(undirected, str(tree["root"]))
    diameter = nx.diameter(undirected, usebounds=True)
    labels = nx.get_edge_attributes(graph, "label_0")
    seen = set()
    for center in undirected:
        for left, right in combinations(undirected.neighbors(center), 2):
            nodes = (center, left, right)
            nodes_key = frozenset(nodes)
            if nodes_key in seen:
                continue
            seen.add(nodes_key)
            edges = tuple(labels.get((a, b), "") for a, b in product(nodes, repeat=2))
            key, matches = motif_signature(edges)
            counts[key] += matches
            depth_totals[key] += matches * sum(lengths[node] for node in nodes) / (3 * diameter)
    return counts, depth_totals


def review_motifs(trees):
    pooled = {}
    for tree in trees:
        if tree["status"] != "ok":
            raise ValueError(f"Expected a verified tree for scene {tree['key']}")
        counts, depths = pooled.setdefault(tree["document_id"], (Counter(), Counter()))
        scene_counts, scene_depths = scene_motifs(tree)
        counts.update(scene_counts)
        depths.update(scene_depths)
    return {name: {key: (count, depths[key]) for key, count in counts.items()}
            for name, (counts, depths) in pooled.items()}


class MotifFeatures(TransformerMixin, BaseEstimator):
    """Fit a fine catalog and optionally project that catalog to coarse names."""

    def __init__(self, mapping=None):
        self.mapping = mapping

    def fit(self, reviews, y=None):
        self.fine_catalog_ = sorted({key for review in reviews for key in review})
        if self.mapping is None:
            projected = self.fine_catalog_
        else:
            # Coarsening changes matching groups but preserves fine match counts.
            projected = [motif_signature(tuple(
                self.mapping[label.lower()] if label not in ("", "/") else label for label in key
            ))[0] for key in self.fine_catalog_]
        self.catalog_ = sorted(set(projected))
        indices = {key: i for i, key in enumerate(self.catalog_)}
        self.assignment_ = {fine: indices[target] for fine, target in zip(self.fine_catalog_, projected, strict=True)}
        return self

    def transform(self, reviews):
        shape = (len(reviews), len(self.catalog_))
        counts = np.zeros(shape, dtype=np.int64)
        depths = np.zeros(shape, dtype=np.float64)
        for row, review in enumerate(reviews):
            for key, (count, depth_total) in review.items():
                index = self.assignment_.get(key)
                if index is not None:
                    counts[row, index] += count
                    depths[row, index] += depth_total
        totals = counts.sum(axis=1, keepdims=True)
        values = np.zeros((len(reviews), 2 * len(self.catalog_)), dtype=np.float64)
        np.divide(counts, totals, out=values[:, ::2], where=totals > 0)
        np.divide(depths, counts, out=values[:, 1::2], where=counts > 0)
        return values


def fit_fold(reviews, by_id, fold, config, mapping):
    train, test = sorted(fold["train_ids"]), sorted(fold["test_ids"])
    options = {**config["logistic_regression"], "random_state": config["seed"]}
    model = make_pipeline(MotifFeatures(mapping), StandardScaler(), LogisticRegression(**options))
    inner = [fold["inner_folds"][name] for name in train]
    search = GridSearchCV(
        model, {"logisticregression__C": config["regularization_C"]},
        scoring=make_scorer(f1_score, labels=[0, 1], average="macro", zero_division=0),
        cv=PredefinedSplit(inner), n_jobs=1, error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit([reviews[name] for name in train], [by_id[name]["label"] for name in train])
    positive = search.predict_proba([reviews[name] for name in test])[:, list(search.classes_).index(1)]
    held_out = [{"document_id": name, "label": by_id[name]["label"], "domain": by_id[name]["domain"],
                 "fold": fold["fold"], "probability_positive": float(probability),
                 "prediction": int(probability >= config["decision_threshold"])}
                for name, probability in zip(test, positive, strict=True)]
    fitted = search.best_estimator_.named_steps["motiffeatures"]
    return held_out, fitted, {
        "fold": fold["fold"], "selected_C": search.best_params_["logisticregression__C"],
        "mean_inner_macro_f1_by_C": dict(zip(map(str, config["regularization_C"]),
                                            search.cv_results_["mean_test_score"].tolist(), strict=True)),
        "motifs": len(fitted.catalog_),
        "metrics": metrics([r["label"] for r in held_out], [r["prediction"] for r in held_out]),
    }


def contrast(first, second):
    point, draws = first
    baseline, baseline_draws = second
    return {"macro_f1_difference": point - baseline,
            "interval_95": np.quantile(draws - baseline_draws, [0.025, 0.975]).tolist()}


def run(prepared, trees, output):
    audit = json.loads((RESULTS / "sfu-full-parser-audit.json").read_text())
    reference = json.loads((RESULTS / "sfu-relation-pooling-v1.json").read_text())
    config_path = HERE / "baseline_config.json"
    inputs = {name: digest(prepared / name) for name in INPUT_FILES}
    inputs["baseline_config.json"] = digest(config_path)
    inputs.update({f"{inv}.jsonl": digest(trees / f"{inv}.jsonl") for inv in audit["inventory_reports"]})
    if any(checksum != reference["input_hashes"][name] for name, checksum in inputs.items()):
        raise ValueError("Use the frozen SFU data, trees, folds and LR configuration")
    if RESEARCH_REVISION != audit["research_revision"]:
        raise ValueError("The map revision differs from the parser audit")
    output.mkdir(parents=True)

    config = json.loads(config_path.read_text())
    records, splits = validate_prepared(prepared, config)
    by_id = {r["document_id"]: r for r in records}
    inventories, predictions = {}, []
    maps = canonical_maps()
    for inventory in audit["inventory_reports"]:
        reviews = review_motifs(read_jsonl(trees / f"{inventory}.jsonl"))
        if reviews.keys() != by_id.keys():
            raise ValueError(f"Missing review trees for {inventory}")
        rows_by_mode = {"fine": [], "coarse": []}
        fits = {"fine": [], "coarse": []}
        for fold in splits["outer_folds"]:
            fine_rows, fine, fine_fit = fit_fold(reviews, by_id, fold, config, None)
            coarse_rows, coarse, coarse_fit = fit_fold(reviews, by_id, fold, config, maps[inventory])
            if fine.fine_catalog_ != coarse.fine_catalog_:
                raise ValueError("Fine and coarse training catalogs differ")
            all_reviews = list(reviews.values())
            fine_mass = fine.transform(all_reviews)[:, ::2].sum(axis=1)
            coarse_mass = coarse.transform(all_reviews)[:, ::2].sum(axis=1)
            if not np.allclose(fine_mass, coarse_mass, rtol=0, atol=1e-14):
                raise ValueError("Projection changed retained motif frequency mass")
            for mode, rows, fit in (("fine", fine_rows, fine_fit), ("coarse", coarse_rows, coarse_fit)):
                rows_by_mode[mode].extend(rows)
                fits[mode].append(fit)
        inventories[inventory] = {
            mode: {"pooled_out_of_fold": metrics([r["label"] for r in rows], [r["prediction"] for r in rows]),
                   "folds": fits[mode]} for mode, rows in rows_by_mode.items()
        }
        for mode, rows in rows_by_mode.items():
            predictions.extend({**row, "inventory": inventory, "granularity": mode} for row in rows)
        print(f"Finished {inventory} fine and coarse", flush=True)

    metadata = sorted(read_jsonl(prepared / "manifest.jsonl"), key=lambda r: r["document_id"])
    truth = np.array([row["label"] for row in metadata], dtype=np.int16)
    weights = bootstrap_weights(metadata)
    points_and_draws = {}
    for inventory, data in inventories.items():
        paired = pair_predictions(predictions, metadata, inventory)
        for mode, predicted in zip(("fine", "coarse"), paired, strict=True):
            points_and_draws[inventory, mode] = data[mode]["pooled_out_of_fold"]["macro_f1"], macro_f1(weights, truth, predicted)
        data["fine_minus_coarse"] = contrast(points_and_draws[inventory, "fine"], points_and_draws[inventory, "coarse"])
    between = {mode: [
        {"first": left, "second": right, **contrast(points_and_draws[left, mode], points_and_draws[right, mode])}
        for left, right in combinations(inventories, 2)
    ] for mode in ("fine", "coarse")}

    path = output / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    report = {
        "protocol": "sfu-motif-v1", "research_revision": RESEARCH_REVISION,
        "features": "M3 frequency and weighted average depth from the training-partition fine catalog and its coarse projection",
        "primary_metric": config["primary_metric"], "inputs_sha256": inputs,
        "predictions_sha256": digest(path), "reviews": len(by_id),
        "source_sha256": {name: digest(HERE / name) for name in (
            "motif_sfu.py", "baseline.py", "parser_pilot.py", "relation_stability.py", "uv.lock",
        )},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "networkx": nx.__version__, "scikit_learn": sklearn.__version__},
        "resampling": {"seed": SEED, "draws": ITERATIONS, "strata": ["outer_fold", "domain", "gold_label"],
                       "intervals": "pointwise percentile intervals conditional on saved held-out predictions"},
        "inventories": inventories, "between_inventories": between,
    }
    write_json(output / "report.json", report)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prepared-dir", type=Path, required=True)
    cli.add_argument("--trees-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    run(args.prepared_dir, args.trees_dir, args.output_dir)


if __name__ == "__main__":
    main()
