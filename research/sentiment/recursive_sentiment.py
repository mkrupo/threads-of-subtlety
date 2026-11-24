"""Learn to combine sentiment up fixed RST trees on the verified Amazon cohort."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from .baseline import metrics, read_jsonl
from .dependency_sentiment import input_files, vectors_for
from .parser_pilot import INVENTORIES, RESEARCH_REVISION, canonical_maps, digest, write_json
from .relation_stability import ITERATIONS, SEED, bootstrap_weights, macro_f1
from .score_sfu import LABELS


MODES = ("shared", "coarse", "fine")
C = 1.0
ROLES = {"NS": (0, 1), "SN": (1, 0), "NN": (0, 0)}


@dataclass(frozen=True)
class Layer:
    parents: np.ndarray
    children: np.ndarray
    roles: np.ndarray
    fine: np.ndarray
    coarse: np.ndarray

    def parameter_indices(self, mode):
        labels = {"shared": np.zeros_like(self.fine), "fine": self.fine, "coarse": self.coarse}
        return 2 * labels[mode][:, None] + self.roles


@dataclass(frozen=True)
class Forest:
    initial: np.ndarray
    roots: np.ndarray
    layers: tuple[Layer, ...]


def build_forest(records, fine_ids, coarse_ids, mapping):
    """Index nodes by height so each layer can be combined in one operation."""
    initial, roots, layers = [], [], defaultdict(list)
    for tree, score in records:
        leaves = iter(score)

        def visit(key):
            node = tree["nodes"][str(key)]
            if "left" not in node:
                initial.append(next(leaves))
                return len(initial) - 1, 0
            left, left_height = visit(node["left"])
            right, right_height = visit(node["right"])
            height = 1 + max(left_height, right_height)
            parent = len(initial)
            initial.append(0.0)
            label = node["relation"].lower()
            layers[height].append((parent, left, right, *ROLES[node["nuclearity"]],
                                   fine_ids[label], coarse_ids[mapping[label]]))
            return parent, height

        root, _ = visit(tree["root"])
        if next(leaves, None) is not None:
            raise ValueError(f"Extra EDU scores for {tree['key']}")
        roots.append(root)
    ordered = []
    for _, rows in sorted(layers.items()):
        layer = np.asarray(rows, dtype=np.int64)
        ordered.append(Layer(layer[:, 0], layer[:, 1:3], layer[:, 3:5], layer[:, 5], layer[:, 6]))
    return Forest(np.asarray(initial, dtype=np.float64), np.asarray(roots, dtype=np.int64), tuple(ordered))


def compose(forest, indices, weights):
    values = forest.initial.copy()
    for layer, keys in zip(forest.layers, indices, strict=True):
        values[layer.parents] = np.tanh(np.sum(weights[keys] * values[layer.children], axis=1))
    return values


def objective(parameters, forest, indices, truth, prior):
    bias, scale = parameters[:2]
    weights = parameters[2:]
    values = compose(forest, indices, weights)
    root_scores = values[forest.roots]
    logits = bias + scale * root_scores
    error = expit(logits) - truth
    shrinkage = (parameters - prior) / C
    loss = np.sum(np.logaddexp(0, logits) - truth * logits)
    loss += 0.5 * (parameters - prior) @ shrinkage

    gradient = shrinkage
    gradient[0] += error.sum()
    gradient[1] += error @ root_scores
    node_gradient = np.zeros_like(values)
    node_gradient[forest.roots] = scale * error
    for layer, keys in reversed(list(zip(forest.layers, indices, strict=True))):
        local = node_gradient[layer.parents] * (1 - values[layer.parents] ** 2)
        child_gradient = local[:, None] * weights[keys]
        np.add.at(node_gradient, layer.children, child_gradient)
        sensitivities = local[:, None] * values[layer.children]
        gradient[2:] += np.bincount(keys.ravel(), weights=sensitivities.ravel(), minlength=len(weights))
    return loss, gradient


def fit_predict(train, test, truth, mode, rule_count):
    train_indices = tuple(layer.parameter_indices(mode) for layer in train.layers)
    test_indices = tuple(layer.parameter_indices(mode) for layer in test.layers)
    # Start from equal contributions. Rare labels can stay near this common starting rule.
    prior = np.r_[0.0, 1.0, np.full(2 * rule_count, 0.5)]
    fit = minimize(objective, prior.copy(), jac=True, method="L-BFGS-B",
                   args=(train, train_indices, truth, prior), options={"maxiter": 500, "ftol": 1e-10})
    if not fit.success:
        raise RuntimeError(f"Recursive {mode} fit failed: {fit.message}")
    values = compose(test, test_indices, fit.x[2:])
    return expit(fit.x[0] + fit.x[1] * values[test.roots]), fit.nit


def paired_results(truth, metadata, predictions, results):
    weights = bootstrap_weights(metadata)
    draws = {name: macro_f1(weights, truth, pred) for name, pred in predictions.items()}
    paired = {}
    for inventory in INVENTORIES:
        contrasts = {}
        for left, right in (("fine", "coarse"), ("fine", "shared"), ("coarse", "shared")):
            a, b = (f"{inventory}/{mode}" for mode in (left, right))
            contrasts[f"{left}_minus_{right}"] = {
                "difference": results[a]["macro_f1"] - results[b]["macro_f1"],
                "interval_95": np.quantile(draws[a] - draws[b], [0.025, 0.975]).tolist(),
            }
        paired[inventory] = contrasts
    return paired


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    audit, representation, scenes, hashes = input_files(args)
    metadata = sorted(read_jsonl(args.prepared / "manifest.jsonl"), key=lambda row: row["document_id"])
    review_ids = {row["document_id"]: i for i, row in enumerate(metadata)}
    if Counter(scene["document_id"] for scene in scenes) != Counter(row["document_id"] for row in metadata):
        raise ValueError("The scenes differ from the prepared review set")
    if len(scenes) != len(review_ids):
        raise ValueError("This experiment expects one scene tree per review")
    splits = json.loads((args.prepared / "splits.json").read_text())["outer_folds"]
    truth = np.array([row["label"] for row in metadata], dtype=np.int16)
    vectors = vectors_for(args, scenes, representation)
    labels = tuple(representation["labels"])
    if labels != LABELS:
        raise ValueError("Unexpected sentiment score order")
    edu_scores = vectors[:, labels.index("positive")] - vectors[:, labels.index("negative")]
    scene_scores = np.split(edu_scores, np.cumsum([len(scene["edus"]) for scene in scenes[:-1]]))
    maps = canonical_maps()
    probabilities, fits = {}, {}
    for inventory in INVENTORIES:
        trees = read_jsonl(args.trees / f"{inventory}.jsonl")
        mapping = maps[inventory]
        fine = sorted({label.rsplit("_", 1)[0].lower()
                       for label in audit["inventory_reports"][inventory]["ordered_relation_table"]})
        coarse = sorted({mapping[label] for label in fine if label in mapping})
        fine_ids, coarse_ids = ({label: i for i, label in enumerate(labels)} for labels in (fine, coarse))
        records = {}
        for tree, scene, scores in zip(trees, scenes, scene_scores, strict=True):
            if (tree["key"], tree["document_id"], tree["status"]) != (scene["key"], scene["document_id"], "ok"):
                raise ValueError("A tree differs from its verified scene")
            records[scene["document_id"]] = (tree, scores)
        rule_counts = {"shared": 1, "fine": len(fine), "coarse": len(coarse)}
        for mode in MODES:
            probabilities[f"{inventory}/{mode}"] = np.full(len(truth), np.nan)
            fits[f"{inventory}/{mode}"] = []
        tested = Counter()
        for fold in splits:
            train_ids, test_ids = (sorted(fold[key]) for key in ("train_ids", "test_ids"))
            train, test = (np.array([review_ids[doc] for doc in ids]) for ids in (train_ids, test_ids))
            if np.intersect1d(train, test).size or any(metadata[i]["outer_fold"] != fold["fold"] for i in test):
                raise ValueError("The saved folds disagree with the manifest")
            tested.update(test)
            train_trees, test_trees = (build_forest((records[doc] for doc in ids), fine_ids, coarse_ids, mapping)
                                       for ids in (train_ids, test_ids))
            for mode in MODES:
                key = f"{inventory}/{mode}"
                scores, iterations = fit_predict(train_trees, test_trees, truth[train], mode, rule_counts[mode])
                probabilities[key][test] = scores
                fits[key].append({"fold": fold["fold"], "iterations": iterations})
            print(f"{inventory}: finished fold {fold['fold'] + 1}", flush=True)
        if tested != Counter(range(len(truth))):
            raise ValueError("Every review must be held out exactly once")
    if any(not np.isfinite(scores).all() for scores in probabilities.values()):
        raise ValueError("Missing or invalid held-out probabilities")
    predictions = {name: (scores >= 0.5).astype(np.int16) for name, scores in probabilities.items()}
    results = {name: metrics(truth, prediction) for name, prediction in predictions.items()}
    paired = paired_results(truth, metadata, predictions, results)
    args.output.mkdir(parents=True)
    path = args.output / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for condition, scores in sorted(probabilities.items()):
            for row, score, prediction in zip(metadata, scores, predictions[condition], strict=True):
                stream.write(json.dumps({
                    "document_id": row["document_id"], "outer_fold": row["outer_fold"],
                    "label": row["label"], "condition": condition, "probability": float(score),
                    "prediction": int(prediction),
                }) + "\n")
    write_json(args.output / "report.json", {
        "protocol": "recursive-sentiment-amazon-v1", "reviews": len(truth), "edus": len(vectors),
        "input_hashes": hashes, "mapping_revision": RESEARCH_REVISION,
        "source_sha256": digest(Path(__file__)),
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
        "edu_score": "positive probability minus negative probability",
        "regularization_c": C, "outer_folds": len(splits), "decision_threshold": 0.5,
        "resampling": {"draws": ITERATIONS, "seed": SEED, "unit": "review", "paired": True,
                       "strata": ["outer_fold", "domain", "label"]},
        "results": results, "paired": paired, "fits": fits, "predictions_sha256": digest(path),
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.set_defaults(corpus="amazon", arm="cardiff")
    cli.add_argument("--prepared", type=Path, required=True)
    cli.add_argument("--trees", type=Path, required=True)
    cli.add_argument("--representation", type=Path, required=True, help="Frozen CardiffNLP scores.jsonl")
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
