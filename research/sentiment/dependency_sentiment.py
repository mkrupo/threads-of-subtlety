"""Test learned relation weights over fixed RST dependencies for review sentiment."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import HERE, metrics, read_jsonl
from .encode_mdeberta import MODEL, REVISION
from .nuclearity import central_leaves
from .parser_pilot import INVENTORIES, RESEARCH_REVISION, canonical_maps, digest, write_json
from .relation_gate import PairGroups, fit_predict
from .relation_stability import ITERATIONS, SEED, bootstrap_weights, macro_f1


COMPONENTS = 8
REGULARIZATION_C = 1.0


def dependency_edges(tree):
    """Return (dependent EDU, head EDU, relation) with leftmost NN heads."""
    nodes = tree["nodes"]
    edges = []
    position = 0

    def visit(key):
        nonlocal position
        node = nodes[str(key)]
        if "left" not in node:
            edu = position
            position += 1
            return edu
        left, right = visit(node["left"]), visit(node["right"])
        if node["nuclearity"] == "SN":
            dependent, head = left, right
        else:
            dependent, head = right, left
        edges.append((dependent, head, node["relation"].lower()))
        return head

    root = visit(tree["root"])
    if {edge[0] for edge in edges} != set(range(position)) - {root} or len(edges) != position - 1:
        raise ValueError(f"Invalid dependency projection for scene {tree['key']}")
    return edges


@dataclass(frozen=True)
class Inventory:
    central: np.ndarray
    reviews: np.ndarray
    dependents: np.ndarray
    heads: np.ndarray
    fine_ids: np.ndarray
    coarse_ids: np.ndarray
    fine_labels: tuple[str, ...]
    coarse_labels: tuple[str, ...]


def build_inventory(trees, scenes, review_ids, label_table, mapping):
    fine = tuple(sorted({label.rsplit("_", 1)[0].lower() for label in label_table}))
    coarse = tuple(sorted({mapping[label] for label in fine if label in mapping}))
    fine_index = {label: i for i, label in enumerate(fine)}
    coarse_index = {label: i for i, label in enumerate(coarse)}
    links, central, offset = [], [], 0
    for tree, scene in zip(trees, scenes, strict=True):
        if (tree["key"], tree["document_id"], tree["status"]) != (scene["key"], scene["document_id"], "ok"):
            raise ValueError("A tree differs from its verified scene")
        flags = central_leaves(tree)
        if len(flags) != len(scene["edus"]):
            raise ValueError("Tree and score order disagree")
        central.extend(flags)
        for dependent, head, label in dependency_edges(tree):
            links.append((review_ids[scene["document_id"]], offset + dependent, offset + head,
                          fine_index[label], coarse_index[mapping[label]]))
        offset += len(flags)
    reviews, dependents, heads, fine_ids, coarse_ids = np.asarray(links, dtype=np.int64).reshape(-1, 5).T
    return Inventory(np.asarray(central, dtype=bool), reviews, dependents, heads, fine_ids, coarse_ids, fine, coarse)


def group_pairs(values, reviews, labels, review_count, label_count):
    groups, inverse, counts = np.unique(reviews * label_count + labels, return_inverse=True, return_counts=True)
    sums = np.zeros((len(groups), values.shape[1]))
    np.add.at(sums, inverse, values)
    return PairGroups(groups // label_count, groups % label_count, counts, sums / counts[:, None],
                      review_count, label_count)


def average(values, reviews, review_count):
    sums = np.zeros((review_count, values.shape[1]))
    np.add.at(sums, reviews, values)
    counts = np.bincount(reviews, minlength=review_count)
    divide = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    return divide, counts


def fit_plain(features, truth, train, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=REGULARIZATION_C, max_iter=300))
    model.fit(features[train], truth[train])
    return model.predict_proba(features[test])[:, 1]


def input_files(args):
    if args.corpus == "sfu":
        audit_path = HERE / "results/sfu-full-parser-audit.json"
        scorer_path = HERE / "results/sfu-edu-scores-v1.json"
    else:
        audit_path = HERE / "replication/results/amazon/parser-audit.json"
        scorer_path = HERE / "replication/results/amazon/scorer.json"
    audit = json.loads(audit_path.read_text())
    if audit["research_revision"] != RESEARCH_REVISION:
        raise ValueError("Parser and coarse mapping revisions differ")
    checks = [(args.prepared / name, audit["prepared_inputs_sha256"][name])
              for name in ("manifest.jsonl", "splits.json")]
    scenes_path = args.trees / "scenes.jsonl"
    scene_hash = audit["segmentation"]["scenes_sha256"]
    checks.append((scenes_path, scene_hash))
    checks.extend((args.trees / f"{name}.jsonl", audit["inventory_reports"][name]["trees_sha256"])
                  for name in INVENTORIES)
    if args.arm == "cardiff":
        representation = json.loads(scorer_path.read_text())
        checks.append((args.representation, representation["scores_sha256"]))
    else:
        representation = json.loads((args.representation / "manifest.json").read_text())
        if (representation["model"], representation["revision"], representation["pooling"]) != (MODEL, REVISION, "CLS"):
            raise ValueError("Expected the frozen mDeBERTa CLS cache")
        checks.append((args.representation / "embeddings.npy", representation["embeddings_sha256"]))
    if representation["scenes_sha256"] != scene_hash:
        raise ValueError("Representation and tree data have different scene order")
    for path, expected in checks:
        if digest(path) != expected:
            raise ValueError(f"Checksum mismatch for {path}")
    return audit, representation, read_jsonl(scenes_path), {path.name: expected for path, expected in checks}


def vectors_for(args, scenes, representation):
    if args.arm == "cardiff":
        scores = read_jsonl(args.representation)
        rows = []
        for scene, score in zip(scenes, scores, strict=True):
            if (score["key"], score["document_id"], len(score["probabilities"])) != (
                    scene["key"], scene["document_id"], len(scene["edus"])):
                raise ValueError("The CardiffNLP cache differs from the verified EDU sequence")
            rows.extend(score["probabilities"])
        vectors = np.asarray(rows, dtype=np.float64)
    else:
        vectors = np.load(args.representation / "embeddings.npy", allow_pickle=False)
        if list(vectors.shape) != representation["shape"]:
            raise ValueError("The mDeBERTa cache shape differs from provenance")
    if len(vectors) != sum(len(scene["edus"]) for scene in scenes) or not np.isfinite(vectors).all():
        raise ValueError("Missing or invalid EDU vectors")
    return vectors


def summaries(predictions, truth, metadata, inventories):
    weights = bootstrap_weights(metadata)
    results, intervals = {}, {}
    draws = {name: macro_f1(weights, truth, guess) for name, guess in predictions.items()}
    for name, guess in predictions.items():
        results[name] = metrics(truth, guess)

    def compare(left, right):
        return {"difference": results[left]["macro_f1"] - results[right]["macro_f1"],
                "interval_95": np.quantile(draws[left] - draws[right], [0.025, 0.975]).tolist()}

    for inventory in inventories:
        pairs = (("unlabeled", "nuclearity"), ("fine", "unlabeled"),
                 ("coarse", "unlabeled"), ("fine", "coarse"))
        intervals[inventory] = {f"{left}_minus_{right}": compare(f"{inventory}/{left}", f"{inventory}/{right}")
                                for left, right in pairs}
    return results, intervals


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    audit, representation, scenes, hashes = input_files(args)
    metadata = sorted(read_jsonl(args.prepared / "manifest.jsonl"), key=lambda row: row["document_id"])
    splits = json.loads((args.prepared / "splits.json").read_text())
    truth = np.array([row["label"] for row in metadata], dtype=np.int16)
    review_count = len(truth)
    review_ids = {row["document_id"]: i for i, row in enumerate(metadata)}
    edu_reviews = np.repeat([review_ids[scene["document_id"]] for scene in scenes],
                            [len(scene["edus"]) for scene in scenes])
    vectors = vectors_for(args, scenes, representation)
    maps = canonical_maps()
    inventories = {name: build_inventory(read_jsonl(args.trees / f"{name}.jsonl"), scenes, review_ids,
                                         audit["inventory_reports"][name]["ordered_relation_table"], maps[name])
                   for name in INVENTORIES}
    probability = defaultdict(lambda: np.full(review_count, np.nan))
    fitted_gates = defaultdict(list)
    tested = Counter()

    for fold in splits["outer_folds"]:
        train, test = (np.array([review_ids[name] for name in sorted(fold[key])])
                       for key in ("train_ids", "test_ids"))
        if np.intersect1d(train, test).size or any(metadata[i]["outer_fold"] != fold["fold"] for i in test):
            raise ValueError("The saved folds disagree with the manifest")
        tested.update(test)
        if args.arm == "mdeberta":
            pca = PCA(n_components=COMPONENTS, svd_solver="randomized", random_state=SEED)
            pca.fit(vectors[np.isin(edu_reviews, train)])
            values = pca.transform(vectors).astype(np.float64)
        else:
            values = vectors
        uniform, _ = average(values, edu_reviews, review_count)
        probability["edu_mean"][test] = fit_plain(uniform, truth, train, test)
        for name, inv in inventories.items():
            central, _ = average(values[inv.central], edu_reviews[inv.central], review_count)
            supporting, counts = average(values[~inv.central], edu_reviews[~inv.central], review_count)
            supporting[counts == 0] = central[counts == 0]
            base = np.concatenate((central, supporting), axis=1)
            probability[f"{name}/nuclearity"][test] = fit_plain(base, truth, train, test)
            pair_values = (values[inv.dependents, :, None] * values[inv.heads, None, :]).reshape(
                len(inv.heads), values.shape[1] ** 2,
            )
            for mode, labels, vocabulary in (("unlabeled", np.zeros_like(inv.fine_ids), ("all",)),
                                              ("fine", inv.fine_ids, inv.fine_labels),
                                              ("coarse", inv.coarse_ids, inv.coarse_labels)):
                pairs = group_pairs(pair_values, inv.reviews, labels, review_count, len(vocabulary))
                fitted, gates, iterations = fit_predict(
                    base[train], pairs.subset(train), truth[train], base[test], pairs.subset(test),
                    use_labels=mode != "unlabeled", c=REGULARIZATION_C,
                )
                key = f"{name}/{mode}"
                probability[key][test] = fitted
                fitted_gates[key].append({"fold": fold["fold"], "iterations": iterations,
                                          "gates": dict(zip(vocabulary, gates.tolist(), strict=True))})
        print(f"{args.corpus} {args.arm}: finished fold {fold['fold'] + 1}", flush=True)

    if tested != Counter(range(review_count)) or any(not np.isfinite(x).all() for x in probability.values()):
        raise ValueError("Every condition must predict each held-out review once")
    predictions = {name: (scores >= 0.5).astype(np.int16) for name, scores in probability.items()}
    results, intervals = summaries(predictions, truth, metadata, inventories)
    args.output.mkdir(parents=True)
    prediction_path = args.output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as stream:
        for condition, scores in sorted(probability.items()):
            for row, score, decision in zip(metadata, scores, predictions[condition], strict=True):
                stream.write(json.dumps({"document_id": row["document_id"], "outer_fold": row["outer_fold"],
                                         "label": row["label"], "condition": condition,
                                         "probability": float(score), "prediction": int(decision)}) + "\n")
    write_json(args.output / "report.json", {
        "protocol": "dependency-sentiment-v1", "corpus": args.corpus, "arm": args.arm,
        "reviews": review_count, "edus": len(vectors), "input_hashes": hashes,
        "representation": representation,
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "source_hashes": {name: digest(HERE / name) for name in ("dependency_sentiment.py", "relation_gate.py")},
        "pca_components": COMPONENTS if args.arm == "mdeberta" else None,
        "regularization_c": REGULARIZATION_C, "outer_folds": len(splits["outer_folds"]),
        "resampling": {"draws": ITERATIONS, "seed": SEED, "unit": "review", "paired": True},
        "results": results, "paired": intervals, "fitted_gates": dict(fitted_gates),
        "predictions_sha256": digest(prediction_path),
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--corpus", choices=("sfu", "amazon"), required=True)
    cli.add_argument("--arm", choices=("cardiff", "mdeberta"), required=True)
    cli.add_argument("--prepared", type=Path, required=True)
    cli.add_argument("--trees", type=Path, required=True)
    cli.add_argument("--representation", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
