"""Test whether fine RST labels carry review-sentiment information beyond their coarse groups.

Exploratory diagnostics on the existing SFU and Amazon cohorts. Both reuse the
verified trees, frozen CardiffNLP EDU scores and saved five-fold splits.

``loss``
    Paired held-out log-loss for fine versus mapped-coarse labels in the
    relation-pooling, relation-gate and (Amazon) recursive models. Relation
    pooling is refitted and must reproduce its published macro-F1. The gate and
    recursive models use their published prediction files, checked by hash.

``scramble``
    Refit the relation-gate model after shuffling fine labels among the links of
    each coarse group. Every fine label keeps its link count and coarse parent,
    but loses its connection to particular links. If real fine labels beat the
    scrambled ones, the distinctions inside coarse groups carry information
    beyond the extra parameters. The refit with real labels must reproduce the
    published fine, coarse and unlabeled predictions.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy
import sklearn

from .baseline import HERE, metrics, read_jsonl
from .dependency_sentiment import (
    REGULARIZATION_C, average, build_inventory, group_pairs, input_files, tree_inputs, vectors_for,
)
from .edu_baseline import mean_edu_scores
from .parser_pilot import INVENTORIES, digest, write_json
from .relation_gate import fit_predict
from .relation_stability import ITERATIONS, SEED, bootstrap_weights
from .stock_pooling import Evaluation, feature_sets, uniform_predictions


CORPORA = {"sfu": "SFU", "amazon": "Amazon"}
CLIP = 1e-12
REPRODUCTION_TOLERANCE = 1e-9
CONFIGS = {"sfu": HERE / "baseline_config.json",
           "amazon": HERE / "replication/results/amazon/baseline-config.json"}
PUBLISHED = {
    "comparison": HERE / "replication/results/uci/comparison.json",
    "gate": HERE / "results/sentiment3-dependency-v1.json",
    "recursive": HERE / "results/amazon-recursive-v1.json",
}


# Statistics -----------------------------------------------------------------

def log_loss(truth, probability):
    """Per-review binary log-loss with probabilities clipped away from 0 and 1."""
    p = np.clip(np.asarray(probability, dtype=np.float64), CLIP, 1 - CLIP)
    return -(truth * np.log(p) + (1 - truth) * np.log(1 - p))


def resampled_means(weights, values):
    """Weighted mean of per-review values for each bootstrap draw."""
    return (weights @ values) / weights.sum(axis=1)


def bootstrap_p(differences):
    """Two-sided percentile-bootstrap p-value for a zero difference."""
    differences = np.asarray(differences)
    return float(min(1.0, 2 * min(np.mean(differences <= 0), np.mean(differences >= 0))))


def holm(pvalues):
    """Holm step-down adjustment for one family of p-values, keyed by contrast."""
    order = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def permutation_p(observed, null, *, lower_is_better):
    """One-sided permutation p-value counting the observed fit as one draw."""
    null = np.asarray(null)
    extreme = null <= observed if lower_is_better else null >= observed
    return float((1 + extreme.sum()) / (1 + len(null)))


# Shared inputs ----------------------------------------------------------------

def load_inputs(corpus, prepared, trees, scores):
    """Verify and load the frozen review manifest, folds, trees and EDU scores."""
    args = SimpleNamespace(corpus=corpus, arm="cardiff", prepared=prepared, trees=trees,
                           representation=scores, stock=None)
    audit, representation, scenes, hashes = input_files(args)
    metadata = sorted(read_jsonl(prepared / "manifest.jsonl"), key=lambda row: row["document_id"])
    splits = json.loads((prepared / "splits.json").read_text())
    vectors = vectors_for(args, scenes, representation)
    return SimpleNamespace(args=args, audit=audit, scenes=scenes, hashes=hashes, metadata=metadata,
                           splits=splits, vectors=vectors,
                           truth=np.array([row["label"] for row in metadata], dtype=np.int16))


def published_predictions(path, expected_hash, metadata, condition_key="condition"):
    """Read one published prediction file into {condition: probability array}."""
    if digest(path) != expected_hash:
        raise ValueError(f"Prediction file differs from the published run: {path}")
    index = {row["document_id"]: i for i, row in enumerate(metadata)}
    probabilities = {}
    for row in read_jsonl(path):
        values = probabilities.setdefault(row[condition_key], np.full(len(metadata), np.nan))
        i = index[row["document_id"]]
        if row["label"] != metadata[i]["label"] or not np.isnan(values[i]):
            raise ValueError(f"Unexpected label or repeated review in {path}")
        values[i] = row["probability"]
    if any(np.isnan(values).any() for values in probabilities.values()):
        raise ValueError(f"Missing held-out predictions in {path}")
    return probabilities


def gate_run(corpus):
    runs = json.loads(PUBLISHED["gate"].read_text())["runs"]
    run, = (row for row in runs if (row["corpus"], row["arm"]) == (corpus, "cardiff"))
    return run


# Relation-gate refits -------------------------------------------------------------

class GateData:
    """Fold-independent inputs of the CardiffNLP relation-gate model for one inventory."""

    def __init__(self, data, inventory):
        tree_path, table, mapping = tree_inputs(data.args, data.audit, dict(data.hashes))[inventory]
        metadata = data.metadata
        review_ids = {row["document_id"]: i for i, row in enumerate(metadata)}
        self.inv = build_inventory(read_jsonl(tree_path), data.scenes, review_ids, table, mapping)
        self.truth = data.truth
        self.review_count = len(metadata)
        edu_reviews = np.repeat([review_ids[scene["document_id"]] for scene in data.scenes],
                                [len(scene["edus"]) for scene in data.scenes])
        values = data.vectors
        central, _ = average(values[self.inv.central], edu_reviews[self.inv.central], self.review_count)
        supporting, counts = average(values[~self.inv.central], edu_reviews[~self.inv.central], self.review_count)
        supporting[counts == 0] = central[counts == 0]
        self.base = np.concatenate((central, supporting), axis=1)
        self.pair_values = (values[self.inv.dependents, :, None] * values[self.inv.heads, None, :]).reshape(
            len(self.inv.heads), values.shape[1] ** 2,
        )
        self.folds = []
        for fold in data.splits["outer_folds"]:
            train, test = (np.array([review_ids[name] for name in sorted(fold[key])])
                           for key in ("train_ids", "test_ids"))
            if np.intersect1d(train, test).size or any(metadata[i]["outer_fold"] != fold["fold"] for i in test):
                raise ValueError("The saved folds disagree with the manifest")
            self.folds.append((fold["fold"], train, test))
        parents = {}
        for fine, coarse in zip(self.inv.fine_ids, self.inv.coarse_ids, strict=True):
            if parents.setdefault(int(fine), int(coarse)) != coarse:
                raise ValueError("A fine label maps to two coarse groups")
        self.fine_parent = np.full(len(self.inv.fine_labels), -1)
        for fine, coarse in parents.items():
            self.fine_parent[fine] = coarse

    def cross_validate(self, labels, label_count, *, use_labels=True):
        """Fit the gate model on each saved training fold and predict its test fold."""
        pairs = group_pairs(self.pair_values, self.inv.reviews, labels, self.review_count, label_count)
        probability = np.full(self.review_count, np.nan)
        fits = []
        for fold, train, test in self.folds:
            fitted, gates, iterations = fit_predict(
                self.base[train], pairs.subset(train), self.truth[train], self.base[test], pairs.subset(test),
                use_labels=use_labels, c=REGULARIZATION_C,
            )
            probability[test] = fitted
            in_train = np.isin(self.inv.reviews, train)
            fits.append({"fold": fold, "gates": gates, "iterations": int(iterations),
                         "link_counts": np.bincount(labels[in_train], minlength=label_count)})
        return probability, fits

    def scramble(self, rng):
        """Shuffle fine labels among the links of each coarse group."""
        scrambled = self.inv.fine_ids.copy()
        for group in np.unique(self.inv.coarse_ids):
            links = np.flatnonzero(self.inv.coarse_ids == group)
            scrambled[links] = scrambled[links][rng.permutation(len(links))]
        return scrambled


def within_group_spread(gates, link_counts, fine_parent):
    """Link-weighted variance of log gates among fine labels sharing a coarse group.

    Groups with fewer than two observed fine labels contribute nothing. Larger
    values mean the model weights sibling fine labels more differently.
    """
    total = weight = 0.0
    for group in np.unique(fine_parent[fine_parent >= 0]):
        members = np.flatnonzero((fine_parent == group) & (link_counts > 0))
        if len(members) < 2:
            continue
        counts = link_counts[members].astype(np.float64)
        logs = np.log(gates[members])
        centre = counts @ logs / counts.sum()
        total += counts @ (logs - centre) ** 2
        weight += counts.sum()
    return total / weight if weight else 0.0


def mean_spread(fits, fine_parent):
    return float(np.mean([within_group_spread(f["gates"], f["link_counts"], fine_parent) for f in fits]))


def gate_table(fits, data):
    """Fold-averaged gates of fine labels inside coarse groups with several observed members."""
    labels, coarse = data.inv.fine_labels, data.inv.coarse_labels
    counts = np.sum([f["link_counts"] for f in fits], axis=0)
    gates = np.array([f["gates"] for f in fits])
    table = {}
    for group in np.unique(data.fine_parent[data.fine_parent >= 0]):
        members = [i for i in np.flatnonzero(data.fine_parent == group) if counts[i] > 0]
        if len(members) < 2:
            continue
        table[coarse[group]] = {
            labels[i]: {"mean_gate": float(gates[:, i].mean()), "min_gate": float(gates[:, i].min()),
                        "max_gate": float(gates[:, i].max()), "training_links_over_folds": int(counts[i])}
            for i in members
        }
    return table


def summarize_fit(truth, probability):
    prediction = (probability >= 0.5).astype(np.int16)
    return {"log_loss": float(log_loss(truth, probability).mean()), **metrics(truth, prediction)}


def run_scramble(args):
    start = perf_counter()
    data = load_inputs(args.corpus, args.prepared, args.trees, args.scores)
    run = gate_run(args.corpus)
    saved = published_predictions(args.gate_predictions, run["predictions_sha256"], data.metadata)
    inventories = args.inventories or list(INVENTORIES)
    report = {"protocol": "granularity-scramble-v1", "corpus": args.corpus, "permutations": args.permutations,
              "permutation_seed": [SEED, "inventory index", "permutation index"],
              "null": "fine labels shuffled among the links of each mapped coarse group, across the corpus",
              "regularization_c": REGULARIZATION_C, "input_hashes": {**data.hashes,
              "gate_predictions.jsonl": run["predictions_sha256"]},
              "inventories": {}}
    args.output.mkdir(parents=True)
    with (args.output / "permutations.jsonl").open("x", encoding="utf-8") as stream:
        for inventory_index, inventory in enumerate(INVENTORIES):
            if inventory not in inventories:
                continue
            gate = GateData(data, inventory)
            inv = gate.inv
            reference, reproduced = {}, {}
            for mode, labels, count in (("unlabeled", np.zeros_like(inv.fine_ids), 1),
                                        ("coarse", inv.coarse_ids, len(inv.coarse_labels)),
                                        ("fine", inv.fine_ids, len(inv.fine_labels))):
                probability, fits = gate.cross_validate(labels, count, use_labels=mode != "unlabeled")
                difference = float(np.max(np.abs(probability - saved[f"{inventory}/{mode}"])))
                if difference > REPRODUCTION_TOLERANCE:
                    raise ValueError(f"Refit {inventory}/{mode} differs from the published predictions by {difference}")
                reproduced[mode] = difference
                reference[mode] = (probability, fits)
            truth = data.truth
            fine_probability, fine_fits = reference["fine"]
            observed = {"log_loss": float(log_loss(truth, fine_probability).mean()),
                        "macro_f1": metrics(truth, (fine_probability >= 0.5).astype(np.int16))["macro_f1"],
                        "gate_spread": mean_spread(fine_fits, gate.fine_parent)}
            null = {key: [] for key in observed}
            for index in range(args.permutations):
                rng = np.random.default_rng([SEED, inventory_index, index])
                labels = gate.scramble(rng)
                if (Counter(labels.tolist()) != Counter(inv.fine_ids.tolist())
                        or not np.array_equal(gate.fine_parent[labels], inv.coarse_ids)):
                    raise ValueError("Scrambling changed label counts or coarse groups")
                probability, fits = gate.cross_validate(labels, len(inv.fine_labels))
                row = {"log_loss": float(log_loss(truth, probability).mean()),
                       "macro_f1": metrics(truth, (probability >= 0.5).astype(np.int16))["macro_f1"],
                       "gate_spread": mean_spread(fits, gate.fine_parent)}
                for key, value in row.items():
                    null[key].append(value)
                stream.write(json.dumps({"inventory": inventory, "permutation": index,
                                         "labels_sha256": sha256(labels.astype(np.int64).tobytes()).hexdigest(),
                                         **row}, sort_keys=True) + "\n")
                stream.flush()
            report["inventories"][inventory] = {
                "fine_labels": len(inv.fine_labels), "coarse_labels": len(inv.coarse_labels),
                "links": int(len(inv.fine_ids)),
                "coarse_groups_with_several_observed_fine_labels": int(sum(
                    len(set(inv.fine_ids[inv.coarse_ids == g].tolist())) > 1 for g in np.unique(inv.coarse_ids))),
                "max_abs_difference_from_published": reproduced,
                "real": {mode: summarize_fit(truth, probability) for mode, (probability, _) in reference.items()},
                "real_fine_gate_spread": observed["gate_spread"],
                "scrambled": {key: {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
                                    "quantiles_2.5_50_97.5": np.quantile(values, [0.025, 0.5, 0.975]).tolist()}
                              for key, values in null.items()},
                "permutation_p": {
                    "fine_log_loss_lower_than_scrambled": permutation_p(observed["log_loss"], null["log_loss"],
                                                                        lower_is_better=True),
                    "fine_macro_f1_higher_than_scrambled": permutation_p(observed["macro_f1"], null["macro_f1"],
                                                                         lower_is_better=False),
                    "gate_spread_higher_than_scrambled": permutation_p(observed["gate_spread"], null["gate_spread"],
                                                                       lower_is_better=False),
                },
                "fine_gates_within_coarse_groups": gate_table(fine_fits, gate),
            }
            print(f"{args.corpus} {inventory}: {args.permutations} permutations after "
                  f"{perf_counter() - start:.0f}s", flush=True)
    report["predictions_sha256"] = digest(args.output / "permutations.jsonl")
    report["source_hashes"] = source_hashes()
    report["runtime"] = runtime()
    report["seconds"] = perf_counter() - start
    write_json(args.output / "report.json", report)


# Log-loss reanalysis -----------------------------------------------------------------

def pooling_probabilities(data, scores_path, corpus):
    """Refit relation pooling and require the published macro-F1 for every condition."""
    config = json.loads(CONFIGS[corpus].read_text())
    published = json.loads(PUBLISHED["comparison"].read_text())["macro_f1_and_accuracy"]
    by_id = {row["document_id"]: row for row in data.metadata}
    uniform = uniform_predictions(mean_edu_scores(scores_path), by_id, data.splits, config)
    evaluation = Evaluation(by_id, data.splits, config, uniform)
    score_rows = read_jsonl(scores_path)
    index = {row["document_id"]: i for i, row in enumerate(data.metadata)}
    probabilities = {}
    for inventory, (path, table, mapping) in tree_inputs(data.args, data.audit, dict(data.hashes)).items():
        for feature, features in feature_sets(score_rows, read_jsonl(path), table, mapping, "fine_checkpoint"):
            rows, outcome, _ = evaluation.fit(features, inventory)
            target = published[f"{inventory}/{feature}"][CORPORA[corpus]]["macro_f1"]
            if abs(outcome["macro_f1"] - target) > 1e-12:
                raise ValueError(f"Pooling refit {inventory}/{feature} differs from the published macro-F1")
            values = np.full(len(index), np.nan)
            for row in rows:
                values[index[row["document_id"]]] = row["probability_positive"]
            probabilities[f"pooling/{inventory}/{feature}"] = values
    return probabilities, digest(CONFIGS[corpus])


def run_loss(args):
    data = load_inputs(args.corpus, args.prepared, args.trees, args.scores)
    truth, metadata = data.truth, data.metadata
    run = gate_run(args.corpus)
    hashes = {**data.hashes, "gate_predictions.jsonl": run["predictions_sha256"]}
    probabilities = {f"gate/{key}": value for key, value in published_predictions(
        args.gate_predictions, run["predictions_sha256"], metadata).items()}
    pooling, config_hash = pooling_probabilities(data, args.scores, args.corpus)
    probabilities.update(pooling)
    hashes[CONFIGS[args.corpus].name] = config_hash
    contrasts = []
    for inventory in INVENTORIES:
        for method in ("relation-text-order", "relation-nucleus-first"):
            contrasts.append(("fine_vs_coarse", f"pooling/{inventory}/{method}/fine",
                              f"pooling/{inventory}/{method}/coarse"))
        contrasts += [("fine_vs_coarse", f"gate/{inventory}/fine", f"gate/{inventory}/coarse"),
                      ("labels_vs_unlabeled", f"gate/{inventory}/fine", f"gate/{inventory}/unlabeled"),
                      ("labels_vs_unlabeled", f"gate/{inventory}/coarse", f"gate/{inventory}/unlabeled")]
    if args.corpus == "amazon":
        recursive = json.loads(PUBLISHED["recursive"].read_text())
        probabilities.update({f"recursive/{key}": value for key, value in published_predictions(
            args.recursive_predictions, recursive["predictions_sha256"], metadata).items()})
        hashes["recursive_predictions.jsonl"] = recursive["predictions_sha256"]
        for inventory in INVENTORIES:
            contrasts += [("fine_vs_coarse", f"recursive/{inventory}/fine", f"recursive/{inventory}/coarse"),
                          ("labels_vs_unlabeled", f"recursive/{inventory}/fine", f"recursive/{inventory}/shared"),
                          ("labels_vs_unlabeled", f"recursive/{inventory}/coarse", f"recursive/{inventory}/shared")]

    weights = bootstrap_weights(metadata).astype(np.float64)
    losses = {name: log_loss(truth, value) for name, value in probabilities.items()}
    draws = {name: resampled_means(weights, loss) for name, loss in losses.items()}
    conditions = {name: summarize_fit(truth, value) for name, value in sorted(probabilities.items())}
    results = {}
    for family, left, right in contrasts:
        difference = draws[left] - draws[right]
        results.setdefault(family, {})[f"{left} - {right}"] = {
            "log_loss_difference": float(losses[left].mean() - losses[right].mean()),
            "interval_95": np.quantile(difference, [0.025, 0.975]).tolist(),
            "bootstrap_p": bootstrap_p(difference),
            "macro_f1_difference": conditions[left]["macro_f1"] - conditions[right]["macro_f1"],
        }
    for family in results.values():
        for key, adjusted in holm({key: row["bootstrap_p"] for key, row in family.items()}).items():
            family[key]["holm_p"] = adjusted
    write_json(args.output, {
        "protocol": "granularity-loss-v1", "corpus": args.corpus,
        "metric": "held-out binary log-loss; negative differences favour the left condition",
        "resampling": {"draws": ITERATIONS, "seed": SEED, "unit": "review", "paired": True,
                       "strata": ["outer_fold", "domain", "label"]},
        "families": "Holm adjustment within each family and corpus",
        "input_hashes": hashes, "source_hashes": source_hashes(), "runtime": runtime(),
        "conditions": conditions, "contrasts": results,
    })


# Provenance -------------------------------------------------------------------

def source_hashes():
    names = ("granularity_diagnostics.py", "dependency_sentiment.py", "relation_gate.py", "relation_pooling.py",
             "nuclearity.py", "stock_pooling.py", "edu_baseline.py", "baseline.py", "relation_stability.py",
             "uv.lock")
    return {name: digest(HERE / name) for name in names}


def runtime():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__}


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("loss", "scramble"):
        command = sub.add_parser(name)
        command.add_argument("--corpus", choices=CORPORA, required=True)
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--trees", type=Path, required=True)
        command.add_argument("--scores", type=Path, required=True, help="Frozen CardiffNLP scores.jsonl")
        command.add_argument("--gate-predictions", type=Path, required=True,
                             help="Published CardiffNLP relation-gate predictions.jsonl")
        command.add_argument("--output", type=Path, required=True)
    sub.choices["loss"].add_argument("--recursive-predictions", type=Path,
                                     help="Published Amazon recursive predictions.jsonl")
    scramble = sub.choices["scramble"]
    scramble.add_argument("--permutations", type=int, default=200)
    scramble.add_argument("--inventories", nargs="+", choices=INVENTORIES)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "loss":
        if (args.corpus == "amazon") != (args.recursive_predictions is not None):
            cli.error("--recursive-predictions is required for Amazon and unavailable for SFU")
        run_loss(args)
    else:
        run_scramble(args)


if __name__ == "__main__":
    main()
