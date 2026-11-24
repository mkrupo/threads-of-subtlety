"""Replay the original logistic-regression sentiment features with stock coarse trees."""

import argparse
from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path

import numpy as np
import scipy
import sklearn

from .baseline import HERE, metrics, read_jsonl
from .dependency_sentiment import input_files, tree_inputs, vectors_for
from .edu_baseline import fit_review_lr, mean_edu_scores
from .nuclearity import evaluate, pool_nuclearity
from .parser_pilot import digest, write_json
from .relation_pooling import coarsen, feature_rows, pool_relations


@dataclass(frozen=True)
class Evaluation:
    metadata: dict
    splits: dict
    config: dict
    uniform: dict

    def fit(self, features, inventory):
        rows, report = evaluate(features, self.metadata, self.splits, self.config, self.uniform, inventory)
        return rows, report["pooled_out_of_fold"], [fold["selected_C"] for fold in report["folds"]]


def uniform_predictions(features, metadata, splits, config):
    predictions = {}
    for fold in splits["outer_folds"]:
        train, test = sorted(fold["train_ids"]), sorted(fold["test_ids"])
        search = fit_review_lr(
            [features[doc] for doc in train], [metadata[doc]["label"] for doc in train],
            [fold["inner_folds"][doc] for doc in train], config,
        )
        scores = search.predict_proba([features[doc] for doc in test])[:, list(search.classes_).index(1)]
        for doc, score in zip(test, scores, strict=True):
            row = metadata[doc]
            if doc in predictions or row["outer_fold"] != fold["fold"]:
                raise ValueError("Uniform prediction differs from the saved outer fold")
            predictions[doc] = {"fold": fold["fold"], "label": row["label"],
                                "prediction": int(score >= config["decision_threshold"])}
    if predictions.keys() != metadata.keys():
        raise ValueError("Expected a uniform prediction for every review")
    return predictions


def feature_sets(score_rows, trees, table, mapping, source):
    nuclearity, _ = pool_nuclearity(score_rows, trees)
    yield "nuclearity", nuclearity
    alphabet = sorted({label.rsplit("_", 1)[0].lower() for label in table})
    coarse = sorted({mapping[label] for label in alphabet if label in mapping})
    for method, nucleus_first in (("relation-text-order", False), ("relation-nucleus-first", True)):
        pooled = pool_relations(score_rows, trees, nucleus_first=nucleus_first)
        if source == "fine_checkpoint":
            yield f"{method}/fine", feature_rows(pooled, nuclearity, alphabet)
            pooled = coarsen(pooled, mapping)
        yield f"{method}/coarse", feature_rows(pooled, nuclearity, coarse)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    audit, representation, scenes, hashes = input_files(args)
    vectors_for(args, scenes, representation)
    score_rows = read_jsonl(args.representation)
    stock = tree_inputs(args, audit, hashes)
    fine = tree_inputs(argparse.Namespace(trees=args.trees, stock=None), audit, hashes)
    metadata = {row["document_id"]: row for row in read_jsonl(args.prepared / "manifest.jsonl")}
    splits = json.loads((args.prepared / "splits.json").read_text())
    amazon_results = HERE / "replication/results/amazon"
    config_path = HERE / "baseline_config.json" if args.corpus == "sfu" else amazon_results / "baseline-config.json"
    config = json.loads(config_path.read_text())
    original_path = amazon_results / "comparison.json"
    original = json.loads(original_path.read_text())["macro_f1_and_accuracy"]
    corpus = {"sfu": "SFU", "amazon": "Amazon"}[args.corpus]
    uniform = uniform_predictions(mean_edu_scores(args.representation), metadata, splits, config)
    evaluation = Evaluation(metadata, splits, config, uniform)
    predictions = list(uniform.values())
    no_rst = metrics([row["label"] for row in predictions], [row["prediction"] for row in predictions])
    if not isclose(no_rst["macro_f1"], original["edu_mean"][corpus]["macro_f1"], abs_tol=1e-12):
        raise ValueError("The refit does not reproduce the original equal-EDU result")

    args.output.mkdir(parents=True)
    path = args.output / "predictions.jsonl"
    results, regularization = {}, {}
    with path.open("x", encoding="utf-8") as stream:
        for source, inputs in (("fine_checkpoint", fine), ("stock", stock)):
            for name in stock:
                trees_path, table, mapping = inputs[name]
                for feature, features in feature_sets(score_rows, read_jsonl(trees_path), table, mapping, source):
                    if features.keys() != metadata.keys():
                        raise ValueError(f"Pooling feature coverage differs from the review set for {name}")
                    rows, outcome, selected = evaluation.fit(features, name)
                    if source == "fine_checkpoint":
                        target = original[f"{name}/{feature}"][corpus]["macro_f1"]
                        if not isclose(outcome["macro_f1"], target, abs_tol=1e-12):
                            raise ValueError(f"Original pooling result changed for {corpus}/{name}/{feature}")
                    key = f"{source}/{name}/{feature}"
                    results[key], regularization[key] = outcome, selected
                    for row in rows:
                        stream.write(json.dumps({
                            "document_id": row["document_id"], "outer_fold": row["fold"],
                            "label": row["label"], "condition": key,
                            "prediction": row["prediction"], "probability": row["probability_positive"],
                        }) + "\n")
                print(f"{corpus}: fitted {source} {name}", flush=True)
    write_json(args.output / "report.json", {
        "protocol": "stock-pooling-v1", "corpus": args.corpus, "input_hashes": {
            **hashes, config_path.name: digest(config_path), "published-comparison.json": digest(original_path),
        },
        "source_hashes": {name: digest(HERE / name) for name in (
            "stock_pooling.py", "dependency_sentiment.py", "edu_baseline.py", "nuclearity.py", "relation_pooling.py",
        )},
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "edu_mean": no_rst, "results": results, "selected_C_by_fold": regularization,
        "predictions_sha256": digest(path),
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.set_defaults(arm="cardiff")
    cli.add_argument("--corpus", choices=("sfu", "amazon"), required=True)
    cli.add_argument("--prepared", type=Path, required=True)
    cli.add_argument("--trees", type=Path, required=True)
    cli.add_argument("--stock", type=Path, required=True)
    cli.add_argument("--representation", type=Path, required=True, help="Frozen CardiffNLP scores.jsonl")
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
