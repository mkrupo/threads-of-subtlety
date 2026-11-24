"""Verify held-out predictions and compare stock versus mapped-coarse sentiment."""

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import HERE, metrics, read_jsonl
from .parser_pilot import digest, write_json
from .relation_stability import ITERATIONS, SEED, bootstrap_weights, macro_f1


def predictions(run, metadata, truth):
    report = json.loads((run / "report.json").read_text())
    path = run / "predictions.jsonl"
    if digest(path) != report["predictions_sha256"]:
        raise ValueError(f"Prediction hash mismatch in {run}")
    indices = {row["document_id"]: i for i, row in enumerate(metadata)}
    guesses = {name: np.full(len(truth), -1, dtype=np.int16) for name in report["results"]}
    for row in read_jsonl(path):
        i = indices[row["document_id"]]
        target = guesses[row["condition"]]
        if (target[i] != -1 or row["outer_fold"] != metadata[i]["outer_fold"] or row["label"] != truth[i]
                or not 0 <= row["probability"] <= 1 or row["prediction"] != int(row["probability"] >= 0.5)):
            raise ValueError(f"Invalid held-out prediction in {run}: {row['document_id']}")
        target[i] = row["prediction"]
    for name, pred in guesses.items():
        if (pred < 0).any() or metrics(truth, pred) != report["results"][name]:
            raise ValueError(f"Cannot reproduce reported metrics for {run}/{name}")
    return report, guesses


def compare(stock, reference, stock_metrics, reference_metrics, truth, weights):
    differences = macro_f1(weights, truth, stock) - macro_f1(weights, truth, reference)
    return {
        "mapped_macro_f1": reference_metrics["macro_f1"],
        "stock_macro_f1": stock_metrics["macro_f1"],
        "stock_minus_mapped": stock_metrics["macro_f1"] - reference_metrics["macro_f1"],
        "interval_95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "changed_predictions": int(np.count_nonzero(stock != reference)),
    }


def summarize(corpus, prepared, runs, published):
    manifest = prepared / "manifest.jsonl"
    metadata = sorted(read_jsonl(manifest), key=lambda row: row["document_id"])
    truth = np.array([row["label"] for row in metadata], dtype=np.int16)
    weights = bootstrap_weights(metadata)
    outcomes = {}

    methods = ("pooling", "cardiff", "mdeberta", "recursive") if corpus == "amazon" else ("pooling", "cardiff", "mdeberta")
    for method in methods:
        original_hash = None
        run = runs / f"{corpus}-{method}"
        stock, new = predictions(run, metadata, truth)
        if method == "pooling":
            old, prior = stock, new
            pairs = [(name, "fine_checkpoint/" + name.removeprefix("stock/"))
                     for name in stock["results"] if name.startswith("stock/")]
        else:
            mapped_run = runs / f"mapped-{corpus}-{method}"
            old, prior = predictions(mapped_run, metadata, truth)
            original_hash = published[corpus, method]["predictions_sha256"]
            if old["predictions_sha256"] != original_hash:
                raise ValueError(f"Earlier {corpus}/{method} predictions failed to reproduce")
            if method in ("cardiff", "mdeberta") and not np.array_equal(new["edu_mean"], prior["edu_mean"]):
                raise ValueError("The comparator without a tree should reproduce across parser choices")
            pairs = [(name, name) for name in stock["results"] if name != "edu_mean"]
        for report in (old, stock):
            if (report["input_hashes"]["manifest.jsonl"] != digest(manifest)
                    or report["input_hashes"]["splits.json"] != digest(prepared / "splits.json")):
                raise ValueError("The runs differ from the prepared reviews or folds")
        deltas = {name.removeprefix("stock/"): compare(
            new[name], prior[previous], stock["results"][name], old["results"][previous], truth, weights,
        ) for name, previous in pairs}
        outcomes[method] = {
            "conditions": deltas,
            "input_hashes": stock["input_hashes"],
            "source_hashes": stock.get("source_hashes", {"recursive_sentiment.py": stock.get("source_sha256")}),
            "stock_predictions_sha256": stock["predictions_sha256"],
            "mapped_predictions_sha256": old["predictions_sha256"],
            "published_predictions_sha256": original_hash,
            "stock_internal_comparisons": stock.get("paired"),
        }
    return {"reviews": len(truth), "outcomes": outcomes}


def run(args):
    deps = json.loads((HERE / "results/sentiment3-dependency-v1.json").read_text())
    published = {(row["corpus"], row["arm"]): row for row in deps["runs"]}
    published["amazon", "recursive"] = json.loads((HERE / "results/amazon-recursive-v1.json").read_text())
    write_json(args.output, {
        "protocol": "stock-coarse-sentiment-v1",
        "resampling": {"draws": ITERATIONS, "seed": SEED, "unit": "review", "paired": True,
                       "strata": ["outer_fold", "domain", "label"]},
        "source_sha256": digest(Path(__file__)),
        "corpora": {corpus: summarize(corpus, prepared, args.runs, published)
                    for corpus, prepared in (("sfu", args.sfu_prepared), ("amazon", args.amazon_prepared))},
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, required=True)
    cli.add_argument("--sfu-prepared", type=Path, required=True)
    cli.add_argument("--amazon-prepared", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
