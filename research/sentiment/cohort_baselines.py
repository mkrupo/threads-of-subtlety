"""Place the RST sentiment models among simple and neural baselines on a prepared cohort.

Added after the confirmation analysis. These comparisons describe where the
RST models land; they are not part of the pre-registered fine/coarse test.

``controls``  Fit the existing text controls (class prior, log length, word and
              word-pair TF-IDF) with baseline.py and the frozen Amazon settings,
              in the cohort's own nested folds.
``compare``   Score every baseline and the relation-gate models on the same
              held-out reviews, with paired bootstrap intervals.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy
import sklearn

from .baseline import classify, fit_control, metrics, read_jsonl
from .confirm_granularity import contrast_table
from .granularity_diagnostics import CONFIGS, log_loss, resampled_means
from .parser_pilot import INVENTORIES, digest, write_json
from .relation_stability import ITERATIONS, SEED, bootstrap_weights

CONTROLS = ("class_prior", "length_only", "tfidf_lr")
HERE = Path(__file__).resolve().parent


def load_prepared(prepared):
    provenance = json.loads((prepared / "provenance.json").read_text())
    for name in ("reviews.jsonl", "manifest.jsonl", "splits.json"):
        if digest(prepared / name) != provenance["outputs_sha256"][name]:
            raise ValueError(f"{name} differs from the cohort provenance")
    reviews = {row["document_id"]: row for row in read_jsonl(prepared / "reviews.jsonl")}
    metadata = sorted(read_jsonl(prepared / "manifest.jsonl"), key=lambda row: row["document_id"])
    return reviews, metadata, json.loads((prepared / "splits.json").read_text()), provenance["outputs_sha256"]


def run_controls(args):
    if digest(args.config) != digest(CONFIGS["amazon"]):
        raise ValueError("Use the frozen Amazon text-control configuration")
    config = json.loads(args.config.read_text())
    reviews, metadata, splits, hashes = load_prepared(args.prepared)
    rows, selections = [], []
    for fold in splits["outer_folds"]:
        train_ids, test_ids = sorted(fold["train_ids"]), sorted(fold["test_ids"])
        train = [reviews[doc] for doc in train_ids]
        test = [reviews[doc] for doc in test_ids]
        inner = [fold["inner_folds"][doc] for doc in train_ids]
        for kind in CONTROLS:
            estimator, selection = fit_control(kind, train, inner, config)
            positive, _ = classify(estimator, kind, test, config)
            selections.append({"fold": fold["fold"], "control": kind, **selection})
            rows.extend({"document_id": doc, "outer_fold": fold["fold"], "label": reviews[doc]["label"],
                         "condition": kind, "probability": float(p)} for doc, p in zip(test_ids, positive))
        print(f"controls: fold {fold['fold']} done", flush=True)
    args.output.mkdir(parents=True)
    path = args.output / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda r: (r["condition"], r["document_id"])):
            stream.write(json.dumps(row) + "\n")
    write_json(args.output / "report.json", {
        "protocol": "cohort-text-controls-v1", "controls": list(CONTROLS), "config_sha256": digest(args.config),
        "input_hashes": dict(hashes), "selections": selections, "predictions_sha256": digest(path),
        "source_hashes": {name: digest(HERE / name) for name in ("cohort_baselines.py", "baseline.py", "uv.lock")},
        "runtime": runtime(),
    })


def read_predictions(path, report_path, metadata):
    report = json.loads(report_path.read_text())
    if digest(path) != report["predictions_sha256"]:
        raise ValueError(f"{path} differs from its report")
    index = {row["document_id"]: i for i, row in enumerate(metadata)}
    values = {}
    for row in read_jsonl(path):
        target = values.setdefault(row["condition"], np.full(len(metadata), np.nan))
        i = index[row["document_id"]]
        if row["label"] != metadata[i]["label"] or not np.isnan(target[i]):
            raise ValueError(f"Label mismatch or repeated review in {path}")
        target[i] = row["probability"]
    if any(np.isnan(v).any() for v in values.values()):
        raise ValueError(f"Missing held-out predictions in {path}")
    return values, report["predictions_sha256"]


def run_compare(args):
    _, metadata, _, hashes = load_prepared(args.prepared)
    truth = np.array([row["label"] for row in metadata], dtype=np.int16)
    probabilities, inputs = {}, dict(hashes)
    for name, (path, report) in {"gate": (args.gate_predictions, args.gate_report),
                                 "controls": (args.controls_predictions, args.controls_report),
                                 "roberta": (args.roberta_predictions, args.roberta_report)}.items():
        values, checksum = read_predictions(path, report, metadata)
        inputs[f"{name}_predictions.jsonl"] = checksum
        probabilities.update(values if name != "gate" else {f"gate/{k}": v for k, v in values.items()})
    weights = bootstrap_weights(metadata).astype(np.float64)
    losses = {name: log_loss(truth, value) for name, value in probabilities.items()}
    draws = {name: resampled_means(weights, loss) for name, loss in losses.items()}
    conditions = {}
    for name, value in sorted(probabilities.items()):
        prediction = (value >= 0.5).astype(np.int16)
        if name == "class_prior":
            prediction = np.zeros_like(truth)  # argmax tie selects class 0, as in baseline.classify
        conditions[name] = {"log_loss": float(losses[name].mean()), **metrics(truth, prediction)}
    for inventory in INVENTORIES:
        conditions[f"gate/{inventory}/fine"]["macro_f1_interval_95"] = macro_f1_interval(
            weights, truth, probabilities[f"gate/{inventory}/fine"])
    for name in ("roberta-base", "tfidf_lr"):
        conditions[name]["macro_f1_interval_95"] = macro_f1_interval(weights, truth, probabilities[name])
    pairs = [(base, f"gate/{inv}/fine") for base in ("roberta-base", "tfidf_lr") for inv in INVENTORIES]
    pairs.append(("roberta-base", "tfidf_lr"))
    write_json(args.output, {
        "protocol": "cohort-baseline-comparison-v1", "reviews": len(truth),
        "note": "Added after the pre-registered confirmation; descriptive, not confirmatory.",
        "uninformative_reference": {"probability": 0.5, "log_loss": float(np.log(2)),
                                    "expected_accuracy_random_guess": 0.5,
                                    "expected_macro_f1_random_guess": 0.5,
                                    "constant_class_macro_f1": 1 / 3},
        "resampling": {"draws": ITERATIONS, "seed": SEED, "unit": "review", "paired": True,
                       "strata": ["outer_fold", "domain", "label"]},
        "input_hashes": inputs, "conditions": conditions,
        "contrasts": contrast_table(pairs, losses, draws, conditions),
        "source_hashes": {name: digest(HERE / name) for name in ("cohort_baselines.py", "confirm_granularity.py",
                                                                  "granularity_diagnostics.py", "uv.lock")},
        "runtime": runtime(),
    })


def macro_f1_interval(weights, truth, probability):
    from .relation_stability import macro_f1
    return np.quantile(macro_f1(weights, truth, (probability >= 0.5).astype(np.int16)), [0.025, 0.975]).tolist()


def runtime():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__}


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="command", required=True)
    controls = sub.add_parser("controls")
    controls.add_argument("--config", type=Path, required=True, help="Frozen Amazon baseline-config.json")
    compare = sub.add_parser("compare")
    for flag in ("gate", "controls", "roberta"):
        compare.add_argument(f"--{flag}-predictions", type=Path, required=True)
        compare.add_argument(f"--{flag}-report", type=Path, required=True)
    for command in (controls, compare):
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run_controls(args) if args.command == "controls" else run_compare(args)


if __name__ == "__main__":
    main()
