"""Fit a review classifier on the mean of the frozen SFU EDU scores."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import (
    HERE, INPUT_FILES, digest, read_jsonl, summarize_predictions, validate_prepared, write_json,
)


PROTOCOL = "sfu-edu-mean-v1"
CONTROL = "edu_mean_lr"
RESULTS = HERE / "results"


def mean_edu_scores(scores_path: Path) -> dict[str, np.ndarray]:
    # Pool all EDUs in a review, giving each EDU the same weight across scenes.
    by_review = defaultdict(list)
    for row in read_jsonl(scores_path):
        by_review[row["document_id"]].extend(row["probabilities"])
    return {name: np.mean(scores, axis=0, dtype=np.float64) for name, scores in by_review.items()}


def fit_review_lr(features, labels, inner_folds, config):
    options = {**config["logistic_regression"], "random_state": config["seed"]}
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(**options))
    search = GridSearchCV(
        pipeline, {"logisticregression__C": config["regularization_C"]},
        scoring=make_scorer(f1_score, labels=[0, 1], average="macro", zero_division=0),
        cv=PredefinedSplit(inner_folds), n_jobs=1, error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(features, labels)
    return search


def run(prepared: Path, scores: Path, output: Path) -> dict:
    reference = json.loads((RESULTS / "sfu-text-controls-v1.json").read_text())
    scorer = json.loads((RESULTS / "sfu-edu-scores-v1.json").read_text())
    config_path = HERE / "baseline_config.json"
    inputs = {name: digest(prepared / name) for name in INPUT_FILES}
    inputs["baseline_config.json"] = digest(config_path)
    if inputs != reference["provenance"]["input_hashes"]:
        raise ValueError("Use the corpus, folds and LR configuration from the frozen text controls")
    inputs["scores.jsonl"] = digest(scores)
    if inputs["scores.jsonl"] != scorer["scores_sha256"]:
        raise ValueError("Use the verified frozen EDU score cache")
    config = json.loads(config_path.read_text())
    records, splits = validate_prepared(prepared, config)
    by_id = {row["document_id"]: row for row in records}
    means = mean_edu_scores(scores)
    if means.keys() != by_id.keys():
        raise ValueError("EDU score coverage differs from the prepared reviews")
    output.mkdir(parents=True)

    predictions, selections = [], []
    for fold in sorted(splits["outer_folds"], key=lambda f: f["fold"]):
        train, test = sorted(fold["train_ids"]), sorted(fold["test_ids"])
        search = fit_review_lr(
            [means[name] for name in train], [by_id[name]["label"] for name in train],
            [fold["inner_folds"][name] for name in train], config,
        )
        probabilities = search.predict_proba([means[name] for name in test])
        positive = probabilities[:, list(search.classes_).index(1)]
        if not np.isfinite(positive).all():
            raise ValueError("Non-finite review probabilities")
        for name, probability in zip(test, positive, strict=True):
            predictions.append({
                "document_id": name, "fold": fold["fold"], "domain": by_id[name]["domain"],
                "label": by_id[name]["label"], "control": CONTROL,
                "probability_positive": float(probability),
                "prediction": int(probability >= config["decision_threshold"]),
            })
        selections.append({
            "outer_fold": fold["fold"], "selected_C": search.best_params_["logisticregression__C"],
            "mean_inner_macro_f1_by_C": dict(zip(
                map(str, config["regularization_C"]),
                search.cv_results_["mean_test_score"].tolist(), strict=True,
            )),
        })

    if Counter(row["document_id"] for row in predictions) != Counter(by_id.keys()):
        raise ValueError("Expected exactly one outer held-out prediction per review")
    predictions_path = output / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    evaluation = {**config, "protocol": PROTOCOL, "controls": [CONTROL],
                  "interpretation": "Equal-weight EDU sentiment baseline without RST features."}
    report = {
        "features": {"columns": scorer["labels"], "pooling": "arithmetic mean across all EDUs in the review",
                     "scaling": "standard scaler fitted within each fitting partition"},
        "primary_metric": config["primary_metric"],
        "input_hashes": inputs,
        "source_hashes": {name: digest(HERE / name) for name in ("edu_baseline.py", "baseline.py", "uv.lock")},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "predictions_sha256": digest(predictions_path),
        "selection": selections,
        "summary": summarize_predictions(predictions, evaluation),
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.prepared_dir, args.scores, args.output_dir)
    print(json.dumps(report["summary"]["controls"][CONTROL]["pooled_out_of_fold"], indent=2))


if __name__ == "__main__":
    main()
