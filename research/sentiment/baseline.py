"""Fit locked text controls with nested, document-level SFU cross-validation.

This module does not load a discourse parser or claim an RST comparison.
Run from the repository root with the locked sentiment environment.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import warnings

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .corpus import build_splits


HERE = Path(__file__).resolve().parent
INPUT_FILES = ("reviews.jsonl", "manifest.jsonl", "splits.json", "provenance.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    # splitlines() also splits Unicode separators that can occur inside review text.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metrics(y, prediction) -> dict:
    return {
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro",
                                    zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "negative_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "positive_f1": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "documents": len(y),
    }


def validate_inputs(records: list[dict], splits: dict, config: dict) -> dict[str, dict]:
    by_id = {row["document_id"]: row for row in records}
    if len(by_id) != len(records) or not records:
        raise ValueError("Duplicate document IDs or empty corpus")
    if len({row["text_sha256"] for row in records}) != len(records):
        raise ValueError("Exact text duplicates require an explicit group-aware protocol")
    if any(row["label"] not in (0, 1) or not row["text"].strip() for row in records):
        raise ValueError("Invalid sentiment label or empty text")
    all_ids = set(by_id)
    all_strata = Counter((row["domain"], row["label"]) for row in records)
    if any(count % config["outer_folds"] for count in all_strata.values()):
        raise ValueError("Corpus strata cannot be divided into equal outer folds")
    expected_test = Counter({key: count // config["outer_folds"]
                             for key, count in all_strata.items()})
    folds = splits["outer_folds"]
    if sorted(fold["fold"] for fold in folds) != list(range(config["outer_folds"])):
        raise ValueError("Outer fold IDs do not match the frozen configuration")
    tested = Counter()
    for fold in folds:
        train_ids, test_ids = fold["train_ids"], fold["test_ids"]
        train, test = set(train_ids), set(test_ids)
        if len(train) != len(train_ids) or len(test) != len(test_ids):
            raise ValueError("Repeated IDs within a fold")
        if train & test or train | test != all_ids:
            raise ValueError("Outer train/test overlap or missing/extra IDs")
        test_strata = Counter((by_id[name]["domain"], by_id[name]["label"])
                              for name in test_ids)
        if test_strata != expected_test:
            raise ValueError("Outer test fold violates domain-by-label stratification")
        train_strata = all_strata - expected_test
        if any(count % config["inner_folds"] for count in train_strata.values()):
            raise ValueError("Training strata cannot be divided into equal inner folds")
        expected_validation = Counter({key: count // config["inner_folds"]
                                       for key, count in train_strata.items()})
        inner = fold["inner_folds"]
        if set(inner) != train:
            raise ValueError("Inner validation IDs must be exactly the outer training IDs")
        if set(inner.values()) != set(range(config["inner_folds"])):
            raise ValueError("Inner fold IDs do not match the frozen configuration")
        for inner_fold in range(config["inner_folds"]):
            validation = [name for name in train_ids if inner[name] == inner_fold]
            fitting = [name for name in train_ids if inner[name] != inner_fold]
            actual_validation = Counter((by_id[name]["domain"], by_id[name]["label"])
                                        for name in validation)
            if actual_validation != expected_validation:
                raise ValueError("Inner validation violates domain-by-label stratification")
            for names in (validation, fitting, test_ids):
                if {by_id[name]["label"] for name in names} != {0, 1}:
                    raise ValueError("A fitting/validation/test split lacks a sentiment class")
        tested.update(test_ids)
    if set(tested) != all_ids or any(count != 1 for count in tested.values()):
        raise ValueError("Every review must have exactly one outer held-out prediction")
    if splits != build_splits(records, seed=config["seed"]):
        raise ValueError("Fold assignments differ from the declared seed and construction")
    return by_id


def validate_prepared(prepared: Path, config: dict) -> tuple[list[dict], dict]:
    records = read_jsonl(prepared / "reviews.jsonl")
    provenance = json.loads((prepared / "provenance.json").read_text())
    expected = config["expected_corpus"]
    if (provenance["archive_sha256"] != expected["archive_sha256"]
            or len(records) != expected["documents"]
            or provenance["document_count"] != len(records)
            or len({row["domain"] for row in records}) != expected["domains"]):
        raise ValueError("Prepared corpus does not match the frozen corpus/version")
    strata = Counter((row["domain"], row["label"]) for row in records)
    if len(strata) != expected["domains"] * 2 or set(strata.values()) != {expected["per_domain_label"]}:
        raise ValueError("Prepared domain/polarity quotas differ from the frozen protocol")
    if set(provenance["outputs_sha256"]) != set(INPUT_FILES) - {"provenance.json"}:
        raise ValueError("Preparation provenance lacks the exact required output inventory")
    for name, expected_hash in provenance["outputs_sha256"].items():
        if digest(prepared / name) != expected_hash:
            raise ValueError(f"Changed prepared artifact: {name}")
    for name, expected_hash in provenance["sources_sha256"].items():
        if digest(HERE / name) != expected_hash:
            raise ValueError(f"Preparation implementation changed: {name}")
    if provenance["split_policy"]["seed"] != config["seed"]:
        raise ValueError("Preparation and baseline seeds differ")
    splits = json.loads((prepared / "splits.json").read_text())
    validate_inputs(records, splits, config)
    folds = {name: fold["fold"] for fold in splits["outer_folds"] for name in fold["test_ids"]}
    for row in records:
        normalized = " ".join(row["text"].split())
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if (row["text_normalized"] != normalized or row["text_sha256"] != text_hash
                or row["document_id"] != f"sfu-{text_hash}"
                or row["whitespace_tokens"] != len(normalized.split())):
            raise ValueError("Review text, normalization, hash, length or ID disagree")
    expected_manifest = []
    for row in records:
        metadata = {key: value for key, value in row.items()
                    if key not in {"text", "text_normalized"}}
        metadata["outer_fold"] = folds[row["document_id"]]
        expected_manifest.append(metadata)
    if read_jsonl(prepared / "manifest.jsonl") != expected_manifest:
        raise ValueError("Prepared metadata manifest differs from review/fold records")
    return records, splits


def make_estimator(kind: str, config: dict):
    options = dict(config["logistic_regression"])
    options["random_state"] = config["seed"]
    if kind == "class_prior":
        return DummyClassifier(strategy="prior")
    if kind == "length_only":
        return Pipeline([("scale", StandardScaler()),
                         ("classifier", LogisticRegression(**options))])
    if kind == "tfidf_lr":
        vectorizer = dict(config["tfidf"])
        vectorizer["ngram_range"] = tuple(vectorizer["ngram_range"])
        return Pipeline([("tfidf", TfidfVectorizer(**vectorizer)),
                         ("classifier", LogisticRegression(**options))])
    raise ValueError(f"Unknown control: {kind}")


def feature_input(kind: str, records: list[dict]):
    if kind == "tfidf_lr":
        return [row["text"] for row in records]
    if kind == "length_only":
        return np.log1p(np.asarray([[len(row["text"].split())] for row in records],
                                  dtype=np.float64))
    if kind == "class_prior":
        return np.zeros((len(records), 1), dtype=np.float64)
    raise ValueError(kind)


def fit_control(kind: str, train_rows: list[dict], inner_folds: list[int], config: dict):
    x = feature_input(kind, train_rows)
    y = np.asarray([row["label"] for row in train_rows], dtype=np.int8)
    estimator = make_estimator(kind, config)
    if kind == "class_prior":
        estimator.fit(x, y)
        return estimator, {"selection": "training class prior; argmax tie selects class 0"}
    candidates = list(config["regularization_C"])
    if candidates != sorted(candidates) or not candidates:
        raise ValueError("C candidates must be nonempty and ordered from smaller to larger")
    scorer = make_scorer(f1_score, labels=[0, 1], average="macro", zero_division=0)
    search = GridSearchCV(
        estimator, {"classifier__C": candidates}, scoring=scorer,
        cv=PredefinedSplit(inner_folds), refit=True, n_jobs=1,
        error_score="raise", return_train_score=False,
    )
    # Each cloned Pipeline fits its own vocabulary/IDF/scaler inside inner fitting.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(x, y)
    return search.best_estimator_, {
        "selection": config["selection_metric"],
        "selected_C": float(search.best_params_["classifier__C"]),
        "candidates": [
            {"C": float(params["classifier__C"]),
             "mean_inner_macro_f1": float(mean),
             "std_inner_macro_f1": float(std)}
            for params, mean, std in zip(
                search.cv_results_["params"], search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"])
        ],
        "refit": "full outer training partition only",
    }


def classify(estimator, kind: str, rows: list[dict], config: dict):
    probabilities = estimator.predict_proba(feature_input(kind, rows))
    classes = list(estimator.classes_)
    positive = np.asarray(probabilities[:, classes.index(1)], dtype=np.float64)
    if not np.isfinite(positive).all():
        raise ValueError("Non-finite probabilities")
    if kind == "class_prior":
        prediction = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    else:
        prediction = (positive >= config["decision_threshold"]).astype(np.int8)
    return positive, prediction


def summarize_predictions(predictions: list[dict], config: dict) -> dict:
    result = {"protocol": config["protocol"], "label_names": config["label_names"],
              "interpretation": config["interpretation"], "controls": {}}
    for kind in config["controls"]:
        rows = [row for row in predictions if row["control"] == kind]
        result["controls"][kind] = {
            "pooled_out_of_fold": metrics([r["label"] for r in rows],
                                           [r["prediction"] for r in rows]),
            "per_fold": {
                str(fold): metrics([r["label"] for r in rows if r["fold"] == fold],
                                   [r["prediction"] for r in rows if r["fold"] == fold])
                for fold in range(config["outer_folds"])
            },
            "per_domain": {
                domain: metrics([r["label"] for r in rows if r["domain"] == domain],
                                [r["prediction"] for r in rows if r["domain"] == domain])
                for domain in sorted({r["domain"] for r in rows})
            },
        }
    return result


def run(prepared: Path, output: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    records, splits = validate_prepared(prepared, config)
    by_id = validate_inputs(records, splits, config)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}; choose a fresh run directory")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {name: digest(prepared / name) for name in INPUT_FILES}
    inputs["baseline_config.json"] = digest(config_path)
    write_json(output / "protocol.json", config)
    predictions, selections = [], []
    for fold in sorted(splits["outer_folds"], key=lambda f: f["fold"]):
        train_ids, test_ids = sorted(fold["train_ids"]), sorted(fold["test_ids"])
        train_rows, test_rows = [by_id[i] for i in train_ids], [by_id[i] for i in test_ids]
        inner = [fold["inner_folds"][i] for i in train_ids]
        for kind in config["controls"]:
            estimator, selection = fit_control(kind, train_rows, inner, config)
            positive, predicted = classify(estimator, kind, test_rows, config)
            selections.append({"outer_fold": fold["fold"], "control": kind, **selection})
            joblib.dump(estimator, output / f"fold-{fold['fold']}-{kind}.joblib")
            for row, probability, label in zip(test_rows, positive, predicted):
                predictions.append({"document_id": row["document_id"], "fold": fold["fold"],
                                    "domain": row["domain"], "label": row["label"],
                                    "control": kind, "probability_positive": float(probability),
                                    "prediction": int(label)})
            print(f"finished outer fold {fold['fold']}, {kind}", flush=True)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in predictions),
        encoding="utf-8")
    summary = summarize_predictions(predictions, config)
    write_json(output / "summary.json", summary)
    write_json(output / "selection.json", selections)
    source_files = ("baseline.py", "corpus.py", "prepare_corpus.py", "pyproject.toml", "uv.lock")
    source_hashes = {name: digest(HERE / name) for name in source_files}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "schema_version": 1, "status": "complete", "git_commit": commit,
        "input_hashes": inputs, "source_hashes": source_hashes,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "documents": len(records), "predictions": len(predictions),
        "artifact_hashes": {path.name: digest(path) for path in sorted(output.iterdir())
                            if path.is_file()},
        "interpretation": "Text controls only; no RST predictions or effects measured.",
    }
    write_json(output / "run_manifest.json", manifest)
    verify(prepared, output, config_path)
    return summary


def verify(prepared: Path, output: Path, config_path: Path) -> None:
    manifest = json.loads((output / "run_manifest.json").read_text())
    config = json.loads(config_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Run is incomplete")
    for name, expected in manifest["input_hashes"].items():
        path = config_path if name == "baseline_config.json" else prepared / name
        if digest(path) != expected:
            raise ValueError(f"Changed input: {name}")
    for name, expected in manifest["artifact_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Changed artifact: {name}")
    for name, expected in manifest["source_hashes"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Changed implementation/environment specification: {name}")
    records, splits = validate_prepared(prepared, config)
    by_id = validate_inputs(records, splits, config)
    test_folds = {name: fold["fold"] for fold in splits["outer_folds"]
                  for name in fold["test_ids"]}
    predictions = read_jsonl(output / "predictions.jsonl")
    for kind in config["controls"]:
        rows = [row for row in predictions if row["control"] == kind]
        if Counter(row["document_id"] for row in rows) != Counter(by_id.keys()):
            raise ValueError(f"Missing/repeated held-out predictions for {kind}")
        for row in rows:
            original = by_id[row["document_id"]]
            if (row["label"] != original["label"] or row["domain"] != original["domain"]
                    or row["fold"] != test_folds[row["document_id"]]):
                raise ValueError("Prediction identity/fold/label mismatch")
            if not 0 <= row["probability_positive"] <= 1 or row["prediction"] not in (0, 1):
                raise ValueError("Invalid prediction")
    if len(predictions) != len(by_id) * len(config["controls"]):
        raise ValueError("Unexpected prediction/control count")
    if summarize_predictions(predictions, config) != json.loads((output / "summary.json").read_text()):
        raise ValueError("Metrics do not match the saved held-out predictions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "baseline_config.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.prepared_dir, args.output_dir, args.config)
        print("Run verified: hashes, identities, held-out coverage, and metrics agree.")
    else:
        summary = run(args.prepared_dir, args.output_dir, args.config)
        print(json.dumps({kind: result["pooled_out_of_fold"]
                          for kind, result in summary["controls"].items()}, indent=2))


if __name__ == "__main__":
    main()
