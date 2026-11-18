"""Compare review sentiment from central and supporting EDU averages."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import scipy
import sklearn

from .baseline import HERE, INPUT_FILES, digest, metrics, read_jsonl, validate_prepared, write_json
from .edu_baseline import CONTROL, RESULTS, fit_review_lr


def central_leaves(tree: dict) -> list[bool]:
    """Mark leaves in left-to-right EDU order by their path from the scene root."""
    nodes = tree["nodes"]
    pending = [(str(tree["root"]), True)]
    flags = []
    while pending:
        key, central = pending.pop()
        node = nodes[key]
        if "left" not in node:
            flags.append(central)
            continue
        left_role, right_role = node["nuclearity"]
        pending.append((str(node["right"]), central and right_role == "N"))
        pending.append((str(node["left"]), central and left_role == "N"))
    return flags


def pool_nuclearity(scores: list[dict], trees: list[dict]) -> tuple[dict, dict]:
    groups = defaultdict(lambda: ([], []))
    for score, tree in zip(scores, trees, strict=True):
        if (tree["key"], tree["document_id"], tree["status"]) != (score["key"], score["document_id"], "ok"):
            raise ValueError(f"Tree and scores disagree for scene {score['key']}")
        central, supporting = groups[score["document_id"]]
        for probabilities, flag in zip(score["probabilities"], central_leaves(tree), strict=True):
            (central if flag else supporting).append(probabilities)

    features = {}
    for document, (central, supporting) in groups.items():
        if not central:
            raise ValueError(f"No central EDU in review {document}")
        central_mean = np.mean(central, axis=0, dtype=np.float64)
        supporting_mean = np.mean(supporting, axis=0, dtype=np.float64) if supporting else central_mean
        features[document] = np.concatenate((central_mean, supporting_mean))
    coverage = {
        "reviews": len(groups), "scenes": len(scores),
        "central_edus": sum(len(central) for central, _ in groups.values()),
        "supporting_edus": sum(len(supporting) for _, supporting in groups.values()),
        "reviews_without_supporting_edus": sum(not supporting for _, supporting in groups.values()),
    }
    return features, coverage


def evaluate(features, by_id, splits, config, uniform, inventory):
    predictions, selections = [], []
    for fold in sorted(splits["outer_folds"], key=lambda f: f["fold"]):
        train, test = sorted(fold["train_ids"]), sorted(fold["test_ids"])
        search = fit_review_lr(
            [features[name] for name in train], [by_id[name]["label"] for name in train],
            [fold["inner_folds"][name] for name in train], config,
        )
        probabilities = search.predict_proba([features[name] for name in test])
        positive = probabilities[:, list(search.classes_).index(1)]
        if not np.isfinite(positive).all():
            raise ValueError("Non-finite review probabilities")
        held_out = []
        for name, probability in zip(test, positive, strict=True):
            label = by_id[name]["label"]
            reference = uniform[name]
            if (reference["fold"], reference["label"]) != (fold["fold"], label):
                raise ValueError("Uniform and nuclearity predictions have different fold or label assignments")
            held_out.append({
                "document_id": name, "fold": fold["fold"], "domain": by_id[name]["domain"],
                "label": label, "inventory": inventory, "probability_positive": float(probability),
                "prediction": int(probability >= config["decision_threshold"]),
            })
        predictions.extend(held_out)
        selections.append({
            "fold": fold["fold"], "selected_C": search.best_params_["logisticregression__C"],
            "mean_inner_macro_f1_by_C": dict(zip(
                map(str, config["regularization_C"]),
                search.cv_results_["mean_test_score"].tolist(), strict=True,
            )),
            "held_out": metrics([row["label"] for row in held_out], [row["prediction"] for row in held_out]),
        })
    if Counter(row["document_id"] for row in predictions) != Counter(by_id.keys()):
        raise ValueError("Expected one outer held-out prediction per review")
    labels = [row["label"] for row in predictions]
    classified = [row["prediction"] for row in predictions]
    previous = [uniform[row["document_id"]]["prediction"] for row in predictions]
    pooled = metrics(labels, classified)
    baseline = metrics(labels, previous)
    paired = list(zip(labels, previous, classified, strict=True))
    return predictions, {
        "pooled_out_of_fold": pooled,
        "macro_f1_delta_vs_uniform": pooled["macro_f1"] - baseline["macro_f1"],
        "corrected_reviews": sum(old != label and new == label for label, old, new in paired),
        "new_errors": sum(old == label and new != label for label, old, new in paired),
        "folds": selections,
    }


def run(prepared: Path, scores: Path, trees: Path, baseline_predictions: Path, output: Path) -> dict:
    baseline = json.loads((RESULTS / "sfu-edu-mean-v1.json").read_text())
    audit = json.loads((RESULTS / "sfu-full-parser-audit.json").read_text())
    runtime = {"python": sys.version.split()[0], "numpy": np.__version__,
               "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    config_path = HERE / "baseline_config.json"
    inputs = {name: digest(prepared / name) for name in INPUT_FILES}
    inputs.update({"baseline_config.json": digest(config_path), "scores.jsonl": digest(scores)})
    if inputs != baseline["input_hashes"]:
        raise ValueError("Use the same prepared reviews, folds, LR config and EDU scores as the uniform baseline")
    inputs["uniform_predictions.jsonl"] = digest(baseline_predictions)
    if inputs["uniform_predictions.jsonl"] != baseline["predictions_sha256"]:
        raise ValueError("Use the verified uniform baseline predictions")

    config = json.loads(config_path.read_text())
    records, splits = validate_prepared(prepared, config)
    by_id = {row["document_id"]: row for row in records}
    uniform = {row["document_id"]: row for row in read_jsonl(baseline_predictions)}
    score_rows = read_jsonl(scores)
    predictions, inventories = [], {}
    for inventory, parser in audit["inventory_reports"].items():
        path = trees / f"{inventory}.jsonl"
        inputs[path.name] = digest(path)
        if inputs[path.name] != parser["trees_sha256"]:
            raise ValueError(f"Tree checksum mismatch for {inventory}")
        features, coverage = pool_nuclearity(score_rows, read_jsonl(path))
        if (features.keys() != by_id.keys() or coverage["scenes"] != audit["scenes"]
                or coverage["central_edus"] + coverage["supporting_edus"] != audit["edus"]):
            raise ValueError(f"Tree coverage differs from the frozen corpus for {inventory}")
        rows, result = evaluate(features, by_id, splits, config, uniform, inventory)
        predictions.extend(rows)
        inventories[inventory] = {"coverage": coverage, **result}

    output.mkdir(parents=True)
    path = output / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    report = {
        "protocol": "sfu-nuclearity-v1",
        "primary_metric": baseline["primary_metric"],
        "features": {
            "group_order": ["central", "supporting"],
            "scores_per_group": baseline["features"]["columns"],
            "central_rule": "only nucleus branches from the scene root to the EDU",
            "aggregation": "mean over EDUs in each group across all scenes of the review",
            "missing_supporting_group": "use the review mean",
        },
        "runtime": runtime,
        "input_hashes": inputs,
        "source_hashes": {name: digest(HERE / name)
                          for name in ("nuclearity.py", "edu_baseline.py", "baseline.py", "uv.lock")},
        "predictions_sha256": digest(path),
        "uniform_out_of_fold": baseline["summary"]["controls"][CONTROL]["pooled_out_of_fold"],
        "inventories": inventories,
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prepared-dir", type=Path, required=True)
    cli.add_argument("--scores", type=Path, required=True)
    cli.add_argument("--trees-dir", type=Path, required=True)
    cli.add_argument("--uniform-predictions", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    report = run(args.prepared_dir, args.scores, args.trees_dir, args.uniform_predictions, args.output_dir)
    for inventory, result in report["inventories"].items():
        print(f"{inventory}: {result['pooled_out_of_fold']['macro_f1']:.4f} macro-F1 "
              f"({result['macro_f1_delta_vs_uniform']:+.4f} versus uniform)")


if __name__ == "__main__":
    main()
