"""Pre-registered fine-versus-coarse analyses for the Amazon confirmation cohort.

The protocol, thresholds and model settings are fixed in
GRANULARITY_CONFIRMATION.md. Every subcommand loads the cohort through
granularity_diagnostics.load_data, which checks the prepared files, scene file,
four tree files and EDU scores against the cohort's own audits.

``primary``  Fine, mapped-coarse and unlabeled relation-gate models: paired
             log-loss contrasts, Holm adjustment and equivalence verdicts.
``curve``    The same gate models on nested training subsets of each outer fold.
``others``   Fine versus coarse for text-order relation pooling and recursive
             composition.

The scrambled-label control is ``granularity_diagnostics scramble --corpus cohort``.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .baseline import metrics, read_jsonl
from .granularity_diagnostics import (
    GateData, add_data_arguments, bootstrap_p, holm, load_data, log_loss, resampled_means, runtime, source_hashes,
)
from .parser_pilot import INVENTORIES, digest, write_json
from .relation_stability import ITERATIONS, SEED, bootstrap_weights


SESOI = 0.002
ALPHA = 0.05
CURVE_SIZES = (1000, 2000, 4000, 8000)
CURVE_DRAWS = 5
MODES = ("unlabeled", "coarse", "fine")


def mode_labels(gate, mode):
    inv = gate.inv
    return {"unlabeled": (np.zeros_like(inv.fine_ids), 1),
            "coarse": (inv.coarse_ids, len(inv.coarse_labels)),
            "fine": (inv.fine_ids, len(inv.fine_labels))}[mode]


def verdict(difference, interval, holm_p):
    """Classify one fine-minus-coarse log-loss contrast under the fixed decision rule.

    A difference can be significant and still lie entirely inside the margin;
    that case gets a combined verdict.
    """
    inside = -SESOI < interval[0] and interval[1] < SESOI
    if holm_p < ALPHA:
        better = "fine better" if difference < 0 else "coarse better"
        return f"{better}, but within ±{SESOI}" if inside else better
    return f"equivalent within ±{SESOI}" if inside else "inconclusive"


def contrast_table(pairs, losses, draws, conditions):
    """Paired log-loss contrasts with bootstrap intervals and Holm-adjusted p-values."""
    rows = {}
    for left, right in pairs:
        difference = draws[left] - draws[right]
        rows[f"{left} - {right}"] = {
            "log_loss_difference": float(losses[left].mean() - losses[right].mean()),
            "interval_95": np.quantile(difference, [0.025, 0.975]).tolist(),
            "bootstrap_p": bootstrap_p(difference),
            "macro_f1_difference": conditions[left]["macro_f1"] - conditions[right]["macro_f1"],
        }
    for key, adjusted in holm({key: row["bootstrap_p"] for key, row in rows.items()}).items():
        rows[key]["holm_p"] = adjusted
    return rows


def write_predictions(path, metadata, probabilities):
    with path.open("x", encoding="utf-8") as stream:
        for condition, values in sorted(probabilities.items()):
            for row, value in zip(metadata, values, strict=True):
                stream.write(json.dumps({"document_id": row["document_id"], "outer_fold": row["outer_fold"],
                                         "label": row["label"], "condition": condition,
                                         "probability": float(value)}) + "\n")


def summarize(truth, probability):
    return {"log_loss": float(log_loss(truth, probability).mean()),
            **metrics(truth, (probability >= 0.5).astype(np.int16))}


def run_primary(args, data):
    probabilities, gates = {}, {}
    for inventory in INVENTORIES:
        gate = GateData(data, inventory)
        for mode in MODES:
            labels, count = mode_labels(gate, mode)
            probability, fits = gate.cross_validate(labels, count, use_labels=mode != "unlabeled")
            probabilities[f"{inventory}/{mode}"] = probability
            if mode != "unlabeled":
                vocabulary = gate.inv.fine_labels if mode == "fine" else gate.inv.coarse_labels
                gates[f"{inventory}/{mode}"] = [{"fold": f["fold"], "iterations": f["iterations"],
                                                 "gates": dict(zip(vocabulary, f["gates"].tolist(), strict=True))}
                                                for f in fits]
        print(f"primary {inventory} done", flush=True)
    truth = data.truth
    weights = bootstrap_weights(data.metadata).astype(np.float64)
    losses = {name: log_loss(truth, value) for name, value in probabilities.items()}
    draws = {name: resampled_means(weights, loss) for name, loss in losses.items()}
    conditions = {name: summarize(truth, value) for name, value in sorted(probabilities.items())}
    primary = contrast_table([(f"{inv}/fine", f"{inv}/coarse") for inv in INVENTORIES], losses, draws, conditions)
    for row in primary.values():
        row["verdict"] = verdict(row["log_loss_difference"], row["interval_95"], row["holm_p"])
    secondary = contrast_table([(f"{inv}/{mode}", f"{inv}/unlabeled") for inv in INVENTORIES
                                for mode in ("fine", "coarse")], losses, draws, conditions)
    args.output.mkdir(parents=True)
    write_predictions(args.output / "predictions.jsonl", data.metadata, probabilities)
    write_json(args.output / "report.json", {
        "protocol": "granularity-confirmation-v1/primary", "reviews": len(truth),
        "decision_rule": {"alpha": ALPHA, "sesoi_log_loss": SESOI, "adjustment": "Holm across the four standards"},
        "resampling": {"draws": ITERATIONS, "seed": SEED, "unit": "review", "paired": True,
                       "strata": ["outer_fold", "domain", "label"]},
        "input_hashes": data.hashes, "source_hashes": source_hashes(extra=("confirm_granularity.py",)),
        "runtime": runtime(), "conditions": conditions,
        "primary_fine_minus_coarse": primary, "secondary_labels_minus_unlabeled": secondary,
        "fitted_gates": gates, "predictions_sha256": digest(args.output / "predictions.jsonl"),
    })


def nested_subsets(metadata, train, fold, draw):
    """Stratified orderings of one training fold. Taking prefixes gives nested subsets."""
    rng = np.random.default_rng([SEED, fold, draw])
    strata = defaultdict(list)
    for index in train:
        strata[metadata[index]["domain"], metadata[index]["label"]].append(index)
    ordered = {key: np.array(members)[rng.permutation(len(members))] for key, members in sorted(strata.items())}
    return {size: np.sort(np.concatenate([members[:size // len(ordered)] for members in ordered.values()]))
            for size in CURVE_SIZES}


def run_curve(args, data):
    truth, metadata = data.truth, data.metadata
    sizes = (*CURVE_SIZES, "full")
    losses = {(inventory, size, mode): np.zeros(len(truth)) for inventory in INVENTORIES
              for size in sizes for mode in MODES}
    subset_hashes = {}
    start = perf_counter()
    for inventory in INVENTORIES:
        gate = GateData(data, inventory)
        labelled = {mode: mode_labels(gate, mode) for mode in MODES}
        folds = list(gate.folds)
        for fold, train, test in folds:
            draws = [nested_subsets(metadata, train, fold, draw) for draw in range(CURVE_DRAWS)]
            for draw, subsets in enumerate(draws):
                for size, subset in subsets.items():
                    key = f"{fold}/{draw}/{size}"
                    digest_value = digest_array(subset)
                    if subset_hashes.setdefault(key, digest_value) != digest_value:
                        raise ValueError("Subsets differ between inventories")
            for mode, (labels, count) in labelled.items():
                for size in sizes:
                    if size == "full":
                        runs = [train]
                    else:
                        runs = [subsets[size] for subsets in draws]
                    total = np.zeros(len(test))
                    for subset in runs:
                        gate.folds = [(fold, subset, test)]
                        probability, _ = gate.cross_validate(labels, count, use_labels=mode != "unlabeled")
                        total += log_loss(truth[test], probability[test])
                    losses[inventory, size, mode][test] = total / len(runs)
        gate.folds = folds
        print(f"curve {inventory} done after {perf_counter() - start:.0f}s", flush=True)
    weights = bootstrap_weights(metadata).astype(np.float64)
    report = {}
    for inventory in INVENTORIES:
        gaps, rows = {}, {}
        for size in sizes:
            gap = losses[inventory, size, "fine"] - losses[inventory, size, "coarse"]
            gaps[size] = resampled_means(weights, gap)
            rows[str(size)] = {
                **{f"{mode}_log_loss": float(losses[inventory, size, mode].mean()) for mode in MODES},
                "fine_minus_coarse": float(gap.mean()),
                "interval_95": np.quantile(gaps[size], [0.025, 0.975]).tolist(),
            }
        change = gaps[CURVE_SIZES[0]] - gaps["full"]
        report[inventory] = {
            "by_training_size": rows,
            "gap_at_smallest_minus_gap_at_full": {
                "difference": rows[str(CURVE_SIZES[0])]["fine_minus_coarse"] - rows["full"]["fine_minus_coarse"],
                "interval_95": np.quantile(change, [0.025, 0.975]).tolist(),
                "bootstrap_p": bootstrap_p(change),
                "reading": "positive means fine gains relative to coarse as training data grows",
            },
        }
    for inventory, adjusted in holm({inventory: row["gap_at_smallest_minus_gap_at_full"]["bootstrap_p"]
                                     for inventory, row in report.items()}).items():
        report[inventory]["gap_at_smallest_minus_gap_at_full"]["holm_p"] = adjusted
    args.output.mkdir(parents=True)
    write_json(args.output / "report.json", {
        "protocol": "granularity-confirmation-v1/curve", "sizes": list(sizes), "draws_per_size": CURVE_DRAWS,
        "interval_note": "bootstrap over test reviews with the training draws fixed; omits draw-to-draw variance",
        "subset_rule": "per outer fold and draw, shuffle each category-by-polarity stratum with seed "
                       "[42, fold, draw] and take the first size/8 reviews of each; prefixes are nested",
        "evaluation": "full outer test fold; per-review log-loss averaged over draws",
        "subset_sha256": subset_hashes, "input_hashes": data.hashes,
        "source_hashes": source_hashes(extra=("confirm_granularity.py",)), "runtime": runtime(),
        "inventories": report,
    })


def digest_array(values):
    from hashlib import sha256
    return sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def run_others(args, data):
    from .recursive_sentiment import build_forest, fit_predict as recursive_fit
    from .stock_pooling import Evaluation, feature_sets, uniform_predictions
    from .edu_baseline import mean_edu_scores

    from .granularity_diagnostics import CONFIGS

    truth, metadata = data.truth, data.metadata
    if digest(args.baseline_config) != digest(CONFIGS["amazon"]):
        raise ValueError("Use the frozen Amazon logistic-regression configuration")
    if len(data.scenes) != len(metadata):
        raise ValueError("Recursive composition expects one scene tree per review")
    by_id = {row["document_id"]: row for row in metadata}
    index = {row["document_id"]: i for i, row in enumerate(metadata)}
    config = json.loads(args.baseline_config.read_text())
    uniform = uniform_predictions(mean_edu_scores(args.scores), by_id, data.splits, config)
    evaluation = Evaluation(by_id, data.splits, config, uniform)
    score_rows = read_jsonl(args.scores)
    labels = ("negative", "neutral", "positive")
    probabilities = {}
    for inventory, (path, table, mapping) in data.tree_specs.items():
        trees = read_jsonl(path)
        for feature, features in feature_sets(score_rows, trees, table, mapping, "fine_checkpoint"):
            if not feature.startswith("relation-text-order"):
                continue
            rows, _, _ = evaluation.fit(features, inventory)
            values = np.full(len(index), np.nan)
            for row in rows:
                values[index[row["document_id"]]] = row["probability_positive"]
            probabilities[f"pooling/{inventory}/{feature.rsplit('/', 1)[1]}"] = values
        fine = sorted({label.rsplit("_", 1)[0].lower() for label in table})
        coarse = sorted({mapping[label] for label in fine if label in mapping})
        fine_ids, coarse_ids = ({label: i for i, label in enumerate(group)} for group in (fine, coarse))
        records = {}
        for tree, score in zip(trees, score_rows, strict=True):
            if (tree["key"], tree["document_id"]) != (score["key"], score["document_id"]):
                raise ValueError("Trees and EDU scores are in different scene order")
            values = np.asarray(score["probabilities"])
            records[score["document_id"]] = (tree, values[:, labels.index("positive")] - values[:, labels.index("negative")])
        for mode, count in (("coarse", len(coarse)), ("fine", len(fine))):
            values = np.full(len(index), np.nan)
            for fold in data.splits["outer_folds"]:
                train_ids, test_ids = (sorted(fold[key]) for key in ("train_ids", "test_ids"))
                train_trees, test_trees = (build_forest((records[doc] for doc in ids), fine_ids, coarse_ids, mapping)
                                           for ids in (train_ids, test_ids))
                scores, _ = recursive_fit(train_trees, test_trees, truth[[index[d] for d in train_ids]], mode, count)
                values[[index[d] for d in test_ids]] = scores
            probabilities[f"recursive/{inventory}/{mode}"] = values
        print(f"others {inventory} done", flush=True)
    weights = bootstrap_weights(metadata).astype(np.float64)
    losses = {name: log_loss(truth, value) for name, value in probabilities.items()}
    draws = {name: resampled_means(weights, loss) for name, loss in losses.items()}
    conditions = {name: summarize(truth, value) for name, value in sorted(probabilities.items())}
    pairs = [(f"{model}/{inv}/fine", f"{model}/{inv}/coarse") for model in ("pooling", "recursive")
             for inv in INVENTORIES]
    args.output.mkdir(parents=True)
    write_predictions(args.output / "predictions.jsonl", metadata, probabilities)
    write_json(args.output / "report.json", {
        "protocol": "granularity-confirmation-v1/others", "input_hashes": {**data.hashes,
        args.baseline_config.name: digest(args.baseline_config)},
        "source_hashes": source_hashes(extra=("confirm_granularity.py", "recursive_sentiment.py")),
        "runtime": runtime(), "conditions": conditions,
        "fine_minus_coarse": contrast_table(pairs, losses, draws, conditions),
        "predictions_sha256": digest(args.output / "predictions.jsonl"),
    })


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("primary", "curve", "others"):
        command = sub.add_parser(name)
        add_data_arguments(command, corpora=("cohort",))
        command.add_argument("--output", type=Path, required=True)
    sub.choices["others"].add_argument("--baseline-config", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = load_data(args)
    {"primary": run_primary, "curve": run_curve, "others": run_others}[args.command](args, data)


if __name__ == "__main__":
    main()
