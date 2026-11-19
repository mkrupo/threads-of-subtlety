"""Check fine/coarse macro-F1 stability on the same held-out SFU reviews."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .baseline import HERE, digest, read_jsonl, write_json
from .edu_baseline import RESULTS


ITERATIONS = 20_000
SEED = 42


def bootstrap_weights(metadata):
    strata = defaultdict(list)
    for i, row in enumerate(metadata):
        strata[row["outer_fold"], row["domain"], row["label"]].append(i)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((ITERATIONS, len(metadata)), dtype=np.int16)
    for _, indices in sorted(strata.items()):
        size = len(indices)
        weights[:, indices] = rng.multinomial(size, np.full(size, 1 / size), size=ITERATIONS)
    return weights


def macro_f1(weights, truth, predictions):
    """Compute the two-class macro-F1 for each row of review weights."""
    tp = weights @ (truth * predictions)
    tn = weights @ ((1 - truth) * (1 - predictions))
    errors = weights.sum(axis=1) - tp - tn
    return tp / (2 * tp + errors) + tn / (2 * tn + errors)


def paired_scores(matrix, truth, weights):
    fine = macro_f1(weights, truth, matrix[0])
    coarse = macro_f1(weights, truth, matrix[1])
    return fine - coarse


def pair_predictions(rows, metadata, inventory):
    predictions = []
    for name in ("fine", "coarse"):
        selected = [r for r in rows if r["inventory"] == inventory and r["granularity"] == name]
        by_id = {r["document_id"]: r for r in selected}
        if len(selected) != len(metadata) or len(by_id) != len(metadata):
            raise ValueError(f"Expected one prediction per review for {inventory}/{name}")
        values = []
        for review in metadata:
            row = by_id[review["document_id"]]
            if (row["label"], row["fold"], row["domain"]) != (review["label"], review["outer_fold"], review["domain"]):
                raise ValueError("Prediction metadata differs from the frozen review manifest")
            if row["prediction"] not in (0, 1):
                raise ValueError("Expected a binary review prediction")
            values.append(row["prediction"])
        predictions.append(values)
    return np.asarray(predictions, dtype=np.int16)


def masked_gaps(matrix, truth, values, omit=False):
    gaps = {}
    for value in np.unique(values):
        selected = values != value if omit else values == value
        gaps[str(value)] = float(paired_scores(matrix, truth, selected[None, :])[0])
    return gaps


def run(predictions, manifest):
    reference = json.loads((RESULTS / "sfu-relation-pooling-v1.json").read_text())
    hashes = {"predictions.jsonl": digest(predictions), "manifest.jsonl": digest(manifest)}
    if (hashes["predictions.jsonl"] != reference["predictions_sha256"]
            or hashes["manifest.jsonl"] != reference["input_hashes"]["manifest.jsonl"]):
        raise ValueError("Use the verified fine/coarse predictions and frozen review manifest")
    metadata = sorted(read_jsonl(manifest), key=lambda row: row["document_id"])
    rows = read_jsonl(predictions)
    if len(rows) != 2 * len(metadata) * len(reference["inventories"]):
        raise ValueError("Unexpected number of held-out predictions")
    truth = np.array([r["label"] for r in metadata], dtype=np.int16)
    folds = np.array([r["outer_fold"] for r in metadata])
    domains = np.array([r["domain"] for r in metadata])
    weights = bootstrap_weights(metadata)
    inventories = {}
    for inventory, prior in reference["inventories"].items():
        matrix = pair_predictions(rows, metadata, inventory)
        observed = float(paired_scores(matrix, truth, np.ones((1, len(truth)), dtype=np.int16))[0])
        if not np.isclose(observed, prior["paired_fine_vs_coarse"]["macro_f1_delta"], rtol=0, atol=1e-14):
            raise ValueError(f"Pooled gap does not match the verified report for {inventory}")
        draws = paired_scores(matrix, truth, weights)
        inventories[inventory] = {
            "fine_minus_coarse": observed,
            "bootstrap_95_percentile_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
            "differences_by_fold": masked_gaps(matrix, truth, folds),
            "differences_by_domain": masked_gaps(matrix, truth, domains),
            "differences_omitting_each_fold": masked_gaps(matrix, truth, folds, omit=True),
        }
    return {
        "protocol": "sfu-relation-stability-v1",
        "comparison": "fine minus coarse macro-F1 on fixed held-out predictions",
        "resampling": {"seed": SEED, "draws": ITERATIONS, "unit": "review", "paired": True,
                       "strata": ["outer_fold", "domain", "gold_label"]},
        "reviews": len(metadata),
        "input_hashes": hashes,
        "source_sha256": digest(HERE / "relation_stability.py"),
        "numpy_version": np.__version__,
        "inventories": inventories,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--predictions", type=Path, required=True)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Choose a fresh output file")
    write_json(args.output, run(args.predictions, args.manifest))


if __name__ == "__main__":
    main()
