"""Verify held-out results and compare the Amazon and SFU replications."""

import argparse
from collections import Counter, defaultdict
import json
from math import isclose
from pathlib import Path
from statistics import median

import numpy as np

from research.sentiment.baseline import read_jsonl
from research.sentiment.parser_pilot import canonical_maps
from research.sentiment.relation_stability import ITERATIONS, SEED, bootstrap_weights, macro_f1

INVENTORIES = ("deu.rst.pcc", "nld.rst.nldt", "eng.erst.gum", "eng.rst.rstdt")
FACTOR_MODES = ("fine", "coarse")


def load(path):
    return json.loads(path.read_text())


def conditions(root, is_sfu):
    text_dir = root / ("text-controls-exact" if is_sfu else "text-controls")
    uniform_dir = root / ("edu-mean-strict" if is_sfu else "edu-mean")
    for name, report in load(text_dir / "summary.json")["controls"].items():
        yield name, text_dir / "predictions.jsonl", {"control": name}, report["pooled_out_of_fold"]
    mean_report = load(uniform_dir / "report.json")["summary"]["controls"]["edu_mean_lr"]
    yield "edu_mean", uniform_dir / "predictions.jsonl", {}, mean_report["pooled_out_of_fold"]
    for stage in ("nuclearity", "relation-text-order", "relation-nucleus-first", "motifs"):
        directory = root / stage
        for inventory, result in load(directory / "report.json")["inventories"].items():
            key = f"{inventory}/{stage}"
            fields = {"inventory": inventory}
            if stage == "nuclearity":
                yield key, directory / "predictions.jsonl", fields, result["pooled_out_of_fold"]
            else:
                for mode in FACTOR_MODES:
                    yield f"{key}/{mode}", directory / "predictions.jsonl", {
                        **fields, "granularity": mode,
                    }, result[mode]["pooled_out_of_fold"]


def analyze_fit(root, is_sfu):
    metadata = sorted(read_jsonl(root / "prepared/manifest.jsonl"), key=lambda row: row["document_id"])
    truth = np.array([row["label"] for row in metadata], dtype=np.int16)
    weights = bootstrap_weights(metadata)
    scores, draws, row_cache = {}, {}, {}
    for name, path, fields, expected in conditions(root, is_sfu):
        if path not in row_cache:
            row_cache[path] = read_jsonl(path)
        rows = sorted(
            (row for row in row_cache[path] if all(row[k] == v for k, v in fields.items())),
            key=lambda row: row["document_id"],
        )
        decisions = []
        for pred, meta in zip(rows, metadata, strict=True):
            if (pred["document_id"], pred["label"], pred["fold"], pred["domain"]) != (
                meta["document_id"], meta["label"], meta["outer_fold"], meta["domain"],
            ) or pred["prediction"] not in (0, 1):
                raise ValueError(f"Invalid held-out row in {name}")
            decisions.append(pred["prediction"])
        counts = Counter(zip(truth, decisions))
        tn, fp, fn, tp = (counts[a, b] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
        f1 = tn / (2 * tn + fp + fn) + tp / (2 * tp + fp + fn)
        accuracy = (tn + tp) / len(metadata)
        if (expected["confusion_matrix_labels_0_1"] != [[tn, fp], [fn, tp]]
                or not isclose(f1, expected["macro_f1"], abs_tol=1e-14, rel_tol=0)
                or not isclose(accuracy, expected["accuracy"], abs_tol=1e-14, rel_tol=0)):
            raise ValueError(f"Independent metric differs in {name}")
        scores[name] = {"macro_f1": expected["macro_f1"], "accuracy": accuracy}
        draws[name] = macro_f1(weights, truth, np.array(decisions, dtype=np.int16))
    return scores, draws, len(metadata) * len(scores)


def paired(scores, draws, left, right):
    return {
        "difference": scores[left]["macro_f1"] - scores[right]["macro_f1"],
        "interval_95": np.quantile(draws[left] - draws[right], [0.025, 0.975]).tolist(),
    }


def contrasts(scores, draws):
    result = {}
    for inventory in INVENTORIES:
        def name(stage, mode=None):
            key = f"{inventory}/{stage}"
            return f"{key}/{mode}" if mode else key
        control = name("nuclearity")
        report = {"nuclearity_minus_mean": paired(scores, draws, control, "edu_mean")}
        for stage in ("relation-text-order", "relation-nucleus-first", "motifs"):
            fine, coarse = (name(stage, mode) for mode in ("fine", "coarse"))
            report[f"{stage}_fine_minus_coarse"] = paired(scores, draws, fine, coarse)
            if stage == "motifs":
                continue
            for mode in ("fine", "coarse"):
                key = name(stage, mode)
                report[f"{stage}_{mode}_minus_nuclearity"] = paired(scores, draws, key, control)
                if stage == "relation-nucleus-first":
                    report[f"{mode}_nucleus_first_minus_text_order"] = paired(
                        scores, draws, key, name("relation-text-order", mode),
                    )
        result[inventory] = report
    return result


def cohort(root):
    audit = load(root / "parser-audit.json")
    scene_counts = Counter()
    edus = Counter()
    for row in read_jsonl(root / "parser-verified/scenes.jsonl"):
        scene_counts[row["document_id"]] += 1
        edus[row["document_id"]] += len(row["edus"])
    texts = read_jsonl(root / "prepared/reviews.jsonl")
    return {
        "reviews": audit["documents"], "scenes": audit["scenes"], "edus": audit["edus"],
        "multi_scene_reviews": sum(count > 1 for count in scene_counts.values()),
        "one_edu_reviews": sum(count == 1 for count in edus.values()),
        "median_edus_per_review": median(edus.values()),
        "median_words_per_review": median(row["whitespace_tokens"] for row in texts),
        "max_scene_xlm_r_tokens": audit["max_scene_tokens"],
        "complete_reviews": audit["complete_documents_across_inventories"],
        "alignment_retries": {name: row["retry"]["alignment_failures_retried"]
                              for name, row in audit["inventory_reports"].items()},
        "trees": {name: row["scene_status"]["ok"] for name, row in audit["inventory_reports"].items()},
    }


def added_maps():
    original = canonical_maps()
    local = load(RUN / "sample.json")["maps"]
    audit = load(RUN / "parser-audit.json")["inventory_reports"]
    outputs = {}
    for inventory, mapping in local.items():
        if any(mapping.get(label) != parent for label, parent in original[inventory].items()):
            raise ValueError("An existing SFU coarse map assignment changed")
        outputs[inventory] = {
            label: {"parent": parent, "predicted_nodes": audit[inventory]["observed_relations"].get(label, 0)}
            for label, parent in sorted(mapping.items()) if label not in original[inventory]
        }
    return outputs


def verify_converter_retry():
    counts = {}
    for inventory in INVENTORIES:
        path = f"{inventory}.jsonl"
        before = read_jsonl(RUN / "parser-initial-retry" / path)
        after = read_jsonl(RUN / "parser-verified" / path)
        matched = recovered = 0
        for old, new in zip(before, after, strict=True):
            if (new["key"], new["document_id"], new["status"]) != (old["key"], old["document_id"], "ok"):
                raise ValueError(f"Changed scene identity or unresolved parse failure in {inventory}")
            if old["status"] == "ok":
                if old != new:
                    raise ValueError(f"Converter retry changed a successful tree in {inventory}")
                matched += 1
            else:
                recovered += 1
        counts[inventory] = {"prior_successes_unchanged": matched, "additional_trees_recovered": recovered}
    return counts


cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--sfu-run", required=True, type=Path)
cli.add_argument("--amazon-run", required=True, type=Path)
args = cli.parse_args()
RUN = args.amazon_run
scorer_sfu = load(args.sfu_run / "edu-scores/manifest.json")
scorer_amazon = load(RUN / "edu-scores/manifest.json")
parser_sfu = load(args.sfu_run / "parser-audit.json")["inventory_reports"]
parser_amazon = load(RUN / "parser-audit.json")["inventory_reports"]
training_sfu = load(args.sfu_run / "text-controls-exact/protocol.json")
training_amazon = load(RUN / "text-controls/protocol.json")
if scorer_sfu["model"] != scorer_amazon["model"] or scorer_sfu["runtime"] != scorer_amazon["runtime"]:
    raise ValueError("Frozen EDU scorer inputs or runtime differ")
for inventory in INVENTORIES:
    for key in ("weights_sha256", "config_sha256", "table_sha256", "ordered_relation_table"):
        if parser_sfu[inventory][key] != parser_amazon[inventory][key]:
            raise ValueError(f"Parser input {key} differs for {inventory}")
for key, value in training_sfu.items():
    if key not in ("protocol", "expected_corpus") and training_amazon[key] != value:
        raise ValueError(f"Training protocol value differs for {key}")
sfu, sfu_draws, sfu_count = analyze_fit(args.sfu_run, True)
amazon, amazon_draws, amazon_count = analyze_fit(RUN, False)
if sfu.keys() != amazon.keys() or len(sfu) != 32:
    raise ValueError("The corpora must have the same 32 fitted conditions")
report = {
    "cohorts": {"SFU": cohort(args.sfu_run), "Amazon": cohort(RUN)},
    "amazon_provenance": load(RUN / "prepared/provenance.json")["selection"],
    "amazon_raw_verification": load(RUN / "cohort-verification.json"),
    "independently_verified_predictions": {"SFU": sfu_count, "Amazon": amazon_count},
    "shared_protocol_verified": {
        "same_parser_weights_configs_and_relation_tables": True,
        "same_edu_scorer_model_and_runtime": True,
        "same_fold_and_classifier_settings": True,
    },
    "bootstrap": {
        "draws": ITERATIONS, "seed": SEED, "strata": ["outer_fold", "domain", "gold_label"],
        "interpretation": "Pointwise paired intervals on fixed held-out decisions, within each corpus.",
    },
    "macro_f1_and_accuracy": {
        name: {"SFU": sfu[name], "Amazon": amazon[name],
               "amazon_minus_sfu_macro_f1": amazon[name]["macro_f1"] - sfu[name]["macro_f1"]}
        for name in amazon
    },
    "paired_macro_f1_gaps": {"SFU": contrasts(sfu, sfu_draws), "Amazon": contrasts(amazon, amazon_draws)},
    "amazon_coarse_map_additions": added_maps(),
}
if (RUN / "parser-initial-retry").exists():
    report["amazon_converter_retry_verification"] = verify_converter_retry()
print(json.dumps(report, indent=2, sort_keys=True))
