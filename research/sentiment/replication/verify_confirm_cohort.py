"""Independently check the Amazon confirmation cohort against the exploration cohort.

Recomputes quotas, label/rating agreement, text hashes, fold balance and
disjointness from the prepared files, and checks that the parser sample uses
the exploration cohort's relation maps. Prints a JSON report.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", required=True, type=Path)
    cli.add_argument("--exploration-run-dir", required=True, type=Path)
    args = cli.parse_args()
    prepared, old = args.run_dir / "prepared", args.exploration_run_dir / "prepared"
    provenance = json.loads((prepared / "provenance.json").read_text())
    for name, expected in provenance["outputs_sha256"].items():
        if digest(prepared / name) != expected:
            raise ValueError(f"Changed prepared file: {name}")
    rows, previous = read_jsonl(prepared / "reviews.jsonl"), read_jsonl(old / "reviews.jsonl")
    manifest = read_jsonl(prepared / "manifest.jsonl")
    quota = provenance["selection"]["per_domain_and_label"]
    strata = Counter((row["domain"], row["label"]) for row in rows)
    problems = []
    if set(strata.values()) != {quota} or len(strata) != 8:
        problems.append(f"Unexpected strata: {dict(strata)}")
    for row in rows:
        normalized = " ".join(row["text"].split())
        if (normalized != row["text_normalized"] or sha256(normalized.encode()).hexdigest() != row["text_sha256"]
                or row["document_id"] != f"amazon23c-{row['text_sha256']}"):
            problems.append(f"Text hash or ID mismatch: {row['document_id']}")
        if row["label"] != int(row["rating"] >= 4) or row["rating"] not in (1, 2, 4, 5):
            problems.append(f"Rating and label disagree: {row['document_id']}")
    for key in ("text_sha256", "reviewer_sha256", "product_sha256"):
        if len({row[key] for row in rows}) != len(rows):
            problems.append(f"Repeated {key} inside the confirmation cohort")
    overlap = {key: len({r[key] for r in rows} & {r[key] for r in previous})
               for key in ("text_sha256", "reviewer_sha256", "product_sha256")}
    if any(overlap.values()):
        problems.append(f"Overlap with the exploration cohort: {overlap}")
    folds = Counter((row["outer_fold"], row["domain"], row["label"]) for row in manifest)
    if set(folds.values()) != {quota // 5}:
        problems.append("Outer folds are not balanced by category and polarity")
    old_selection = json.loads((old / "provenance.json").read_text())["selection"]
    selection = provenance["selection"]
    same_inputs = {
        "tokenizer": (selection["xlm_r_content_tokens"]["tokenizer_sha256"],
                      old_selection["xlm_r_content_tokens"]["tokenizer_sha256"]),
        "language_id_model": (selection["english_classifier"]["sha256"], old_selection["english_classifier"]["sha256"]),
        "tweeteval_train": (selection["tweeteval_train"]["sha256"], old_selection["tweeteval_train"]["sha256"]),
        "tweeteval_revision": (selection["tweeteval_train"]["revision"], old_selection["tweeteval_train"]["revision"]),
        "archives": (provenance["archive_sha256"],
                     json.loads((old / "provenance.json").read_text())["archive_sha256"]),
    }
    for name, (new, previous) in same_inputs.items():
        if new != previous:
            problems.append(f"{name} differs from the exploration cohort")
    sample = json.loads((args.run_dir / "sample.json").read_text())
    old_sample = json.loads((args.exploration_run_dir / "sample.json").read_text())
    if sample["maps"] != old_sample["maps"]:
        problems.append("Relation maps differ from the exploration parser sample")
    if [doc["document_id"] for doc in sample["documents"]] != [row["document_id"] for row in manifest]:
        problems.append("Parser sample order differs from the manifest")
    print(json.dumps({
        "reviews": len(rows), "per_domain_and_label": quota, "outer_fold_cells": len(folds),
        "overlap_with_exploration": overlap,
        "selection_inputs_match_exploration": sorted(name for name, (a, b) in same_inputs.items() if a == b),
        "exploration_reviews": len(previous),
        "prepared_outputs_sha256": provenance["outputs_sha256"], "sample_sha256": digest(args.run_dir / "sample.json"),
        "timestamps": [provenance["selection"]["earliest"], provenance["selection"]["latest"]],
        "rating_counts": provenance["selection"]["rating_counts"],
        "xlm_r_content_tokens": provenance["selection"]["xlm_r_content_tokens"],
        "problems": problems,
    }, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
