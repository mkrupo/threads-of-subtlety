"""Prepare all 3,000 rows of the UCI Sentiment Labelled Sentences release."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .parse_sfu import save_rows
from .parser_pilot import RESEARCH_REVISION, canonical_maps, digest, write_json

SEED = 42
SOURCE_FILES = {
    "amazon_cells": "amazon_cells_labelled.txt",
    "imdb": "imdb_labelled.txt",
    "yelp": "yelp_labelled.txt",
}


def normalized_group(text):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", html.unescape(text)).casefold()))


def text_digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def group_assignments(records, n_splits, seed):
    strata = [f"{row['domain']}/{row['label']}" for row in records]
    groups = [row["duplicate_group"] for row in records]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return {
        records[index]["document_id"]: fold
        for fold, (_, test) in enumerate(splitter.split(records, strata, groups))
        for index in test
    }


def grouped_splits(records, seed):
    assignments = group_assignments(records, 5, seed)
    folds = []
    for fold in range(5):
        train = [row for row in records if assignments[row["document_id"]] != fold]
        folds.append({
            "fold": fold,
            "train_ids": sorted(row["document_id"] for row in train),
            "test_ids": sorted(name for name, assigned in assignments.items() if assigned == fold),
            "inner_folds": group_assignments(train, 4, seed + fold + 1),
        })
    return {"outer_folds": folds}


def validate_grouped_inputs(records, splits, seed):
    by_id = {row["document_id"]: row for row in records}
    if len(by_id) != len(records) or not records:
        raise ValueError("Expected unique release-row IDs")
    label_by_group = {}
    for row in records:
        text, label = row["text"], row["label"]
        key = text_digest(normalized_group(text))
        if not text.strip() or label not in (0, 1) or row["duplicate_group"] != key:
            raise ValueError("Invalid text, label, or duplicate group")
        if label_by_group.setdefault(key, label) != label:
            raise ValueError("A duplicate group has conflicting labels")
    if grouped_splits(records, seed) != splits:
        raise ValueError("Fold assignments differ from the frozen grouped protocol")
    all_ids = set(by_id)
    tested = Counter()
    for fold in splits["outer_folds"]:
        train, test = set(fold["train_ids"]), set(fold["test_ids"])
        inner = fold["inner_folds"]
        if train & test or train | test != all_ids or set(inner) != train:
            raise ValueError("Invalid outer or inner split coverage")

        def groups(ids):
            return {by_id[name]["duplicate_group"] for name in ids}

        def strata(ids):
            return {(by_id[name]["domain"], by_id[name]["label"]) for name in ids}

        if groups(train) & groups(test) or strata(test) != strata(all_ids):
            raise ValueError("Outer test leaks a duplicate group or lacks a stratum")
        for index in range(4):
            validation = {name for name, value in inner.items() if value == index}
            if groups(validation) & groups(train - validation) or strata(validation) != strata(all_ids):
                raise ValueError("Inner validation leaks a duplicate group or lacks a stratum")
        tested.update(test)
    if set(tested.values()) != {1} or tested.keys() != by_id.keys():
        raise ValueError("Expected one outer test decision for each released row")
    return by_id


def grouped_bootstrap_weights(metadata, iterations, seed):
    """Resample duplicate groups together while preserving fold and stratum sizes."""
    groups = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[row["duplicate_group"]].append(index)
    strata = defaultdict(list)
    for indices in groups.values():
        fold, label = metadata[indices[0]]["outer_fold"], metadata[indices[0]]["label"]
        if any((metadata[i]["outer_fold"], metadata[i]["label"]) != (fold, label) for i in indices):
            raise ValueError("A group spans outer folds or polarities")
        domains = tuple(sorted(metadata[i]["domain"] for i in indices))
        strata[fold, label, domains].append(indices)
    rng = np.random.default_rng(seed)
    weights = np.zeros((iterations, len(metadata)), dtype=np.int16)
    for _, members in sorted(strata.items()):
        size = len(members)
        draws = rng.multinomial(size, np.full(size, 1 / size), size=iterations)
        for group_index, indices in enumerate(members):
            weights[:, indices] = draws[:, group_index, None]
    return weights


def load_archive(path):
    rows = []
    with zipfile.ZipFile(path) as archive:
        for domain, filename in SOURCE_FILES.items():
            member = f"sentiment labelled sentences/{filename}"
            source = archive.read(member).decode("utf-8-sig").split("\n")
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                text, label = line.rstrip("\r").rsplit("\t", 1)
                normalized = " ".join(text.split())
                text_hash = text_digest(normalized)
                group = normalized_group(text)
                rows.append({
                    "document_id": f"uci-{text_hash}-{domain}-{line_number}",
                    "domain": domain,
                    "label": int(label),
                    "text": text,
                    "text_normalized": normalized,
                    "text_sha256": text_hash,
                    "whitespace_tokens": len(normalized.split()),
                    "duplicate_group": text_digest(group),
                    "source_member": f"{member}:{line_number}",
                    "release_row": line_number,
                })
    if Counter((row["domain"], row["label"]) for row in rows) != Counter({
        (domain, label): 500 for domain in SOURCE_FILES for label in (0, 1)
    }):
        raise ValueError("Expected the full release with 500 examples per domain and polarity")
    return sorted(rows, key=lambda row: row["document_id"])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive", required=True, type=Path)
    cli.add_argument("--tweeteval-train", required=True, type=Path)
    cli.add_argument("--run-dir", required=True, type=Path)
    args = cli.parse_args()
    run = args.run_dir
    prepared = run / "prepared"
    if prepared.exists() or (run / "sample.json").exists():
        raise FileExistsError("Choose a fresh cohort output")

    records = load_archive(args.archive)
    splits = grouped_splits(records, SEED)
    validate_grouped_inputs(records, splits, SEED)
    tweets = args.tweeteval_train.read_text(encoding="utf-8").splitlines()
    tweets = [normalized_group(text) for text in tweets if text.strip()]
    if len(tweets) != 45615:
        raise ValueError(f"Unexpected TweetEval sentiment training count: {len(tweets)}")
    tweet_hashes = {text_digest(text) for text in tweets}
    groups = Counter(row["duplicate_group"] for row in records)
    matches = sum(groups[key] for key in groups.keys() & tweet_hashes)
    outer = {name: fold["fold"] for fold in splits["outer_folds"] for name in fold["test_ids"]}
    manifest = [
        {**{key: value for key, value in row.items() if key not in ("text", "text_normalized")},
         "outer_fold": outer[row["document_id"]]}
        for row in records
    ]
    prepared.mkdir()
    save_rows(prepared / "reviews.jsonl", records)
    save_rows(prepared / "manifest.jsonl", manifest)
    write_json(prepared / "splits.json", splits)
    hashes = {name: digest(prepared / name) for name in ("reviews.jsonl", "manifest.jsonl", "splits.json")}
    archive_hash = digest(args.archive)
    write_json(prepared / "provenance.json", {
        "corpus": "UCI Sentiment Labelled Sentences (Kotzias et al., 2015)",
        "document_count": len(records),
        "archive_sha256": archive_hash,
        "sources_sha256": {Path(__file__).name: digest(Path(__file__))},
        "outputs_sha256": hashes,
        "split_policy": {
            "seed": SEED, "outer_folds": 5, "inner_folds": 4, "strata": ["domain", "label"],
            "algorithm": "StratifiedGroupKFold (shuffle=True)", "groups": "duplicate_group",
            "inner_seed": "seed + zero-based outer fold + 1",
            "outer_test_counts": [
                dict(sorted(Counter(
                    f"{row['domain']}/{row['label']}" for row in manifest if row["outer_fold"] == fold
                ).items())) for fold in range(5)
            ],
        },
        "labels": {"negative": 0, "positive": 1},
        "document_id": "SHA-256 of whitespace-normalized body plus original domain and file line",
        "group_identity": "SHA-256 of HTML-unescaped, NFKC, casefolded alphanumeric word sequence",
        "duplicate_text_groups": sum(size > 1 for size in groups.values()),
        "distinct_groups": len(groups),
        "tweeteval_audit": {"training_rows": len(tweets), "sha256": digest(args.tweeteval_train),
                             "normalized_full_text_matches": matches},
    })
    config = json.loads(Path(__file__).with_name("baseline_config.json").read_text())
    config.update(
        protocol="uci-sls-text-controls-v1", document_id_prefix="uci", grouped_duplicate_cv=True,
        expected_corpus={"archive_sha256": archive_hash, "documents": len(records),
                         "domains": len(SOURCE_FILES), "per_domain_label": 500},
    )
    write_json(run / "baseline_config.json", config)
    write_json(run / "sample.json", {
        "documents": [{**row, "text": record["text"]} for row, record in zip(manifest, records, strict=True)],
        "archive_sha256": archive_hash,
        "inputs_sha256": hashes,
        "research_revision": RESEARCH_REVISION,
        "maps": canonical_maps(),
    })
    print(f"Prepared {len(records):,} released rows, {len(groups):,} text groups, and {matches} TweetEval matches")


if __name__ == "__main__":
    main()
