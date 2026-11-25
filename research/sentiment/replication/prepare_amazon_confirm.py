"""Freeze a larger Amazon Reviews '23 confirmation cohort disjoint from the exploration cohort.

The rules are those of prepare_amazon.py: the same four category archives,
timestamps from 2022, stars 1-2 versus 4-5, 40-160 XLM-R content tokens,
fastText English probability of at least 0.80, no normalized match to the
TweetEval sentiment training texts, and at most one review per normalized
text, reviewer and product. Candidates are visited in the same seeded hash
order. Every review text, reviewer and product of the exploration cohort is
excluded before selection, so no review, author or product occurs in both.

Stage this file next to prepare_amazon.py in the Amazon parser source.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from statistics import median, quantiles

import fasttext
from tokenizers import Tokenizer

from .corpus import build_splits
from .parse_sfu import read_jsonl, save_rows
from .parser_pilot import RESEARCH_REVISION, canonical_maps, digest, write_json
from .prepare_amazon import (
    BODY_BREAK, DOMAINS, LID_MIN, MAX_TOKENS, MIN_TOKENS, SEED, START, candidates, checksum, normalize_for_match,
)

PREFIX = "amazon23c"


def exploration_exclusions(prepared):
    """Normalized texts, reviewer hashes and product hashes of the exploration cohort."""
    provenance = json.loads((prepared / "provenance.json").read_text())
    for name, expected in provenance["outputs_sha256"].items():
        if digest(prepared / name) != expected:
            raise ValueError(f"Exploration cohort file changed: {name}")
    rows = read_jsonl(prepared / "reviews.jsonl")
    return ({normalize_for_match(row["text_normalized"]) for row in rows},
            {row["reviewer_sha256"] for row in rows}, {row["product_sha256"] for row in rows},
            provenance)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", required=True, type=Path, help="Fresh run directory for this cohort")
    cli.add_argument("--inputs", required=True, type=Path,
                     help="Exploration run directory with the raw archives, lid.176.ftz and TweetEval files")
    cli.add_argument("--exclude-prepared", required=True, type=Path, help="Exploration cohort prepared/ folder")
    cli.add_argument("--tokenizer", required=True, type=Path)
    cli.add_argument("--per-stratum", required=True, type=int)
    args = cli.parse_args()
    run, inputs = args.run_dir, args.inputs
    prepared = run / "prepared"
    if prepared.exists() or (run / "sample.json").exists():
        raise FileExistsError("Choose a fresh cohort output")
    excluded_text, excluded_users, excluded_items, exploration = exploration_exclusions(args.exclude_prepared)
    sources = {domain: inputs / f"{domain}.jsonl.gz" for domain in DOMAINS}
    archives = {path.name: digest(path) for path in sources.values()}
    archive_hash = checksum(json.dumps(archives, sort_keys=True))
    if archive_hash != exploration["archive_sha256"]:
        raise ValueError("Raw archives differ from the exploration cohort")
    tweet_path = inputs / "tweeteval-sentiment-train.txt"
    with tweet_path.open(encoding="utf-8") as stream:
        tweet_keys = {normalize_for_match(row) for row in stream if row.strip()}
    lid = fasttext.load_model(str(inputs / "lid.176.ftz"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    seen_reviews, seen_users, seen_items = set(excluded_text), set(excluded_users), set(excluded_items)
    records, eligibility = [], {}
    for domain, source in sources.items():
        buckets = candidates(domain, source)
        for label, rows in buckets.items():
            count = 0
            audit = Counter(candidate_time_and_rating=len(rows))
            for _, line, user, item, timestamp, rating, raw in sorted(rows):
                user_hash, item_hash = checksum(user), checksum(item)
                text = BODY_BREAK.sub("\n", html.unescape(raw)).replace("\r\n", "\n").replace("\r", "\n")
                normalized = " ".join(text.split())
                key = normalize_for_match(normalized)
                if key in excluded_text or user_hash in excluded_users or item_hash in excluded_items:
                    audit["exploration_text_reviewer_or_product"] += 1
                    continue
                tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
                if not MIN_TOKENS <= tokens <= MAX_TOKENS:
                    audit["length_exclusions"] += 1
                    continue
                language, confidence = lid.predict(normalized, k=1)
                if language[0] != "__label__en" or confidence[0] < LID_MIN:
                    audit["language_exclusions"] += 1
                    continue
                if key in tweet_keys:
                    audit["tweeteval_matches_excluded"] += 1
                    continue
                if key in seen_reviews or user_hash in seen_users or item_hash in seen_items:
                    audit["duplicate_text_reviewer_or_product"] += 1
                    continue
                seen_reviews.add(key)
                seen_users.add(user_hash)
                seen_items.add(item_hash)
                text_hash = checksum(normalized)
                records.append({
                    "document_id": f"{PREFIX}-{text_hash}", "domain": domain, "label": label, "rating": rating,
                    "text": text, "text_normalized": normalized, "text_sha256": text_hash,
                    "whitespace_tokens": len(normalized.split()), "xlm_r_content_tokens": tokens,
                    "source_member": f"{source.name}:{line}",
                    "timestamp_utc": datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat(),
                    "reviewer_sha256": user_hash, "product_sha256": item_hash,
                    "english_confidence": float(confidence[0]),
                })
                count += 1
                if count == args.per_stratum:
                    break
            eligibility[f"{domain}/{label}"] = dict(audit, selected=count)
            print(f"Selected {domain}/{label}: {count} reviews from {len(rows):,} candidates", flush=True)
    short = {stratum: row["selected"] for stratum, row in eligibility.items() if row["selected"] != args.per_stratum}
    if short:
        raise ValueError(f"Unable to fill every stratum at {args.per_stratum}: {short}")

    records.sort(key=lambda row: row["document_id"])
    splits = build_splits(records, SEED)
    outer = {name: fold["fold"] for fold in splits["outer_folds"] for name in fold["test_ids"]}
    manifest = [{**{k: v for k, v in row.items() if k not in ("text", "text_normalized")},
                 "outer_fold": outer[row["document_id"]]} for row in records]
    prepared.mkdir(parents=True)
    save_rows(prepared / "reviews.jsonl", records)
    save_rows(prepared / "manifest.jsonl", manifest)
    write_json(prepared / "splits.json", splits)
    hashes = {name: digest(prepared / name) for name in ("reviews.jsonl", "manifest.jsonl", "splits.json")}
    lengths = [row["xlm_r_content_tokens"] for row in records]
    words = [row["whitespace_tokens"] for row in records]
    write_json(prepared / "provenance.json", {
        "corpus": "McAuley Lab Amazon Reviews '23, confirmation cohort",
        "document_count": len(records),
        "archive_sha256": archive_hash,
        "source_archives_sha256": archives,
        "sources_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                           (Path(__file__).name, "prepare_amazon.py", "corpus.py")},
        "outputs_sha256": hashes,
        "excluded_exploration_cohort": {
            "prepared_outputs_sha256": exploration["outputs_sha256"],
            "texts": len(excluded_text), "reviewers": len(excluded_users), "products": len(excluded_items),
        },
        "split_policy": {"seed": SEED, "outer_folds": 5, "inner_folds": 4, "strata": ["domain", "label"]},
        "selection": {
            "domains": list(DOMAINS), "per_domain_and_label": args.per_stratum,
            "timestamps_from": START.isoformat(),
            "earliest": min(row["timestamp_utc"] for row in records),
            "latest": max(row["timestamp_utc"] for row in records),
            "ratings": {"negative": [1, 2], "positive": [4, 5]},
            "rating_counts": dict(Counter(str(row["rating"]) for row in records)),
            "domain_by_label": eligibility,
            "xlm_r_content_tokens": {"min_allowed": MIN_TOKENS, "max_allowed": MAX_TOKENS, "min": min(lengths),
                                     "max": max(lengths), "median": median(lengths),
                                     "p95": quantiles(lengths, n=100)[94], "tokenizer_sha256": digest(args.tokenizer)},
            "whitespace_tokens": {"min": min(words), "max": max(words), "median": median(words),
                                  "p95": quantiles(words, n=100)[94]},
            "english_classifier": {"model": "lid.176.ftz", "sha256": digest(inputs / "lid.176.ftz"), "min_p": LID_MIN},
            "tweeteval_train": {"revision": (inputs / "tweeteval-revision.txt").read_text().strip(),
                                "rows_normalized_unique": len(tweet_keys), "sha256": digest(tweet_path)},
            "unique_reviewers": len({row["reviewer_sha256"] for row in records}),
            "unique_products": len({row["product_sha256"] for row in records}),
        },
    })
    write_json(run / "sample.json", {
        "documents": [{**row, "text": record["text"]} for row, record in zip(manifest, records, strict=True)],
        "archive_sha256": archive_hash, "inputs_sha256": hashes,
        "research_revision": RESEARCH_REVISION, "maps": canonical_maps(),
    })
    print(f"Frozen {len(records):,} reviews, median {median(lengths):g} XLM-R tokens", flush=True)


if __name__ == "__main__":
    main()
