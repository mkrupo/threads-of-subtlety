"""Freeze the post-2021 short English Amazon Reviews '23 replication cohort."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import html
import json
from pathlib import Path
import re
from statistics import median, quantiles
import unicodedata

import fasttext
from tokenizers import Tokenizer

from .corpus import build_splits
from .parse_sfu import save_rows
from .parser_pilot import RESEARCH_REVISION, canonical_maps, digest, write_json

SEED = 42
DOMAINS = ("All_Beauty", "Appliances", "Musical_Instruments", "Software")
PER_STRATUM = 250
MIN_TOKENS, MAX_TOKENS = 40, 160
LID_MIN = 0.80
START = datetime(2022, 1, 1, tzinfo=timezone.utc)
BODY_BREAK = re.compile(r"<br\s*/?\s*>", re.IGNORECASE)
WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def checksum(text):
    return sha256(text.encode("utf-8")).hexdigest()


def normalize_for_match(text):
    return " ".join(WORDS.findall(unicodedata.normalize("NFKC", html.unescape(text)).casefold()))


def candidates(domain, source):
    buckets = {0: [], 1: []}
    cutoff = int(START.timestamp() * 1000)
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line_index, line in enumerate(stream, 1):
            row = json.loads(line)
            rating = row["rating"]
            if (row["timestamp"] < cutoff or rating not in (1, 2, 4, 5)
                    or not row["text"].strip() or not row["user_id"] or not row["parent_asin"]):
                continue
            key = checksum(json.dumps((SEED, domain, row["user_id"], row["parent_asin"],
                                       row["timestamp"], row["text"]), ensure_ascii=False))
            buckets[int(rating >= 4)].append(
                (key, line_index, row["user_id"], row["parent_asin"], row["timestamp"],
                 int(rating), row["text"])
            )
    return buckets


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", required=True, type=Path)
    cli.add_argument("--tokenizer", required=True, type=Path)
    args = cli.parse_args()
    run = args.run_dir
    prepared = run / "prepared"
    if prepared.exists() or (run / "sample.json").exists():
        raise FileExistsError("Choose a fresh cohort output")
    sources = {domain: run / f"{domain}.jsonl.gz" for domain in DOMAINS}
    tweet_path = run / "tweeteval-sentiment-train.txt"
    with tweet_path.open(encoding="utf-8") as stream:
        tweet_train = [normalize_for_match(row) for row in stream if row.strip()]
    if len(tweet_train) != 45615:
        raise ValueError(f"Unexpected TweetEval sentiment training count: {len(tweet_train)}")
    tweet_keys = set(tweet_train)
    lang_path = run / "lid.176.ftz"
    lid = fasttext.load_model(str(lang_path))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    seen_reviews, seen_users, seen_items = set(), set(), set()
    records, eligibility = [], {}
    for domain, source in sources.items():
        buckets = candidates(domain, source)
        for label, rows in buckets.items():
            count = 0
            audit = Counter(candidate_time_and_rating=len(rows))
            for _, line, user, item, timestamp, rating, raw in sorted(rows):
                text = BODY_BREAK.sub("\n", html.unescape(raw)).replace("\r\n", "\n").replace("\r", "\n")
                normalized = " ".join(text.split())
                tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
                if not MIN_TOKENS <= tokens <= MAX_TOKENS:
                    audit["length_exclusions"] += 1
                    continue
                language, confidence = lid.predict(normalized, k=1)
                if language[0] != "__label__en" or confidence[0] < LID_MIN:
                    audit["language_exclusions"] += 1
                    continue
                key = normalize_for_match(normalized)
                if key in tweet_keys:
                    audit["tweeteval_matches_excluded"] += 1
                    continue
                if key in seen_reviews or user in seen_users or item in seen_items:
                    audit["duplicate_text_reviewer_or_product"] += 1
                    continue
                seen_reviews.add(key)
                seen_users.add(user)
                seen_items.add(item)
                text_hash = checksum(normalized)
                records.append({
                    "document_id": f"amazon23-{text_hash}",
                    "domain": domain,
                    "label": label,
                    "rating": rating,
                    "text": text,
                    "text_normalized": normalized,
                    "text_sha256": text_hash,
                    "whitespace_tokens": len(normalized.split()),
                    "xlm_r_content_tokens": tokens,
                    "source_member": f"{source.name}:{line}",
                    "timestamp_utc": datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat(),
                    "reviewer_sha256": checksum(user),
                    "product_sha256": checksum(item),
                    "english_confidence": float(confidence[0]),
                })
                count += 1
                if count == PER_STRATUM:
                    break
            if count != PER_STRATUM:
                raise ValueError(f"Unable to fill {domain}/{label}: {count}/{PER_STRATUM}")
            eligibility[f"{domain}/{label}"] = dict(audit, selected=count)
            print(f"Selected {domain}/{label}: {count} reviews from {len(rows):,} candidates", flush=True)
    records.sort(key=lambda row: row["document_id"])
    splits = build_splits(records, SEED)
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
    archives = {path.name: digest(path) for path in sources.values()}
    archive_hash = checksum(json.dumps(archives, sort_keys=True))
    lengths = [row["xlm_r_content_tokens"] for row in records]
    words = [row["whitespace_tokens"] for row in records]
    provenance = {
        "corpus": "McAuley Lab Amazon Reviews '23",
        "document_count": len(records),
        "archive_sha256": archive_hash,
        "archive_sha256_definition": "SHA-256 of the sorted JSON object mapping source filenames to raw SHA-256",
        "source_archives_sha256": archives,
        "sources_sha256": {name: digest(Path(__file__).with_name(name))
                           for name in (Path(__file__).name, "corpus.py")},
        "outputs_sha256": hashes,
        "split_policy": {"seed": SEED, "outer_folds": 5, "inner_folds": 4, "strata": ["domain", "label"]},
        "selection": {
            "domains": list(DOMAINS), "per_domain_and_label": PER_STRATUM,
            "timestamps_from": START.isoformat(), "earliest": min(row["timestamp_utc"] for row in records),
            "latest": max(row["timestamp_utc"] for row in records),
            "ratings": {"negative": [1, 2], "positive": [4, 5]},
            "rating_counts": dict(Counter(str(row["rating"]) for row in records)),
            "domain_by_label": eligibility,
            "xlm_r_content_tokens": {
                "min_allowed": MIN_TOKENS, "max_allowed": MAX_TOKENS, "min": min(lengths),
                "max": max(lengths), "median": median(lengths), "p95": quantiles(lengths, n=100)[94],
                "tokenizer_sha256": digest(args.tokenizer),
            },
            "whitespace_tokens": {"min": min(words), "max": max(words), "median": median(words),
                                  "p95": quantiles(words, n=100)[94]},
            "english_classifier": {"model": lang_path.name, "sha256": digest(lang_path), "min_p": LID_MIN},
            "tweeteval_train": {
                "revision": (run / "tweeteval-revision.txt").read_text().strip(),
                "rows": len(tweet_train), "sha256": digest(tweet_path),
                "normalized_full_review_matches": len(seen_reviews & tweet_keys),
            },
            "unique_normalized_reviews": len(seen_reviews),
            "unique_reviewers": len(seen_users),
            "unique_products": len(seen_items),
        },
    }
    write_json(prepared / "provenance.json", provenance)
    config = json.loads(Path(__file__).with_name("baseline_config.json").read_text())
    config.update(
        protocol="amazon23-short-text-controls-v1", document_id_prefix="amazon23",
        expected_corpus={"archive_sha256": archive_hash, "documents": len(records),
                         "domains": len(DOMAINS), "per_domain_label": PER_STRATUM},
    )
    write_json(run / "baseline_config.json", config)
    write_json(run / "sample.json", {
        "documents": [{**row, "text": record["text"]} for row, record in zip(manifest, records, strict=True)],
        "archive_sha256": archive_hash,
        "inputs_sha256": hashes,
        "research_revision": RESEARCH_REVISION,
        "maps": canonical_maps(),
    })
    print(f"Frozen {len(records):,} reviews, {median(lengths):g} median XLM-R tokens, 0 TweetEval matches", flush=True)


if __name__ == "__main__":
    main()
