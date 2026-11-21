"""Check all selected bodies and timestamps against the four source archives."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import unicodedata

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--run-dir", required=True, type=Path)
run = cli.parse_args().run_dir
by_archive = defaultdict(dict)
with (run / "prepared/reviews.jsonl").open(encoding="utf-8") as rows:
    for line in rows:
        row = json.loads(line)
        name, number = row["source_member"].rsplit(":", 1)
        if int(number) in by_archive[name]:
            raise ValueError("Duplicate selected source line")
        by_archive[name][int(number)] = row

def norm(text):
    normalized = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized))

with (run / "tweeteval-sentiment-train.txt").open(encoding="utf-8") as source:
    training = {norm(line) for line in source if line.strip()}

body_matches = 0
timestamps = []
identities = {"body": set(), "product": set(), "reviewer": set()}
for name, items in sorted(by_archive.items()):
    with gzip.open(run / name, "rt", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if number not in items:
                continue
            original = json.loads(line)
            row = items[number]
            body = re.sub(r"<br\b\s*/?\s*>", "\n", html.unescape(original["text"]), flags=re.I)
            body = body.replace("\r\n", "\n").replace("\r", "\n")
            timestamp = datetime.fromtimestamp(original["timestamp"] / 1000, timezone.utc)
            if (body != row["text"] or timestamp.isoformat() != row["timestamp_utc"]
                    or timestamp < datetime(2022, 1, 1, tzinfo=timezone.utc)
                    or original["rating"] != row["rating"] or original["rating"] not in (1, 2, 4, 5)
                    or int(original["rating"] > 3) != row["label"]):
                raise ValueError(f"Source mismatch in {name}, line {number}")
            for field, value, metadata in (
                ("body", norm(body), "text_sha256"),
                ("reviewer", original["user_id"], "reviewer_sha256"),
                ("product", original["parent_asin"], "product_sha256"),
            ):
                hash_input = " ".join(body.split()) if field == "body" else value
                if value in identities[field] or sha256(hash_input.encode()).hexdigest() != row[metadata]:
                    raise ValueError(f"Duplicate or invalid {field} in {name}, line {number}")
                identities[field].add(value)
            if norm(body) in training:
                raise ValueError("Selected body matches a TweetEval sentiment training text")
            timestamps.append(timestamp)
            body_matches += 1
            if number == max(items):
                break
if body_matches != sum(map(len, by_archive.values())):
    raise ValueError("Some selected rows were not found in the raw sources")
print(json.dumps({
    "raw_rows_verified": body_matches,
    "tweet_train_unique_normalized": len(training),
    "normalized_full_review_overlap": 0,
    "earliest_review": min(timestamps).isoformat(),
    "latest_review": max(timestamps).isoformat(),
    "unique_texts_reviewers_products": {key: len(values) for key, values in identities.items()},
}, indent=2))
