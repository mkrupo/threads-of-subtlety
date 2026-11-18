"""Load SFU reviews and assign fixed, duplicate-aware sentiment folds."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import BinaryIO
import zipfile

if __package__:
    from .audit_corpora import SFU_REVIEW, decode_text
else:
    from audit_corpora import SFU_REVIEW, decode_text


def _group_reviews(records: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for record in records:
        groups[record["text_sha256"]].append(record)
    for digest, group in groups.items():
        if len({(record["domain"], record["label"]) for record in group}) > 1:
            raise ValueError(f"Duplicate review text crosses domains or labels: {digest}")
    return groups


def load_sfu(archive_path: str | Path | BinaryIO) -> list[dict]:
    """Read reviews directly from the ZIP and return rows sorted by document ID."""
    records = []
    seen_members = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            match = SFU_REVIEW.fullmatch(member.filename)
            if not match:
                continue
            if member.filename in seen_members:
                raise ValueError(f"Repeated ZIP member: {member.filename}")
            seen_members.add(member.filename)
            text, encoding = decode_text(archive.read(member))
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            normalized = " ".join(text.split())
            if not normalized:
                raise ValueError(f"Empty review: {member.filename}")
            records.append({
                "source_member": member.filename,
                "domain": match["domain"].lower(),
                "label": int(match["label"].lower() == "yes"),
                "text": text,
                "text_normalized": normalized,
                "text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                "encoding": encoding,
                "whitespace_tokens": len(normalized.split(" ")),
            })
    if not records:
        raise ValueError(f"No SFU review documents in {archive_path}")

    for digest, group in _group_reviews(records).items():
        ordered = sorted(group, key=lambda row: (row["text"], row["source_member"]))
        for index, record in enumerate(ordered):
            # Preserve multiple copies without putting polarity in the ID.
            record["document_id"] = f"sfu-{digest}" + (f"-{index + 1}" if index else "")
    return sorted(records, key=lambda row: row["document_id"])


def make_folds(records: list[dict], n_splits: int, seed: int) -> dict[str, int]:
    """Assign whole duplicate groups to zero-based domain-by-label folds."""
    if n_splits < 2:
        raise ValueError("At least two folds are required")
    if not records:
        raise ValueError("Cannot split an empty corpus")
    if len({record["document_id"] for record in records}) != len(records):
        raise ValueError("Document IDs must be unique")

    strata = defaultdict(list)
    for digest, group in _group_reviews(records).items():
        stratum = (group[0]["domain"], group[0]["label"])
        shuffle_key = hashlib.sha256(f"{seed}:{digest}".encode("ascii")).digest()
        strata[stratum].append((len(group), shuffle_key, digest, group))

    assignments = {}
    total_counts = [0] * n_splits
    for stratum, groups in sorted(strata.items()):
        if len(groups) < n_splits:
            raise ValueError(f"Stratum {stratum} needs at least {n_splits} distinct review texts")
        stratum_counts = [0] * n_splits
        # Place larger groups first. Equal-size groups follow their seeded hash order.
        for size, _, _, group in sorted(groups, key=lambda item: (-item[0], item[1], item[2])):
            fold = min(range(n_splits), key=lambda i: (stratum_counts[i], total_counts[i], i))
            for record in group:
                assignments[record["document_id"]] = fold
            stratum_counts[fold] += size
            total_counts[fold] += size
    return dict(sorted(assignments.items()))


def build_splits(records: list[dict], seed: int = 42) -> dict:
    """Freeze five outer folds and four training-only inner folds for each."""
    outer_assignments = make_folds(records, 5, seed)
    outer_folds = []
    for fold in range(5):
        train = [row for row in records if outer_assignments[row["document_id"]] != fold]
        outer_folds.append({
            "fold": fold,
            "train_ids": sorted(row["document_id"] for row in train),
            "test_ids": [doc_id for doc_id in outer_assignments if outer_assignments[doc_id] == fold],
            "inner_folds": make_folds(train, 4, seed + fold + 1),
        })
    return {"outer_folds": outer_folds}
