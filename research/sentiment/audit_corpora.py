"""Audit public sentiment archives without extracting or publishing their texts.

Only standard-library dependencies are required. This is corpus preparation,
not a sentiment experiment or a discourse-parser benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile
import zipfile


SFU_REVIEW = re.compile(r"(?P<domain>[^/]+)/(?P<label>yes|no)\d+\.txt", re.I)
CORNELL_REVIEW = re.compile(
    r"txt_sentoken/(?P<label>pos|neg)/cv(?P<cv>\d{3})_\d+\.txt"
)
RATING_CUE = re.compile(
    r"\b\d{1,2}\s*/\s*(?:5|10)\b|"
    r"\b(?:[0-5](?:\.5)?|one|two|three|four|five)\s+(?:out of\s+)?stars\b",
    re.I,
)


def decode_text(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decoding must succeed")


def document(name: str, raw: bytes, label: str, domain: str, fold=None) -> dict:
    text, encoding = decode_text(raw)
    normalized = " ".join(text.split())
    return {
        "id": name,
        "label": label,
        "domain": domain,
        "fold": fold,
        "encoding": encoding,
        "characters": len(normalized),
        "whitespace_tokens": len(normalized.split()),
        "nonempty_lines": sum(bool(line.strip()) for line in text.splitlines()),
        "rating_cue": bool(RATING_CUE.search(text)),
        "text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95_nearest_rank": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def summarize(path: Path, rows: list[dict], ignored: list[str]) -> dict:
    if not rows:
        raise ValueError(f"No matching review documents in {path}")
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["text_sha256"]].append(row)
    duplicates = [group for group in by_hash.values() if len(group) > 1]
    result = {
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_bytes": path.stat().st_size,
        "documents": len(rows),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "domain_labels": {
            domain: dict(sorted(Counter(
                row["label"] for row in rows if row["domain"] == domain
            ).items()))
            for domain in sorted({row["domain"] for row in rows})
        },
        "decode_encodings": dict(Counter(row["encoding"] for row in rows)),
        "characters": distribution([row["characters"] for row in rows]),
        "whitespace_tokens": distribution([row["whitespace_tokens"] for row in rows]),
        "nonempty_lines": distribution([row["nonempty_lines"] for row in rows]),
        "empty_documents": sum(row["characters"] == 0 for row in rows),
        "exact_duplicate_groups_after_whitespace_normalization": len(duplicates),
        "duplicate_groups_crossing_labels": sum(
            len({row["label"] for row in group}) > 1 for group in duplicates
        ),
        "rating_cue_documents_heuristic": sum(row["rating_cue"] for row in rows),
        "ignored_nonreview_members": sorted(ignored),
    }
    folds = [row for row in rows if row["fold"] is not None]
    if folds:
        result["original_fold_labels"] = {
            str(fold): dict(sorted(Counter(
                row["label"] for row in folds if row["fold"] == fold
            ).items()))
            for fold in sorted({row["fold"] for row in folds})
        }
    return result


def audit_sfu(path: Path) -> dict:
    rows, ignored = [], []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            match = SFU_REVIEW.fullmatch(member.filename)
            if match is None:
                ignored.append(member.filename)
                continue
            rows.append(document(
                member.filename,
                archive.read(member),
                "positive" if match["label"].lower() == "yes" else "negative",
                match["domain"].lower(),
            ))
    return summarize(path, rows, ignored)


def audit_cornell(path: Path) -> dict:
    rows, ignored = [], []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            match = CORNELL_REVIEW.fullmatch(member.name)
            if match is None:
                ignored.append(member.name)
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read {member.name}")
            rows.append(document(
                member.name,
                stream.read(),
                "positive" if match["label"] == "pos" else "negative",
                "movies",
                int(match["cv"]) // 100,
            ))
    return summarize(path, rows, ignored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfu", type=Path, required=True)
    parser.add_argument("--cornell", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "audit_date": args.audit_date.isoformat(),
        "method": {
            "text": "Archive members read in memory; no extraction or review text in report.",
            "length": "Whitespace tokens, not parser subword tokens or runtime estimates.",
            "identity": "SHA-256 after whitespace normalization only; not a near-duplicate audit.",
            "rating_cue": "Heuristic occurrence only; not proof of label leakage or removal guidance.",
            "labels": "Supplied polarity inferred from corpus filenames; not human reannotation.",
        },
        "sfu": audit_sfu(args.sfu),
        "cornell": audit_cornell(args.cornell),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key in ("sfu", "cornell"):
        result = report[key]
        print(json.dumps({"corpus": key, "documents": result["documents"],
                          "labels": result["labels"],
                          "lengths": result["whitespace_tokens"],
                          "duplicates": result["exact_duplicate_groups_after_whitespace_normalization"],
                          "rating_cues": result["rating_cue_documents_heuristic"]}))


if __name__ == "__main__":
    main()
