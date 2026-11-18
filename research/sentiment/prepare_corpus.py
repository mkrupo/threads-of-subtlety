"""Write reproducible local SFU review, manifest, split, and provenance files."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

if __package__:
    from . import audit_corpora, corpus
else:
    import audit_corpora
    import corpus


def _json_bytes(value, *, indent: int | None = None) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent) + "\n").encode("utf-8")


def prepare_corpus(archive: Path, output_dir: Path, seed: int = 42) -> dict:
    archive_bytes = archive.read_bytes()
    records = corpus.load_sfu(BytesIO(archive_bytes))
    splits = corpus.build_splits(records, seed)
    outer_by_id = {
        doc_id: fold["fold"]
        for fold in splits["outer_folds"]
        for doc_id in fold["test_ids"]
    }
    manifest = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"text", "text_normalized"}}
        row["outer_fold"] = outer_by_id[record["document_id"]]
        manifest.append(row)

    files = {
        "reviews.jsonl": b"".join(map(_json_bytes, records)),
        "manifest.jsonl": b"".join(map(_json_bytes, manifest)),
        "splits.json": _json_bytes(splits, indent=2),
    }
    sources = (Path(__file__), Path(corpus.__file__), Path(audit_corpora.__file__))
    provenance = {
        "schema_version": 1,
        "corpus": "SFU Review Corpus raw reviews",
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_bytes": len(archive_bytes),
        "document_count": len(records),
        "labels": {"negative": 0, "positive": 1},
        "normalization": {
            "decode_order": ["utf-8-sig", "cp1252", "latin-1"],
            "text": "Convert CRLF and CR to LF; preserve other decoded characters.",
            "text_normalized": "Join whitespace-delimited tokens with single ASCII spaces.",
            "text_sha256": "SHA-256 of UTF-8 text_normalized.",
            "document_id": "Prefix text_sha256 with sfu-; append an ordinal from the second "
                           "copy onward within a duplicate group sorted by canonical text, "
                           "then by source member for identical copies.",
        },
        "split_policy": {
            "seed": seed,
            "stratify_by": ["domain", "label"],
            "group_by": "text_sha256",
            "outer_folds": 5,
            "inner_folds": 4,
            "inner_seed": "seed + zero-based outer fold + 1",
            "fold_index_origin": 0,
        },
        "sources_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "outputs_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files["provenance.json"] = _json_bytes(provenance, indent=2)

    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError(f"Output path must be a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        if all((output_dir / name).is_file() and (output_dir / name).read_bytes() == data
               for name, data in files.items()):
            return provenance
        raise ValueError(f"Existing output differs from this preparation: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "corpus"
        staging.mkdir()
        for name, data in files.items():
            (staging / name).write_bytes(data)
        staging.rename(output_dir)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    try:
        provenance = prepare_corpus(args.archive, args.output_dir, args.seed)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Prepared {provenance['document_count']} SFU reviews in {args.output_dir}")


if __name__ == "__main__":
    main()
