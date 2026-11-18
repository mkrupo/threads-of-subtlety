"""Independently verify the SFU v1 text-control artifacts using the standard library."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from math import isclose
from pathlib import Path
import re
import zipfile


SOURCE_DIR = Path(__file__).resolve().parent
CONTROLS = ("class_prior", "length_only", "tfidf_lr")
PREPARED_FILES = {"reviews.jsonl", "manifest.jsonl", "splits.json"}
ARCHIVE_SHA256 = "3c53884a218c6c218b128440a004de8168d549ebd911277a47c8e7fac28c03ac"
REVIEW_MEMBER = re.compile(r"(?P<domain>[^/]+)/(?P<label>yes|no)\d+\.txt", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def verify_hashes(directory, recorded, expected):
    require(set(recorded) == expected, f"Unexpected checksum inventory for {directory}")
    for name, value in recorded.items():
        path = directory / name
        require(digest(path) == value, f"Checksum mismatch for {path}")


def verify_archive(archive, provenance, reviews):
    archive_bytes = archive.read_bytes()
    require(sha256(archive_bytes).hexdigest() == provenance["archive_sha256"], "Raw archive checksum mismatch")
    require(len(archive_bytes) == provenance["archive_bytes"], "Raw archive size mismatch")
    with zipfile.ZipFile(BytesIO(archive_bytes)) as source:
        members = [m.filename for m in source.infolist() if REVIEW_MEMBER.fullmatch(m.filename)]
        require(Counter(members) == Counter(row["source_member"] for row in reviews.values()),
                "Raw archive review membership mismatch")
        for doc_id, row in reviews.items():
            raw = source.read(row["source_member"])
            for encoding in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    text = raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
                except UnicodeDecodeError:
                    continue
                break
            require(text == row["text"] and encoding == row["encoding"],
                    f"Raw review text or encoding mismatch for {doc_id}")


def verify_splits(prepared, reviews, metadata):
    ids = set(reviews)
    strata = {doc_id: (row["domain"], row["label"]) for doc_id, row in reviews.items()}
    require(len(set(strata.values())) == 16 and set(Counter(strata.values()).values()) == {25},
            "Unexpected domain-label distribution")
    expected_balance = dict.fromkeys(strata.values(), 5)
    folds = read_json(prepared / "splits.json")["outer_folds"]
    require(sorted(fold["fold"] for fold in folds) == list(range(5)), "Invalid outer fold indices")
    tested = Counter()
    for fold in folds:
        index = fold["fold"]
        train, test = set(fold["train_ids"]), set(fold["test_ids"])
        inner = fold["inner_folds"]
        require(len(train) == len(fold["train_ids"]) == 320 and len(test) == len(fold["test_ids"]) == 80,
                f"Invalid train or test size in outer fold {index}")
        require(train | test == ids and not train & test and set(inner) == train,
                f"Invalid train, test or inner membership in outer fold {index}")
        require(all(metadata[doc_id]["outer_fold"] == index for doc_id in test),
                f"Manifest membership mismatch in outer fold {index}")
        require(Counter(strata[i] for i in test) == expected_balance,
                f"Outer fold {index} is imbalanced")
        require(set(inner.values()) == set(range(4)), f"Invalid inner fold indices for outer fold {index}")
        for k in range(4):
            require(Counter(strata[i] for i in inner if inner[i] == k) == expected_balance,
                    f"Inner fold {k} is imbalanced for outer fold {index}")
        tested.update(test)
    require(tested == dict.fromkeys(ids, 1), "Each review must appear in exactly one outer test fold")


def checked_metrics(rows, reported, context):
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        matrix[row["label"]][row["prediction"]] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    metrics = {
        "accuracy": (tn + tp) / len(rows),
        "negative_f1": negative,
        "positive_f1": positive,
        "macro_f1": (negative + positive) / 2,
    }
    require(matrix == reported["confusion_matrix_labels_0_1"] and len(rows) == reported["documents"],
            f"Confusion matrix or document count mismatch for {context}")
    for key, value in metrics.items():
        require(isclose(value, reported[key], rel_tol=1e-13, abs_tol=1e-13),
                f"{key} mismatch for {context}")
    return {"confusion_matrix_labels_0_1": matrix, **metrics}


def verify_predictions(run, metadata, manifest):
    summary = read_json(run / "summary.json")
    require(summary["protocol"] == "sfu-text-controls-v1" and summary["label_names"] == {"0": "negative", "1": "positive"},
            "Summary protocol or label names mismatch")
    predictions = read_jsonl(run / "predictions.jsonl")
    require(len(predictions) == manifest["predictions"] == 1200, "Unexpected prediction count")
    groups = defaultdict(list)
    for row in predictions:
        doc_id = row["document_id"]
        require(doc_id in metadata, f"Unknown prediction ID {doc_id}")
        expected = metadata[doc_id]
        require((row["fold"], row["domain"], row["label"]) ==
                (expected["outer_fold"], expected["domain"], expected["label"]),
                f"Prediction fold, domain or label mismatch for {doc_id}")
        probability = row["probability_positive"]
        threshold_prediction = int(probability > 0.5 if row["control"] == "class_prior" else probability >= 0.5)
        require(0 <= probability <= 1 and row["prediction"] == threshold_prediction,
                f"Invalid prediction or probability for {doc_id}")
        groups[row["control"]].append(row)
    require(set(groups) == set(summary["controls"]) == set(CONTROLS), "Unexpected controls")
    pooled = {}
    for control, rows in groups.items():
        require(len(rows) == len(metadata) and {row["document_id"] for row in rows} == set(metadata),
                f"Missing or repeated out-of-fold prediction IDs for {control}")
        reported = summary["controls"][control]
        pooled[control] = checked_metrics(rows, reported["pooled_out_of_fold"], control)
        for field in ("fold", "domain"):
            partitions = defaultdict(list)
            for row in rows:
                partitions[str(row[field])].append(row)
            breakdown = reported[f"per_{field}"]
            require(set(partitions) == set(breakdown), f"Unexpected {field} entries for {control}")
            for key, subset in partitions.items():
                checked_metrics(subset, breakdown[key], f"{control}, {field}={key}")
    return pooled


def verify(prepared, run, archive):
    provenance = read_json(prepared / "provenance.json")
    manifest = read_json(run / "run_manifest.json")
    require(provenance["archive_sha256"] == ARCHIVE_SHA256, "Unexpected SFU raw archive identity")
    require(provenance["schema_version"] == manifest["schema_version"] == 1 and manifest["status"] == "complete",
            "Unexpected provenance schema or incomplete run")
    verify_hashes(prepared, provenance["outputs_sha256"], PREPARED_FILES)
    verify_hashes(SOURCE_DIR, provenance["sources_sha256"], {"audit_corpora.py", "corpus.py", "prepare_corpus.py"})
    verify_hashes(SOURCE_DIR, manifest["source_hashes"], {"baseline.py", "corpus.py", "prepare_corpus.py", "pyproject.toml", "uv.lock"})
    inputs = dict(manifest["input_hashes"])
    verify_hashes(SOURCE_DIR, {"baseline_config.json": inputs.pop("baseline_config.json")}, {"baseline_config.json"})
    verify_hashes(prepared, inputs, PREPARED_FILES | {"provenance.json"})
    model_files = {f"fold-{i}-{control}.joblib" for i in range(5) for control in CONTROLS}
    verify_hashes(run, manifest["artifact_hashes"],
                  model_files | {"predictions.jsonl", "protocol.json", "selection.json", "summary.json"})
    require(read_json(run / "protocol.json") == read_json(SOURCE_DIR / "baseline_config.json"),
            "Saved protocol differs from the input configuration")

    review_rows = read_jsonl(prepared / "reviews.jsonl")
    manifest_rows = read_jsonl(prepared / "manifest.jsonl")
    reviews = {row["document_id"]: row for row in review_rows}
    metadata = {row["document_id"]: row for row in manifest_rows}
    require(len(reviews) == len(metadata) == len(review_rows) == len(manifest_rows)
            == provenance["document_count"] == manifest["documents"] == 400,
            "Unexpected prepared review count or repeated IDs")
    require(set(reviews) == set(metadata), "Review and manifest document IDs differ")
    for doc_id, row in reviews.items():
        member = REVIEW_MEMBER.fullmatch(row["source_member"])
        require(member and row["domain"] == member["domain"].lower()
                and row["label"] == int(member["label"].lower() == "yes"),
                f"Source member metadata mismatch for {doc_id}")
        tokens = row["text"].split()
        normalized = " ".join(tokens)
        digest_text = sha256(normalized.encode("utf-8")).hexdigest()
        require(tokens and "\r" not in row["text"] and row["text_normalized"] == normalized
                and row["text_sha256"] == digest_text and doc_id == f"sfu-{digest_text}"
                and row["whitespace_tokens"] == len(tokens),
                f"Review text normalization, length or ID mismatch for {doc_id}")
        require({k: v for k, v in row.items() if k not in {"text", "text_normalized"}}
                == {k: v for k, v in metadata[doc_id].items() if k != "outer_fold"},
                f"Review manifest metadata mismatch for {doc_id}")

    if archive is not None:
        verify_archive(archive, provenance, reviews)
    verify_splits(prepared, reviews, metadata)
    pooled = verify_predictions(run, metadata, manifest)
    return {
        "status": "verified",
        "protocol": "sfu-text-controls-v1",
        "verifier_sha256": digest(Path(__file__)),
        "archive_sha256": provenance["archive_sha256"],
        "raw_archive_verified": archive is not None,
        "run_manifest_sha256": digest(run / "run_manifest.json"),
        "input_hashes": manifest["input_hashes"],
        "documents": len(reviews),
        "predictions": manifest["predictions"],
        "outer_folds": 5,
        "inner_folds_per_outer_fold": 4,
        "reviews_per_domain_label_per_validation_or_test_fold": 5,
        "pooled_out_of_fold": pooled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.prepared_dir, args.run_dir, args.archive)
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{parser.prog}: verification failed: {error}\n")


if __name__ == "__main__":
    main()
