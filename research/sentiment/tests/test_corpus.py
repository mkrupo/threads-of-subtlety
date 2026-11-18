from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from research.sentiment.corpus import build_splits, load_sfu, make_folds
from research.sentiment.prepare_corpus import prepare_corpus


class CorpusTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.members = {
            f"{domain}/{label}{number}.txt": f"Review {domain} {label} {number}".encode()
            for domain in ("BOOKS", "MOVIES")
            for label in ("yes", "no")
            for number in range(10)
        }

    def archive(self, members, name="fixture.zip"):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, data in members.items():
                archive.writestr(member, data)
        return path

    def test_loader_selects_reviews_and_preserves_decoded_layout(self):
        raw = {
            "BOOKS/no0.txt": b"  caf\xe9\r\nsecond\rthird\tline  ",
            "BOOKS/yes0.txt": b"\xef\xbb\xbfA review",
            "MOVIES/yes0.txt": b"\x81 latin fallback",
            "README.txt": b"ignored",
            "MOVIES/movies.zip": b"ignored",
            "MOVIES/archive/yes0.txt": b"ignored",
        }
        rows = load_sfu(self.archive(raw))
        self.assertEqual(len(rows), 3)
        by_name = {row["source_member"]: row for row in rows}
        negative = by_name["BOOKS/no0.txt"]
        self.assertEqual(negative["text"], "  café\nsecond\nthird\tline  ")
        self.assertEqual(negative["text_normalized"], "café second third line")
        self.assertEqual(negative["text_sha256"], hashlib.sha256("café second third line".encode()).hexdigest())
        self.assertEqual(negative["whitespace_tokens"], 4)
        self.assertEqual((negative["domain"], negative["label"], negative["encoding"]), ("books", 0, "cp1252"))
        self.assertEqual(by_name["BOOKS/yes0.txt"]["text"], "A review")
        self.assertEqual(by_name["BOOKS/yes0.txt"]["label"], 1)
        self.assertEqual(by_name["MOVIES/yes0.txt"]["encoding"], "latin-1")

    def test_ids_and_folds_are_deterministic(self):
        rows = load_sfu(self.archive(self.members))
        reordered = dict(reversed(list(self.members.items())))
        self.assertEqual(rows, load_sfu(self.archive(reordered, "reordered.zip")))
        self.assertEqual(make_folds(rows, 5, 42), make_folds(rows[::-1], 5, 42))
        self.assertNotEqual(make_folds(rows, 5, 42), make_folds(rows, 5, 43))

        renamed = {"MOVIES/no9.txt": b"The same text"}
        original = {"BOOKS/yes0.txt": b"The same\n text"}
        original_row = load_sfu(self.archive(original, "original.zip"))[0]
        renamed_row = load_sfu(self.archive(renamed, "renamed.zip"))[0]
        self.assertNotEqual(original_row["label"], renamed_row["label"])
        self.assertEqual(original_row["document_id"], renamed_row["document_id"])

    def test_duplicate_groups_stay_together_in_every_split(self):
        members = {**self.members, "BOOKS/yes10.txt": b"Review   BOOKS yes 0"}
        rows = load_sfu(self.archive(members))
        same_text = {
            row["document_id"] for row in rows if row["text_normalized"] == "Review BOOKS yes 0"
        }
        self.assertEqual(len(same_text), 2)
        for split in build_splits(rows)["outer_folds"]:
            overlap = same_text.intersection(split["test_ids"])
            self.assertIn(len(overlap), (0, 2))
            if not overlap:
                self.assertEqual(len({split["inner_folds"][doc_id] for doc_id in same_text}), 1)

    def test_duplicates_crossing_domain_or_label_are_rejected(self):
        for other in ("BOOKS/no1.txt", "MOVIES/yes1.txt"):
            with self.subTest(other=other):
                path = self.archive({"BOOKS/yes1.txt": b"Same text", other: b"Same \r\ntext"})
                with self.assertRaisesRegex(ValueError, "crosses domains or labels"):
                    load_sfu(path)

    def test_nested_folds_are_disjoint_exhaustive_and_balanced(self):
        rows = load_sfu(self.archive(self.members))
        strata = {row["document_id"]: (row["domain"], row["label"]) for row in rows}
        tested = Counter()
        for split in build_splits(rows)["outer_folds"]:
            train = set(split["train_ids"])
            test = set(split["test_ids"])
            inner = split["inner_folds"]
            self.assertFalse(train & test)
            self.assertEqual(train | test, set(strata))
            self.assertEqual(set(inner), train)
            self.assertEqual(Counter(strata[doc_id] for doc_id in test), dict.fromkeys(strata.values(), 2))
            for inner_fold in range(4):
                validation = {doc_id for doc_id, fold in inner.items() if fold == inner_fold}
                self.assertEqual(Counter(strata[doc_id] for doc_id in validation), dict.fromkeys(strata.values(), 2))
            tested.update(test)
        self.assertEqual(tested, dict.fromkeys(strata, 1))
        with self.assertRaisesRegex(ValueError, "needs at least 5 distinct"):
            make_folds(rows[:1], 5, 42)

    def test_preparation_preserves_matching_output_and_rejects_changes(self):
        archive = self.archive(self.members)
        output = self.root / "output"
        output.mkdir()
        provenance = prepare_corpus(archive, output)
        self.assertEqual(provenance["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(prepare_corpus(archive, output), provenance)
        for name, digest in provenance["outputs_sha256"].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
        reviews = [json.loads(line) for line in (output / "reviews.jsonl").read_text().splitlines()]
        manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len(manifest), len(reviews))
        self.assertTrue(all("text" in review for review in reviews))
        self.assertTrue(all("text" not in row and "text_normalized" not in row for row in manifest))
        outer_assignments = make_folds(reviews, 5, 42)
        self.assertEqual({row["document_id"]: row["outer_fold"] for row in manifest}, outer_assignments)
        with self.assertRaisesRegex(ValueError, "Existing output differs"):
            prepare_corpus(archive, output, seed=43)
        self.assertEqual(prepare_corpus(archive, output), provenance)


if __name__ == "__main__":
    unittest.main()
