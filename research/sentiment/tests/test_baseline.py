from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from research.sentiment import baseline
from research.sentiment.corpus import build_splits, make_folds
from research.sentiment.edu_baseline import mean_edu_scores
from research.sentiment.nuclearity import pool_nuclearity
from research.sentiment.prepare_corpus import prepare_corpus


class RecordingVectorizer(TfidfVectorizer):
    fitting_sizes = []

    def fit_transform(self, raw_documents, y=None):
        documents = list(raw_documents)
        type(self).fitting_sizes.append(len(documents))
        return super().fit_transform(documents, y)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((baseline.HERE / "baseline_config.json").read_text())
        self.records = []
        for domain in ("books", "movies"):
            for label in (0, 1):
                for index in range(10):
                    text = f"shared review {'pleasant' if label else 'awful'} sample {domain} {index}"
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    self.records.append({"document_id": digest, "domain": domain,
                                         "label": label, "text": text, "text_sha256": digest})

    def test_nested_validation_rejects_leaked_or_repeated_ids(self):
        splits = build_splits(self.records)
        self.assertEqual(len(baseline.validate_inputs(self.records, splits, self.config)), 40)
        for mutation in ("test_in_train", "test_in_inner", "repeated_test"):
            corrupted = deepcopy(splits)
            fold = corrupted["outer_folds"][0]
            if mutation == "test_in_train":
                fold["train_ids"].append(fold["test_ids"][0])
            elif mutation == "test_in_inner":
                fold["inner_folds"][fold["test_ids"][0]] = 0
            else:
                fold["test_ids"].append(fold["test_ids"][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                baseline.validate_inputs(self.records, corrupted, self.config)

    def test_vocabulary_is_fitted_in_inner_training_and_excludes_outer_test(self):
        splits = build_splits(self.records)
        fold = splits["outer_folds"][0]
        by_id = {row["document_id"]: row for row in self.records}
        training = [by_id[name] for name in sorted(fold["train_ids"])]
        inner = [fold["inner_folds"][row["document_id"]] for row in training]
        RecordingVectorizer.fitting_sizes = []
        with patch.object(baseline, "TfidfVectorizer", RecordingVectorizer):
            fitted, _ = baseline.fit_control("tfidf_lr", training, inner, self.config)
        counts = RecordingVectorizer.fitting_sizes
        # Three C candidates each fit four inner training subsets, then one refit.
        self.assertEqual(counts.count(24), 12)
        self.assertEqual(counts.count(32), 1)
        test_row = {"text": "outertestonlytoken outertestonlytoken", "domain": "unused",
                    "label": 1, "document_id": "unused"}
        baseline.classify(fitted, "tfidf_lr", [test_row], self.config)
        self.assertNotIn("outertestonlytoken", fitted.named_steps["tfidf"].vocabulary_)
        self.assertEqual(counts, RecordingVectorizer.fitting_sizes)

    def test_features_do_not_include_ids_domains_or_labels(self):
        original = self.records[0]
        altered = {**original, "document_id": "secret-positive", "domain": "changed",
                   "label": 1, "source_member": "positive/filename.txt"}
        self.assertEqual(baseline.feature_input("tfidf_lr", [original]),
                         baseline.feature_input("tfidf_lr", [altered]))
        np.testing.assert_array_equal(baseline.feature_input("length_only", [original]),
                                      baseline.feature_input("length_only", [altered]))

    def test_edu_mean_gives_equal_weight_to_edus_across_unequal_scenes(self):
        rows = [
            {"document_id": "a", "probabilities": [[0, 0, 1], [0, 1, 0]]},
            {"document_id": "b", "probabilities": [[0, 1, 0]]},
            {"document_id": "a", "probabilities": [[1, 0, 0], [1, 0, 0]]},
            {"document_id": "a", "probabilities": [[1, 0, 0]]},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            means = mean_edu_scores(path)
        self.assertEqual(set(means), {"a", "b"})
        np.testing.assert_allclose(means["a"], [0.6, 0.2, 0.2])
        np.testing.assert_allclose(means["b"], [0, 1, 0])

    def test_nuclearity_pooling_respects_ancestors_across_scenes(self):
        nodes = {
            "6": {"left": 4, "right": 5, "nuclearity": "NS"},
            "4": {"left": 0, "right": 1, "nuclearity": "NN"},
            "5": {"left": 2, "right": 3, "nuclearity": "SN"},
            **{str(index): {"text": "EDU"} for index in range(4)},
        }
        trees = [
            {"key": "a0", "document_id": "a", "status": "ok", "root": 6, "nodes": nodes},
            {"key": "a1", "document_id": "a", "status": "ok", "root": 0, "nodes": {"0": nodes["0"]}},
            {"key": "b0", "document_id": "b", "status": "ok", "root": 0, "nodes": {"0": nodes["0"]}},
        ]
        scores = [
            {"key": tree["key"], "document_id": tree["document_id"], "probabilities": probabilities}
            for tree, probabilities in zip(trees, [
                [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]], [[0, 0, 1]], [[0, 1, 0]],
            ], strict=True)
        ]
        features, coverage = pool_nuclearity(scores, trees)
        np.testing.assert_allclose(features["a"], [0, 1 / 3, 2 / 3, 1, 0, 0])
        np.testing.assert_allclose(features["b"], [0, 1, 0, 0, 1, 0])
        self.assertEqual(coverage["reviews_without_supporting_edus"], 1)

    def test_jsonl_handles_unicode_separators_inside_review_text(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            record = {"text": "first\u2028second\u0085third", "label": 1}
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(baseline.read_jsonl(path), [record])

    def test_prior_control_uses_training_labels_and_explicit_tie_rule(self):
        training = self.records[:10] + self.records[10:20]
        fitted, _ = baseline.fit_control("class_prior", training, [], self.config)
        probability, predicted = baseline.classify(fitted, "class_prior", training[:2], self.config)
        np.testing.assert_array_equal(probability, [0.5, 0.5])
        np.testing.assert_array_equal(predicted, [0, 0])

    def test_domain_held_out_folds_do_not_pass_as_stratified_review_folds(self):
        records = []
        for domain in range(5):
            for label in (0, 1):
                for index in range(10):
                    text = f"review {domain} {label} {index}"
                    identity = hashlib.sha256(text.encode()).hexdigest()
                    records.append({"document_id": identity, "text_sha256": identity,
                                    "text": text, "domain": str(domain), "label": label})
        folds = []
        for domain in range(5):
            training = [r for r in records if r["domain"] != str(domain)]
            folds.append({"fold": domain,
                          "train_ids": [r["document_id"] for r in training],
                          "test_ids": [r["document_id"] for r in records if r["domain"] == str(domain)],
                          "inner_folds": make_folds(training, 4, 42 + domain + 1)})
        with self.assertRaisesRegex(ValueError, "domain-by-label stratification"):
            baseline.validate_inputs(records, {"outer_folds": folds}, self.config)

    def prepare_fixture(self, root):
        archive = root / "reviews.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            for domain in ("BOOKS", "MOVIES"):
                for label in ("no", "yes"):
                    for index in range(10):
                        stream.writestr(f"{domain}/{label}{index}.txt",
                                        f"A shared review {'awful' if label == 'no' else 'pleasant'} "
                                        f"unique {domain} {index}")
        prepared = root / "prepared"
        provenance = prepare_corpus(archive, prepared)
        config = deepcopy(self.config)
        config["expected_corpus"] = {
            "archive_sha256": provenance["archive_sha256"], "documents": 40,
            "domains": 2, "per_domain_label": 10,
        }
        return prepared, config

    def test_prepared_hash_and_semantic_consistency_are_checked(self):
        for mutation in ("bytes", "text", "metadata"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                prepared, config = self.prepare_fixture(Path(directory))
                baseline.validate_prepared(prepared, config)
                filename = "manifest.jsonl" if mutation == "metadata" else "reviews.jsonl"
                target = prepared / filename
                if mutation == "bytes":
                    target.write_bytes(target.read_bytes() + b"\n")
                else:
                    rows = baseline.read_jsonl(target)
                    if mutation == "text":
                        rows[0]["text"] += " extra words"
                    else:
                        rows[0]["label"] = 1 - rows[0]["label"]
                    target.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                    provenance_path = prepared / "provenance.json"
                    provenance = json.loads(provenance_path.read_text())
                    provenance["outputs_sha256"][filename] = baseline.digest(target)
                    baseline.write_json(provenance_path, provenance)
                with self.assertRaises(ValueError):
                    baseline.validate_prepared(prepared, config)

    def test_complete_fixture_run_verifies_and_detects_changed_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, config = self.prepare_fixture(root)
            config_path = root / "config.json"
            baseline.write_json(config_path, config)
            output = root / "run"
            result = baseline.run(prepared, output, config_path)
            baseline.verify(prepared, output, config_path)
            self.assertEqual(set(result["controls"]), set(config["controls"]))
            predictions = baseline.read_jsonl(output / "predictions.jsonl")
            self.assertEqual(len(predictions), 120)
            summary = output / "summary.json"
            summary.write_bytes(summary.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Changed artifact"):
                baseline.verify(prepared, output, config_path)


if __name__ == "__main__":
    unittest.main()
