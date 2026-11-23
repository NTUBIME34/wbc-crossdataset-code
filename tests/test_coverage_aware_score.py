#!/usr/bin/env python3
"""Offline scorer regressions: python -B -m unittest discover -s tests -v.

No model weights or datasets are required. Fixtures live in an OS temporary folder.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

SCORER_PATH = Path(__file__).resolve().parents[1] / "stats" / "coverage_aware_score.py"
SPEC = importlib.util.spec_from_file_location("coverage_aware_score", SCORER_PATH)
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wbc_scorer_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def csv_file(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def score(self, pred, fields=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return SCORER.score_seed(pred, fields)

    def test_documented_labels_and_aliases_casefold_only(self):
        expected = {name: name for name in SCORER.CLS7}
        expected.update(SCORER.ALIAS)
        for name, target in expected.items():
            for variant in (name, name.lower(), name.upper(), f" {name} "):
                with self.subTest(label=variant):
                    self.assertEqual(SCORER.canonical(variant), target)

    def test_ignore_rules_preserved_and_casefolded(self):
        for name in SCORER.IGNORE_CLASSES:
            self.assertIsNone(SCORER.canonical(name))
            self.assertIsNone(SCORER.canonical(name.lower()))
            self.assertEqual(SCORER.canonical_prediction(name), "Skipped_Class")

    def test_unknown_names_fail_not_fuzzy_matched(self):
        for name in ("basophill", "ig", "platelet", "not_a_cell", "", "Blast Group", None):
            with self.subTest(label=name), self.assertRaisesRegex(ValueError, "unsupported class label"):
                SCORER.canonical(name)

    def test_direct_scores_reject_unknown_truth(self):
        with self.assertRaisesRegex(ValueError, "unsupported true labels"):
            SCORER.scores(["basophil"], ["Basophil"])

    def test_direct_scores_reject_unknown_prediction(self):
        with self.assertRaisesRegex(ValueError, "unsupported predicted labels"):
            SCORER.scores(["Basophil"], ["basophill"])

    def test_48_native_pbc_equivalent_to_canonical(self):
        names = ["Basophil", "Eosinophil", "Erythroblast", "Lymphocyte", "Monocyte", "Neutrophil"]
        canonical, native = [], []
        for name in names:
            for i in range(8):
                row = {"source_path": f"/data/{name}/{name}_{i}.png",
                       "pred_label": SCORER.canonical(name), "det_status": "detected"}
                canonical.append(row)
                native.append({**row, "source_path": f"/data/{name.lower()}/{name}_{i}.png"})
        c = SCORER.load_predictions(self.csv_file("canonical.csv", canonical))
        n = SCORER.load_predictions(self.csv_file("native.csv", native))
        self.assertEqual(c, n)
        result = self.score(n)["as_scored"]
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 6 / 7)
        self.assertEqual(result["f1_Promyelocyte"], 0.0)

    def test_explicit_truth_columns_normalized_and_ignored(self):
        for col in ("true_label", "true"):
            with self.subTest(column=col):
                path = self.csv_file(f"explicit_{col}.csv", [
                    {"image_path": "/data/unrelated/a.png", col: "erythroblast", "pred_label": "Blast_Group"},
                    {"image_path": "/data/unrelated/b.png", col: "plasma_cell", "pred_label": "Monocyte"},
                ])
                result = SCORER.load_predictions(path)
                self.assertEqual(result["true"], ["Blast_Group"])
                self.assertEqual(result["id"], ["a"])

    def test_unknown_explicit_truth_fails_with_context(self):
        for col in ("true_label", "true"):
            path = self.csv_file(f"bad_{col}.csv", [
                {"image_path": "/data/Basophil/a.png", col: "incorrect_label", "pred_label": "Basophil"}
            ])
            with self.subTest(column=col), self.assertRaisesRegex(ValueError, "row 2.*unsupported"):
                SCORER.load_predictions(path)

    def test_unknown_folder_truth_fails(self):
        path = self.csv_file("bad_folder.csv", [
            {"source_path": "/data/not_a_cell/a.png", "pred_label": "Basophil"}
        ])
        with self.assertRaisesRegex(ValueError, "unsupported class label"):
            SCORER.load_predictions(path)

    def test_prediction_columns_normalize_supported_aliases(self):
        for col in ("pred_label", "pred", "pred_class"):
            path = self.csv_file(f"prediction_{col}.csv", [
                {"source_path": "/data/erythroblast/a.png", "true_label": "erythroblast", col: " myeloblast "},
                {"source_path": "/data/monocyte/b.png", "true_label": "monocyte", col: "plasma_cell"},
            ])
            result = SCORER.load_predictions(path)
            self.assertEqual(result["pred"], ["Blast_Group", "Skipped_Class"])
            self.assertEqual(self.score(result)["as_scored"]["accuracy"], 0.5)

    def test_prediction_columns_reject_unknown_names(self):
        for col in ("pred_label", "pred", "pred_class"):
            path = self.csv_file(f"bad_prediction_{col}.csv", [
                {"source_path": "/data/Basophil/a.png", col: "basophill"},
            ])
            with self.subTest(column=col), self.assertRaisesRegex(ValueError, "row 2.*unsupported"):
                SCORER.load_predictions(path)

    def test_prediction_sentinels_preserved(self):
        for name in SCORER.PREDICTION_SENTINELS:
            self.assertEqual(SCORER.canonical_prediction(name), name)
            self.assertEqual(SCORER.canonical_prediction(name.upper()), name)

    def test_labels_csv_normalizes_and_missing_entries_remain_unlabelled(self):
        path = self.csv_file("flat.csv", [
            {"source_path": "/data/flat/a.png", "pred_label": "Basophil"},
            {"source_path": "/data/flat/b.png", "pred_label": "Monocyte"},
        ])
        labels = self.csv_file("labels.csv", [{"image_id": "a", "true_label": "basophil"}])
        result = SCORER.load_predictions(path, SCORER.load_labels(labels))
        self.assertEqual(result["true"], ["Basophil"])
        self.assertEqual(result["id"], ["a"])

    def test_label_file_unknown_entries_fail_even_if_not_predicted(self):
        labels = self.csv_file("bad_labels.csv", [{"image_id": "a", "true_label": "not_a_cell"}])
        with self.assertRaisesRegex(ValueError, "row 2.*unsupported"):
            SCORER.load_labels(labels)

    def test_label_file_rejects_conflicting_duplicate_ids(self):
        labels = self.csv_file("conflicting_labels.csv", [
            {"image_path": "/data/Basophil/a.jpg", "true_label": "Basophil"},
            {"image_path": "/data/Monocyte/a.png", "true_label": "Monocyte"},
        ])
        with self.assertRaisesRegex(ValueError, "conflicting truth labels"):
            SCORER.load_labels(labels)

    def test_label_file_accepts_equivalent_duplicate_aliases(self):
        labels = self.csv_file("consistent_labels.csv", [
            {"image_id": "a", "true_label": "Erythroblast"},
            {"image_id": "a", "true_label": "blast_group"},
        ])
        self.assertEqual(SCORER.canonical(SCORER.load_labels(labels)["a"]), "Blast_Group")

    def test_detection_truth_validated(self):
        path = self.csv_file("bad_detection.csv", [
            {"source_path": "/data/not_a_cell/a.png", "status": "no_detection"}
        ])
        with self.assertRaisesRegex(ValueError, "unsupported class label"):
            SCORER.load_detections(path, None)
        with self.assertRaisesRegex(ValueError, "unsupported class label"):
            SCORER.load_detections(path, {"a": "not_a_cell"})
        self.assertEqual(SCORER.load_detections(path, {"a": "basophil"})["a"]["label"], "Basophil")
        self.assertIsNone(SCORER.load_detections(path, {})["a"]["label"])

    def test_detection_id_collision_with_conflicting_truth_fails(self):
        path = self.csv_file("detection_collision.csv", [
            {"source_path": "/data/Basophil/a.png", "status": "detected"},
            {"source_path": "/data/Monocyte/a.png", "status": "detected"},
        ])
        with self.assertRaisesRegex(ValueError, "conflicting truth labels"):
            SCORER.load_detections(path, None)

    def test_multiple_prediction_rows_per_field_fail(self):
        path = self.csv_file("duplicate_predictions.csv", [
            {"source_path": "/data/Basophil/a__det00.png", "pred_label": "Basophil"},
            {"source_path": "/data/Basophil/a__det01.png", "pred_label": "Basophil"},
        ])
        with self.assertRaisesRegex(SystemExit, "more than one row per field"):
            SCORER.load_predictions(path)

    def test_no_detection_and_ignored_rows_keep_metric_definition(self):
        pred_rows = [
            {"source_path": "/data/basophil/a.png", "pred_label": "Basophil", "det_status": "detected"},
            {"source_path": "/data/monocyte/b.png", "pred_label": "no_detection", "det_status": "no_detection"},
            {"source_path": "/data/plasma_cell/c.png", "pred_label": "no_detection", "det_status": "no_detection"},
        ]
        det_rows = [{"source_path": r["source_path"], "status": r["det_status"]} for r in pred_rows]
        pred = SCORER.load_predictions(self.csv_file("miss_predictions.csv", pred_rows))
        fields = SCORER.load_detections(self.csv_file("miss_detections.csv", det_rows), None)
        result = self.score(pred, fields)
        self.assertEqual(result["coverage"]["labelled"], 2)
        self.assertEqual(result["coverage"]["missed"], 1)
        self.assertEqual(result["coverage"]["missed_unlabelled"], 1)
        self.assertEqual(result["coverage_aware"]["n"], 2)
        self.assertEqual(result["coverage_aware"]["accuracy"], 0.5)
        self.assertAlmostEqual(result["coverage_aware"]["macro_f1"], 1 / 7)

    def test_nine_class_prediction_alias_and_missing_sentinel(self):
        path = self.csv_file("legacy_predictions.csv", [
            {"source_path": "/data/Erythroblast/a.png", "pred_class": "Myeloblast", "det_status": "detected"},
            {"source_path": "/data/Monocyte/b.png", "pred_class": "read_failed", "det_status": "read_failed"},
        ])
        self.assertEqual(SCORER.load_predictions(path), {
            "id": ["a", "b"], "true": ["Blast_Group", "Monocyte"],
            "pred": ["Blast_Group", "read_failed"], "status": ["detected", "read_failed"],
        })

    def test_missing_detected_prediction_fails(self):
        pred = {"id": ["a"], "true": ["Basophil"], "pred": ["Basophil"], "status": ["detected"]}
        fields = {
            "a": {"label": "Basophil", "detected": True},
            "b": {"label": "Monocyte", "detected": True},
        }
        with self.assertRaisesRegex(ValueError, "no prediction row"):
            self.score(pred, fields)

    def test_prediction_absent_from_detections_fails(self):
        pred = {"id": ["a", "b"], "true": ["Basophil", "Monocyte"],
                "pred": ["Basophil", "Monocyte"], "status": ["detected", "detected"]}
        fields = {"a": {"label": "Basophil", "detected": True}}
        with self.assertRaisesRegex(ValueError, "no field in the detections file"):
            self.score(pred, fields)

    def test_prediction_detection_truth_disagreement_fails(self):
        pred = {"id": ["a"], "true": ["Basophil"], "pred": ["Basophil"], "status": ["detected"]}
        fields = {"a": {"label": "Monocyte", "detected": True}}
        with self.assertRaisesRegex(ValueError, "conflicting truth labels"):
            self.score(pred, fields)

    def test_failed_classification_not_silently_dropped(self):
        pred = {"id": ["a"], "true": ["Basophil"], "pred": ["read_failed"], "status": ["read_failed"]}
        fields = {"a": {"label": "Basophil", "detected": True}}
        with self.assertRaisesRegex(ValueError, "no usable classification"):
            self.score(pred, fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
