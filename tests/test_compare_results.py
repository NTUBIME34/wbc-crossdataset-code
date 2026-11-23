"""Numerical differences must remain differences, even if commands exit normally."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('compare_results', Path(__file__).resolve().parents[1] / 'reproduce/compare_results.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComparisonTests(unittest.TestCase):
    def test_exact_counts(self):
        cm = {'Neutrophil': {'Neutrophil': 100, 'Monocyte': 1}}
        result = MODULE.compare(cm, cm)
        self.assertTrue(result['exact_confusion_match'])
        self.assertEqual(result['state'], 'EXACT_COUNT_MATCH')

    def test_difference_not_hidden_by_rounding(self):
        reference = {'Neutrophil': {'Neutrophil': 100000, 'Monocyte': 1}}
        actual = {'Neutrophil': {'Neutrophil': 99999, 'Monocyte': 2}}
        result = MODULE.compare(reference, actual)
        self.assertFalse(result['exact_confusion_match'])
        self.assertEqual(result['state'], 'NUMERICAL_DIFFERENCE')
        self.assertTrue(result['three_decimal_match']['accuracy'])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.metrics({})

    def test_miss_counts_false_negative(self):
        m = MODULE.metrics({'Neutrophil': {'Neutrophil': 9, '__no_detection__': 1}})
        self.assertEqual(m['n'], 10)
        self.assertEqual(m['accuracy'], .9)
        self.assertAlmostEqual(m['per_class_f1']['Neutrophil'], 18/19)

    def test_zero_cells_normalized(self):
        result = MODULE.compare({'Monocyte': {'Monocyte': 3}}, {'Monocyte': {'Monocyte': 3, 'Basophil': 0}})
        self.assertTrue(result['exact_confusion_match'])

    def test_missed_row_cannot_be_credited_by_comparator(self):
        predictions = {'id': ['first', 'missed'], 'true': ['Basophil', 'Basophil'],
                       'pred': ['Basophil', 'Basophil'], 'status': ['detected', 'no_detection']}
        fields = {'first': {'label': 'Basophil', 'detected': True},
                  'missed': {'label': 'Basophil', 'detected': False}}
        actual = MODULE.confusion_from(predictions, fields)
        self.assertEqual(actual, {'Basophil': {'Basophil': 1, MODULE.scorer.MISS: 1}})
        result = MODULE.compare({'Basophil': {'Basophil': 2}}, actual)
        self.assertFalse(result['exact_confusion_match'])
        self.assertEqual(result['generated']['accuracy'], .5)

    def test_status_sentinel_is_normalized_to_coverage_miss(self):
        predictions = {'id': ['first', 'missed'], 'true': ['Basophil', 'Basophil'],
                       'pred': ['Basophil', 'no_detection'], 'status': ['detected', 'no_detection']}
        fields = {'first': {'label': 'Basophil', 'detected': True},
                  'missed': {'label': 'Basophil', 'detected': False}}
        actual = MODULE.confusion_from(predictions, fields)
        self.assertTrue(MODULE.compare({'Basophil': {'Basophil': 1, MODULE.scorer.MISS: 1}}, actual)['exact_confusion_match'])


if __name__ == '__main__':
    unittest.main()
