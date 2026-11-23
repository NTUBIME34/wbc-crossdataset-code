import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('internal_evaluator', ROOT / 'eval/evaluate_internal.py')
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class InternalTransformTests(unittest.TestCase):
    @staticmethod
    def image():
        values = np.arange(29 * 43 * 3, dtype=np.uint8).reshape(29, 43, 3)
        return Image.fromarray(values)

    def test_classifier_meta_uses_actual_native_transform(self):
        transform, args, protocol = EVAL.transform_from_trainer('classifier', dict(recipe='meta', img_size=32))
        trainer = EVAL.import_file(ROOT / 'train/train_classifier.py', '_internal_classifier')
        expected = trainer.build_transforms(args)[1]
        self.assertTrue(torch.equal(transform(self.image()), expected(self.image())))
        self.assertTrue(torch.equal(transform(self.image()), transform(self.image())))
        self.assertEqual(protocol['resize'], 35)
        self.assertTrue(protocol['square_padding'])
        self.assertEqual(protocol['tta'], 'none')

    def test_classifier_paper_does_not_add_square_padding(self):
        transform, args, protocol = EVAL.transform_from_trainer('classifier', dict(recipe='paper', img_size=32))
        trainer = EVAL.import_file(ROOT / 'train/train_classifier.py', '_internal_classifier')
        expected = trainer.build_transforms(args)[1]
        self.assertTrue(torch.equal(transform(self.image()), expected(self.image())))
        self.assertFalse(protocol['square_padding'])
        self.assertEqual(protocol['resize'], 36)

    def test_cnn_all_saved_padding_modes(self):
        for mode in ('black', 'reflect', 'replicate'):
            with self.subTest(mode=mode):
                transform, args, protocol = EVAL.transform_from_trainer('cnn', dict(img_size=32, pad_mode=mode))
                trainer = EVAL.import_file(ROOT / 'train/train_cnn_baseline.py', '_internal_cnn')
                expected = trainer.build_transforms(img_size=32, pad_mode=mode)[1]
                self.assertTrue(torch.equal(transform(self.image()), expected(self.image())))
                self.assertEqual(protocol['pad_mode'], mode)

    def test_dinobloom_native_transform(self):
        transform, args, protocol = EVAL.transform_from_trainer('dinobloom', dict(recipe='meta', img_size=32))
        trainer = EVAL.import_file(ROOT / 'train/train_dinobloom.py', '_internal_dinobloom')
        self.assertTrue(torch.equal(transform(self.image()), trainer.build_transforms(args)[1](self.image())))
        self.assertEqual(protocol['resize'], 35)

    def test_saved_transform_arguments_are_validated(self):
        for family, args in [('classifier', {'recipe': 'unknown'}), ('cnn', {'pad_mode': 'unknown'}),
                             ('classifier', {'img_size': 0}), ('classifier', {'img_size': True})]:
            with self.subTest(family=family, args=args):
                with self.assertRaises(ValueError):
                    EVAL.transform_from_trainer(family, args)


class InternalInputTests(unittest.TestCase):
    def test_duplicate_and_unsafe_classes_rejected(self):
        for classes in ([], ['A', 'A'], ['../A'], ['.'], ['a/b'], ['a\\b']):
            with self.subTest(classes=classes):
                with self.assertRaises(ValueError):
                    EVAL.validate_classes(classes)

    def test_checkpoint_class_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('A', 'B'):
                (root / name).mkdir()
                Image.new('RGB', (7, 9)).save(root / name / 'sample.png')
            dataset = EVAL.ExactSplitDataset(root, ['B', 'A'], lambda im: im.size)
            self.assertEqual(dataset[0], ((7, 9), 0, 'B/sample.png'))
            self.assertEqual(dataset[1], ((7, 9), 1, 'A/sample.png'))

    def test_unknown_missing_and_empty_classes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'A').mkdir()
            with self.assertRaises(ValueError):
                EVAL.ExactSplitDataset(root, ['A', 'B'], lambda x: x)
            with self.assertRaises(ValueError):
                EVAL.ExactSplitDataset(root, ['A'], lambda x: x)
            Image.new('RGB', (4, 5)).save(root / 'A' / 'sample.png')
            with self.assertRaises(ValueError):
                EVAL.ExactSplitDataset(root, ['B'], lambda x: x)

    def test_corrupt_image_not_replaced_with_blank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'A').mkdir()
            (root / 'A' / 'broken.png').write_bytes(b'not an image')
            dataset = EVAL.ExactSplitDataset(root, ['A'], lambda x: x)
            with self.assertRaises(OSError):
                dataset[0]

    def test_cli_rejects_missing_model_bad_batch_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / 'input'
            split.mkdir()
            model = root / 'model.pth'
            args = argparse.Namespace(model_path=model, test_root=split, output_dir=root / 'out',
                                      batch_size=2, num_workers=0)
            with self.assertRaises(ValueError):
                EVAL.validate_cli(args)
            model.write_bytes(b'fixture')
            args.batch_size = 0
            with self.assertRaises(ValueError):
                EVAL.validate_cli(args)
            args.batch_size = 2
            self.assertEqual(EVAL.validate_cli(args), root / 'out')
            args.output_dir.mkdir()
            with self.assertRaises(ValueError):
                EVAL.validate_cli(args)

    def test_small_end_to_end_no_training_or_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / 'input'
            classes = ['Myeloblast', 'Lymphoblast', 'Erythroblast']
            for i, name in enumerate(classes):
                (split / name).mkdir(parents=True)
                Image.new('RGB', (8 + i, 11), (i * 60, 100, 150)).save(split / name / 'sample.png')
            tiny = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(), torch.nn.Linear(3, 3))
            with torch.no_grad():
                tiny[-1].weight.zero_()
                tiny[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
            checkpoint = root / 'model.pth'
            torch.save(dict(state_dict=tiny.state_dict(), classes=classes,
                            args=dict(recipe='meta', img_size=16, model_name='fixture')), checkpoint)
            common = EVAL.import_file(ROOT / 'eval/evaluate_classifier.py', '_internal_model_factory')
            out = root / 'output'
            with mock.patch.object(common, 'create_model', return_value=tiny):
                rc = EVAL.main(['--model-path', str(checkpoint), '--trainer', 'classifier',
                                '--test-root', str(split), '--output-dir', str(out), '--device', 'cpu',
                                '--batch-size', '2', '--num-workers', '0'])
            self.assertEqual(rc, 0)
            result = json.loads((out / 'test_results.json').read_text())
            provenance = json.loads((out / 'provenance.json').read_text())
            self.assertEqual(result['n'], 3)
            self.assertAlmostEqual(result['f1'], 1 / 6)
            self.assertAlmostEqual(result['ba'], 1 / 3)
            self.assertEqual(result['classes'], classes)
            self.assertEqual(set(result['blast_subtypes']), set(classes))
            self.assertEqual(provenance['protocol']['tta'], 'none')
            self.assertEqual(provenance['status'], 'COMPLETE')
            self.assertEqual(len((out / 'predictions.csv').read_text().splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
