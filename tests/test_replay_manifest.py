"""Offline fixtures for exact historical-input replay (no dataset or weights)."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

SPEC = importlib.util.spec_from_file_location('replay_manifest', Path(__file__).resolve().parents[1] / 'data_prep/replay_manifest.py')
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.rng = np.random.default_rng(71)

    def tearDown(self):
        self.temp.cleanup()

    def write_image(self, path, value=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        value = self.rng.integers(0, 256, (15, 21, 3), dtype=np.uint8) if value is None else value
        self.assertTrue(cv2.imwrite(str(path), value))
        return path

    def args(self, **kw):
        value = dict(input_root=self.inputs, manifest=self.root / 'manifest.json',
                     output_dir=self.root / 'replayed', match_content=False,
                     allow_equivalent_copies=False)
        value.update(kw)
        return argparse.Namespace(**value)

    def fixture_crop(self):
        source = self.write_image(self.inputs / 'A' / 'one.png')
        crop_root = self.root / 'historical_crops'
        cut = REPLAY.image(source)[2:12, 3:17]
        crop = self.write_image(crop_root / 'A' / 'one__det00.png', cut)
        second = self.write_image(self.inputs / 'B' / 'two.png')
        detections = self.root / 'detections.csv'
        with detections.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source_path', 'crop_path', 'status', 'x1', 'y1', 'x2', 'y2'])
            writer.writeheader()
            writer.writerow(dict(source_path=str(source), crop_path=str(crop), status='detected', x1=3, y1=2, x2=17, y2=12))
            writer.writerow(dict(source_path=str(second), crop_path='', status='no_detection', x1='', y1='', x2='', y2=''))
        return self.args(detections=detections, crop_root=crop_root)

    def rewrite(self, args, transform):
        value = json.loads(args.manifest.read_text())
        transform(value)
        args.manifest.write_text(json.dumps(value))

    def test_crop_export_and_replay_exact_pixels_and_misses(self):
        args = self.fixture_crop()
        REPLAY.export_crops(args)
        result = REPLAY.replay_crops(args)
        self.assertEqual(result['fields'], 2)
        self.assertEqual(result['decoded_rgb_equal'], 1)
        self.assertEqual(result['statuses']['no_detection'], 1)
        original = REPLAY.signature(args.crop_root / 'A' / 'one__det00.png')
        self.assertEqual(REPLAY.signature(args.output_dir / 'crops/A/one__det00.png')['rgb_sha256'], original['rgb_sha256'])
        with (args.output_dir / 'detections.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[1]['crop_path'], '')

    def test_parallel_crop_export_and_replay_preserves_order_and_pixels(self):
        args = self.fixture_crop()
        args.workers = 2
        REPLAY.export_crops(args)
        manifest = json.loads(args.manifest.read_text())
        self.assertEqual([r['source_relative'] for r in manifest['entries']], ['A/one.png', 'B/two.png'])
        result = REPLAY.replay_crops(args)
        self.assertEqual(result['fields'], 2)
        self.assertEqual(result['decoded_rgb_equal'], 1)
        with (args.output_dir / 'detections.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([Path(r['source_path']).name for r in rows], ['one.png', 'two.png'])
        self.assertEqual(rows[1]['status'], 'no_detection')

    def test_opencv_thread_configuration_occurs_once_before_parallel_reads(self):
        from concurrent.futures import ThreadPoolExecutor
        source = self.write_image(self.inputs / 'one.png')
        with mock.patch.object(REPLAY, '_CV_READY', False), \
             mock.patch.object(cv2, 'setNumThreads', wraps=cv2.setNumThreads) as configure:
            with ThreadPoolExecutor(max_workers=4) as pool:
                values = list(pool.map(REPLAY.image, [source] * 12))
            self.assertEqual(len(values), 12)
            self.assertEqual(configure.call_count, 1)

    def test_parallel_crop_validation_failure_precedes_all_writes(self):
        args = self.fixture_crop()
        args.workers = 2
        REPLAY.export_crops(args)
        self.rewrite(args, lambda m: m['entries'][0].update(box_xyxy=[2, 2, 16, 12]))
        with self.assertRaisesRegex(ValueError, 'differs'):
            REPLAY.replay_crops(args)
        self.assertFalse(args.output_dir.exists())

    def test_parallel_content_matching_preserves_field_ids(self):
        args = self.fixture_crop()
        REPLAY.export_crops(args)
        pool = self.root / 'downloaded'
        self.write_image(pool / 'raw_images/renamed1.png', REPLAY.image(self.inputs / 'A/one.png'))
        self.write_image(pool / 'raw_images/renamed2.png', REPLAY.image(self.inputs / 'B/two.png'))
        report = REPLAY.replay_crops(self.args(input_root=pool, match_content=True, workers=2))
        self.assertEqual(report['fields'], 2)
        self.assertTrue((args.output_dir / 'fields/A/one.png').is_file())

    def test_crop_box_tampering_rejected_before_outputs(self):
        args = self.fixture_crop()
        REPLAY.export_crops(args)
        self.rewrite(args, lambda m: m['entries'][0].update(box_xyxy=[2, 2, 16, 12]))
        with self.assertRaisesRegex(ValueError, 'differs'):
            REPLAY.replay_crops(args)
        self.assertFalse(args.output_dir.exists())

    def test_crop_out_of_bounds_rejected(self):
        args = self.fixture_crop()
        REPLAY.export_crops(args)
        self.rewrite(args, lambda m: m['entries'][0].update(box_xyxy=[0, 0, 100, 20]))
        with self.assertRaisesRegex(ValueError, 'bounds'):
            REPLAY.replay_crops(args)

    def test_crop_fractional_box_rejected(self):
        args = self.fixture_crop()
        REPLAY.export_crops(args)
        self.rewrite(args, lambda m: m['entries'][0].update(box_xyxy=[3.5, 2, 17, 12]))
        with self.assertRaisesRegex(ValueError, 'integers'):
            REPLAY.replay_crops(args)

    def test_export_rejects_wrong_historical_crop(self):
        args = self.fixture_crop()
        self.write_image(args.crop_root / 'A/one__det00.png')
        with self.assertRaisesRegex(ValueError, 'exactly reproduce'):
            REPLAY.export_crops(args)

    def test_unsupported_status_rejected(self):
        args = self.fixture_crop()
        REPLAY.export_crops(args)
        self.rewrite(args, lambda m: m['entries'][1]['detection'].update(status='unknown'))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            REPLAY.replay_crops(args)

    def test_source_pixels_changed_rejected(self):
        args = self.fixture_crop()
        REPLAY.export_crops(args)
        self.write_image(self.inputs / 'A/one.png')
        with self.assertRaisesRegex(ValueError, 'No exact'):
            REPLAY.replay_crops(args)

    def test_crop_content_matching_preserves_field_ids_and_truth_layout(self):
        args = self.fixture_crop()
        REPLAY.export_crops(args)
        pool = self.root / 'downloaded'
        self.write_image(pool / 'raw_images/renamed1.png', REPLAY.image(self.inputs / 'A/one.png'))
        self.write_image(pool / 'raw_images/renamed2.png', REPLAY.image(self.inputs / 'B/two.png'))
        REPLAY.replay_crops(self.args(input_root=pool, match_content=True))
        with (args.output_dir / 'detections.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(Path(rows[0]['source_path']).relative_to(args.output_dir).as_posix(), 'fields/A/one.png')
        self.assertEqual(Path(rows[1]['source_path']).relative_to(args.output_dir).as_posix(), 'fields/B/two.png')
        self.assertTrue(Path(rows[0]['source_path']).is_file())

    def test_tree_split_layout_and_bytes_replayed(self):
        self.write_image(self.inputs / 'train/A/one.png')
        self.write_image(self.inputs / 'test/B/two.png')
        args = self.args()
        REPLAY.export_tree(args)
        result = REPLAY.replay_tree(args)
        self.assertEqual(result['images'], 2)
        self.assertEqual(result['bytes_equal'], 2)
        self.assertTrue((args.output_dir / 'test/B/two.png').is_file())

    def test_tree_explicit_hardlink_mode(self):
        source = self.write_image(self.inputs / 'train/A/one.png')
        args = self.args(copy_mode='hardlink')
        REPLAY.export_tree(args)
        REPLAY.replay_tree(args)
        self.assertEqual(source.stat().st_ino, (args.output_dir / 'train/A/one.png').stat().st_ino)

    def test_tree_expected_count_mismatch_fails_before_export(self):
        self.write_image(self.inputs / 'A/one.png')
        args = self.args(expected_images=2)
        with self.assertRaisesRegex(ValueError, 'Expected 2'):
            REPLAY.export_tree(args)
        self.assertFalse(args.manifest.exists())

    def test_tree_parallel_export(self):
        self.write_image(self.inputs / 'A/one.png')
        self.write_image(self.inputs / 'B/two.png')
        args = self.args(workers=2, expected_images=2)
        REPLAY.export_tree(args)
        self.assertEqual(json.loads(args.manifest.read_text())['n_images'], 2)

    def test_tree_parallel_replay(self):
        self.write_image(self.inputs / 'train/A/one.png')
        self.write_image(self.inputs / 'test/B/two.png')
        args = self.args(workers=2)
        REPLAY.export_tree(args)
        report = REPLAY.replay_tree(args)
        self.assertEqual(report['decoded_rgb_equal'], 2)

    def test_tree_can_restore_split_from_merged_pool_by_name(self):
        path = self.write_image(self.inputs / 'train/A/one.png')
        args = self.args()
        REPLAY.export_tree(args)
        pool = self.root / 'pool'
        self.write_image(pool / 'A/one.png', REPLAY.image(path))
        result = REPLAY.replay_tree(self.args(input_root=pool))
        self.assertEqual(result['decoded_rgb_equal'], 1)

    def test_tree_content_matching_handles_renamed_inputs(self):
        path = self.write_image(self.inputs / 'train/A/one.png')
        args = self.args()
        REPLAY.export_tree(args)
        pool = self.root / 'pool'
        self.write_image(pool / 'OriginalLabel/original_name.png', REPLAY.image(path))
        with self.assertRaisesRegex(ValueError, 'No exact'):
            REPLAY.replay_tree(self.args(input_root=pool))
        result = REPLAY.replay_tree(self.args(input_root=pool, match_content=True))
        self.assertEqual(result['decoded_rgb_equal'], 1)

    def test_tree_ambiguous_content_fails_without_permission(self):
        path = self.write_image(self.inputs / 'train/A/one.png')
        REPLAY.export_tree(self.args())
        pool = self.root / 'pool'
        self.write_image(pool / 'X/one.png', REPLAY.image(path))
        self.write_image(pool / 'Y/one.png', REPLAY.image(path))
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            REPLAY.replay_tree(self.args(input_root=pool))
        result = REPLAY.replay_tree(self.args(input_root=pool, allow_equivalent_copies=True))
        self.assertEqual(result['decoded_rgb_equal'], 1)

    def test_duplicate_target_fails_before_writing(self):
        self.write_image(self.inputs / 'A/one.png')
        args = self.args()
        REPLAY.export_tree(args)
        self.rewrite(args, lambda m: m['entries'].append(m['entries'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            REPLAY.replay_tree(args)
        self.assertFalse(args.output_dir.exists())

    def test_path_traversal_rejected(self):
        for bad in ('../escape.png', '/absolute.png', 'A/../../out.png', 'A\\out.png', ''):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                REPLAY.relative(bad)

    def test_repository_output_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            REPLAY.external(REPLAY.REPO / 'private_manifest.json')

    def test_rooted_image_rejects_final_symlink_escape(self):
        outside = self.write_image(self.root / 'outside.png')
        (self.inputs / 'alias.png').symlink_to(outside)
        with self.assertRaises(ValueError):
            REPLAY.rooted_image(str(self.inputs / 'alias.png'), self.inputs, {})

    def test_rooted_image_rejects_parent_symlink_escape(self):
        outside = self.root / 'outside'
        self.write_image(outside / 'one.png')
        (self.inputs / 'alias').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            REPLAY.rooted_image(str(self.inputs / 'alias/one.png'), self.inputs, {})

    def test_explicit_file_symlink_root_preserves_logical_label_and_id(self):
        outside = self.root / 'raw_download'
        actual = self.write_image(outside / 'download_name.png')
        link = self.inputs / 'NativeLabel/logical_id.png'
        link.parent.mkdir()
        link.symlink_to(actual)
        resolved, name = REPLAY.rooted_image(str(link), self.inputs, {}, [outside])
        self.assertEqual(resolved, actual)
        self.assertEqual(name, 'NativeLabel/logical_id.png')

    def test_explicit_root_does_not_allow_another_root(self):
        allowed = self.root / 'allowed'
        allowed.mkdir()
        actual = self.write_image(self.root / 'different/raw.png')
        link = self.inputs / 'alias.png'
        link.symlink_to(actual)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            REPLAY.rooted_image(str(link), self.inputs, {}, [allowed])

    def test_nested_relative_file_links_resolve_only_with_declared_root(self):
        raw = self.root / 'raw'
        actual = self.write_image(raw / 'physical/one.png')
        (raw / 'indirect.png').symlink_to('physical/one.png')
        logical = self.inputs / 'A/logical.png'
        logical.parent.mkdir()
        logical.symlink_to(raw / 'indirect.png')
        resolved, name = REPLAY.rooted_image(logical, self.inputs, {}, [raw])
        self.assertEqual(resolved, actual)
        self.assertEqual(name, 'A/logical.png')

    def test_final_file_symlink_cycle_rejected(self):
        first, second = self.inputs / 'one.png', self.inputs / 'two.png'
        first.symlink_to(second)
        second.symlink_to(first)
        with self.assertRaisesRegex(ValueError, 'cycle'):
            REPLAY.rooted_image(first, self.inputs, {})

    def test_resolver_rejects_undeclared_symlink_target(self):
        actual = self.write_image(self.root / 'raw/one.png')
        (self.inputs / 'one.png').symlink_to(actual)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            REPLAY.Resolver(self.inputs)

    def test_crop_export_and_replay_with_declared_raw_links(self):
        args = self.fixture_crop()
        raw_root = self.root / 'raw_download'
        raw_root.mkdir()
        for source in list(self.inputs.rglob('*.png')):
            dest = raw_root / source.name
            source.rename(dest)
            source.symlink_to(dest)
        args.allow_input_root = [raw_root]
        REPLAY.export_crops(args)
        manifest = json.loads(args.manifest.read_text())
        self.assertEqual(manifest['entries'][0]['source_relative'], 'A/one.png')
        report = REPLAY.replay_crops(args)
        self.assertEqual(report['fields'], 2)
        with (args.output_dir / 'detections.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(Path(rows[0]['source_path']).relative_to(self.inputs).as_posix(), 'A/one.png')

    def test_invalid_additional_input_root_rejected(self):
        self.write_image(self.inputs / 'A/one.png')
        with self.assertRaisesRegex(ValueError, 'not a directory'):
            REPLAY.Resolver(self.inputs, allowed_roots=[self.root / 'missing'])

    def test_tree_export_requires_declared_raw_symlink_root(self):
        actual = self.write_image(self.root / 'raw/one.png')
        (self.inputs / 'one.png').symlink_to(actual)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            REPLAY.export_tree(self.args())
        args = self.args(allow_input_root=[actual.parent])
        REPLAY.export_tree(args)
        self.assertEqual(json.loads(args.manifest.read_text())['entries'][0]['relative_path'], 'one.png')

    def test_aligned_export_rejects_undeclared_raw_symlink(self):
        actual = self.write_image(self.root / 'raw/one.png')
        (self.inputs / 'one.png').symlink_to(actual)
        crops = self.root / 'crops'
        self.write_image(crops / 'one.png', REPLAY.image(actual)[2:12, 3:17])
        args = self.args(crop_root=crops, report=self.root / 'geometry.json', workers=1)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            REPLAY.export_aligned_crops(args)

    def test_existing_output_not_overwritten(self):
        self.write_image(self.inputs / 'A/one.png')
        args = self.args()
        REPLAY.export_tree(args)
        args.output_dir.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            REPLAY.replay_tree(args)

    def test_input_output_overlap_rejected(self):
        self.write_image(self.inputs / 'A/one.png')
        args = self.args(output_dir=self.inputs / 'new_outputs')
        REPLAY.export_tree(args)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            REPLAY.replay_tree(args)

    def test_empty_manifest_rejected(self):
        args = self.args()
        args.manifest.write_text(json.dumps(dict(schema=REPLAY.SCHEMA, kind='tree', entries=[])))
        with self.assertRaisesRegex(ValueError, 'empty'):
            REPLAY.replay_tree(args)

    def test_wrong_manifest_schema_rejected(self):
        args = self.args()
        args.manifest.write_text(json.dumps(dict(schema='not-supported', kind='tree', entries=[1])))
        with self.assertRaisesRegex(ValueError, 'schema'):
            REPLAY.replay_tree(args)

    def test_export_does_not_overwrite_manifest(self):
        self.write_image(self.inputs / 'A/one.png')
        args = self.args()
        REPLAY.export_tree(args)
        with self.assertRaises(FileExistsError):
            REPLAY.export_tree(args)

    def test_no_images_is_not_success(self):
        with self.assertRaisesRegex(ValueError, 'No images'):
            REPLAY.export_tree(self.args())

    def test_exact_subimage_geometry_recovery(self):
        raw = self.rng.integers(0, 256, (20, 28, 3), dtype=np.uint8)
        self.assertEqual(REPLAY.recover_exact_box(raw, raw[3:15, 4:21]), [4, 3, 21, 15])

    def test_subimage_similarity_is_not_accepted(self):
        raw = self.rng.integers(0, 256, (20, 28, 3), dtype=np.uint8)
        crop = raw[3:15, 4:21].copy()
        crop[0, 1, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'found 0'):
            REPLAY.recover_exact_box(raw, crop)

    def test_repeated_subimage_geometry_is_ambiguous(self):
        raw = np.zeros((20, 28, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'found'):
            REPLAY.recover_exact_box(raw, raw[:8, :9])

    def test_aligned_export_records_retrospective_geometry(self):
        source = self.write_image(self.inputs / 'A/one.png')
        crop_root = self.root / 'historical'
        self.write_image(crop_root / 'A/one.png', REPLAY.image(source)[2:11, 3:16])
        args = self.args(crop_root=crop_root, report=self.root / 'recovery.json', workers=1)
        REPLAY.export_aligned_crops(args)
        result = json.loads(args.manifest.read_text())
        self.assertIn('Retrospectively', result['geometry_provenance'])
        self.assertEqual(result['entries'][0]['box_xyxy'], [3, 2, 16, 11])
        self.assertEqual(result['entries'][0]['detection']['yolo_conf'], '')

    def test_aligned_failure_is_not_exported_as_complete_manifest(self):
        self.write_image(self.inputs / 'A/one.png')
        crop_root = self.root / 'historical'
        self.write_image(crop_root / 'A/one.png')
        args = self.args(crop_root=crop_root, report=self.root / 'recovery.json', workers=1)
        with self.assertRaisesRegex(ValueError, 'no complete replay manifest'):
            REPLAY.export_aligned_crops(args)
        self.assertEqual(json.loads(args.report.read_text())['failures'], 1)
        self.assertFalse(args.manifest.exists())


if __name__ == '__main__':
    unittest.main()
