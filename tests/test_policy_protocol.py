"""The reported policy sweep used hflip; the separate no-flip task did not."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PolicyProtocolTests(unittest.TestCase):
    def setUp(self):
        registry = Path(__file__).resolve().parents[1] / 'reproduce/study_tasks.json'
        self.tasks = {row['id']: row for row in json.loads(registry.read_text())['tasks']}

    def test_reported_policy_sweep_uses_horizontal_flip(self):
        task = self.tasks['policy-sweep']
        self.assertEqual(task['environment_overrides']['TTA'], 'hflip')
        self.assertEqual(task['environment_overrides']['BATCH_SIZE'], '32')
        self.assertEqual(task['environment_overrides']['SEEDS'], '42 62 82')
        self.assertEqual(task['outputs'][0]['path'],
                         '{OUT_ROOT}/augmentation_recipes/aug_{policy}/seed_{seed}/predictions.csv')

    def test_explicit_no_flip_comparison_stays_separate(self):
        self.assertEqual(self.tasks['no-flip-comparison']['environment_overrides']['TTA'], 'none')

    def run_batch_fixture(self, value):
        """The fake interpreter records flags; no model or dataset is loaded."""
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='policy_batch_test_') as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            (root / 'weights').mkdir()
            (root / 'weights/swin_base_colour_aug_cwce_seed42.pth').write_bytes(b'fixture')
            arguments = root / 'arguments.jsonl'
            fake_python = root / 'record_arguments.py'
            fake_python.write_text(
                '#!' + sys.executable + '\n'
                'import json, os, sys\n'
                'with open(os.environ["POLICY_TEST_ARGUMENTS"], "a") as stream:\n'
                '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n')
            fake_python.chmod(0o700)
            env = dict(os.environ, OUT_ROOT=str(root / 'out'), WBCB_P1_CROPS=str(root / 'data'),
                       MODEL_DIR=str(root / 'weights'), RECIPES='colour_aug_proposed',
                       SEEDS='42', PYTHON=str(fake_python), FORCE='1', POLICY_TEST_ARGUMENTS=str(arguments))
            env.pop('BATCH_SIZE', None)
            if value is not None:
                env['BATCH_SIZE'] = value
            result = subprocess.run(['bash', str(repo / 'eval/run_augmentation_recipe_eval.sh')],
                                    cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=20)
            calls = [json.loads(line) for line in arguments.read_text().splitlines()] if arguments.exists() else []
            return result, calls

    def test_zero_batch_is_rejected_before_evaluation(self):
        result, calls = self.run_batch_fixture('0')
        self.assertEqual(result.returncode, 2)
        self.assertIn('BATCH_SIZE must be a positive integer', result.stdout)
        self.assertEqual(calls, [])

    def test_negative_batch_is_rejected_before_evaluation(self):
        result, calls = self.run_batch_fixture('-1')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])

    def test_nonnumeric_batch_is_rejected_before_evaluation(self):
        result, calls = self.run_batch_fixture('invalid')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])

    def test_batch_override_reaches_evaluator(self):
        result, calls = self.run_batch_fixture('32')
        self.assertEqual(result.returncode, 0, result.stdout)
        evaluations = [call for call in calls if '--batch-size' in call]
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations[0][evaluations[0].index('--batch-size') + 1], '32')

    def test_default_batch_stays_64_for_existing_recipes(self):
        result, calls = self.run_batch_fixture(None)
        self.assertEqual(result.returncode, 0, result.stdout)
        evaluations = [call for call in calls if '--batch-size' in call]
        self.assertEqual(evaluations[0][evaluations[0].index('--batch-size') + 1], '64')


if __name__ == '__main__':
    unittest.main()
