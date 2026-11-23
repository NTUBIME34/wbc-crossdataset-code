"""Direct evaluation runners must not report success for absent requested models."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class MissingInputsTests(unittest.TestCase):
    def run_missing(self, runner, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ('data', 'weights', 'models', 'external', 'internal/test'):
                (root / rel).mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, DATA_ROOT=str(root / 'data'), MODEL_DIR=str(root / 'weights'),
                       MODEL_ROOT=str(root / 'models'), OUT_ROOT=str(root / 'output'),
                       WBCB_P1_CROPS=str(root / 'external'), CORPUS_ROOT=str(root / 'internal'),
                       CNN_ROOT=str(root / 'models'), DINO_ROOT=str(root / 'models'),
                       LODO_ROOT=str(root / 'models'), SEEDS='42', SIZES='224',
                       PYTHON='false', **overrides)
            result = subprocess.run(['bash', str(REPO / 'eval' / runner)], env=env,
                                    cwd=root, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=20)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn('checkpoint not found', result.stdout)
            self.assertIn('[FAIL]', result.stdout)

    def test_backbone_missing_seed(self):
        self.run_missing('run_backbone_external_eval.sh', RUN_GROUPS='cnn')

    def test_recipe_missing_seed(self):
        self.run_missing('run_augmentation_recipe_eval.sh', RECIPES='class_weighted_ce')

    def test_dinobloom_missing_seed(self):
        self.run_missing('run_dinobloom_external_eval.sh')

    def test_lodo_missing_seed(self):
        self.run_missing('run_lodo_eval.sh', FOLDS='LMU', ARM='proposed')

    def test_resolution_missing_seed(self):
        self.run_missing('run_resolution_scaling_eval.sh')

    def test_internal_missing_seed(self):
        self.run_missing('run_internal_mapped_eval.sh', RECIPES='colour_aug_proposed')

    def test_routes_missing_seed(self):
        self.run_missing('run_routes_ablation.sh', ROUTES='A')


if __name__ == '__main__':
    unittest.main()
