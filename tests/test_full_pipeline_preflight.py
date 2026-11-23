"""Offline checks: an incomplete requested seed set must not report success."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "eval" / "run_full_pipeline_eval.sh"


class FullPipelinePreflightTests(unittest.TestCase):
    def invoke(self, seeds="42 62 82", available=(), splits="phase1", abstention="0"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in available:
                (root / f"seed{seed}.pth").touch()
            (root / "detector.pt").touch()
            env = dict(os.environ, OUT_ROOT=str(root / "outputs"),
                       DATA_ROOT=str(root / "data"), DEVICE="cpu",
                       PYTHON="false", CKPT_PATTERN=str(root / "seed{seed}.pth"),
                       YOLO_MODEL=str(root / "detector.pt"),
                       SEEDS=seeds, SPLITS=splits, RUN_ABSTENTION=abstention)
            return subprocess.run(["bash", str(SCRIPT)], cwd=REPO, env=env,
                                  text=True, capture_output=True)

    def test_no_checkpoints_fails_before_detection(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertIn("requested seed 42", result.stderr)
        self.assertNotIn("[DETECT]", result.stdout)

    def test_partial_seed_set_is_not_success(self):
        result = self.invoke(available=(42, 62))
        self.assertEqual(result.returncode, 2)
        self.assertIn("requested seed 82", result.stderr)

    def test_duplicate_seed_is_rejected(self):
        result = self.invoke(seeds="42 42", available=(42,))
        self.assertEqual(result.returncode, 2)
        self.assertIn("distinct", result.stderr)

    def test_blank_seed_list_is_rejected(self):
        result = self.invoke(seeds=" ")
        self.assertEqual(result.returncode, 2)
        self.assertIn("at least one", result.stderr)

    def test_s12_requires_all_three_weights(self):
        result = self.invoke(seeds="42", available=(42,), splits="phase2", abstention="1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("S12 requires seed 62", result.stderr)

    def test_invalid_seed_is_rejected(self):
        result = self.invoke(seeds="invalid")
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-negative integers", result.stderr)

    def test_help_is_available_without_weights(self):
        result = subprocess.run(["bash", str(SCRIPT), "--help"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Every requested classifier checkpoint is required", result.stdout)


if __name__ == "__main__":
    unittest.main()
