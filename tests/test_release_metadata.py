import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_metadata", ROOT / "docs/check_release_metadata.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "docs/RELEASE_METADATA.json").read_text())

    def test_draft_is_explicit_and_valid(self):
        result = MODULE.validate(self.data)
        self.assertTrue(result["passed"])
        self.assertEqual(result["mode"], "draft")
        self.assertTrue(result["warnings"])

    def test_draft_cannot_pass_publication(self):
        result = MODULE.validate(self.data, publication=True)
        self.assertFalse(result["passed"])
        self.assertTrue(any("weights.download_url" in error for error in result["errors"]))

    def test_published_state_automatically_requires_links(self):
        self.data["release_state"] = "published"
        self.assertFalse(MODULE.validate(self.data)["passed"])

    def test_placeholder_is_never_a_link(self):
        self.data["weights"]["download_url"] = "https://example.org/ADD AT PUBLICATION"
        self.assertFalse(MODULE.validate(self.data)["passed"])

    def test_credentials_and_non_https_rejected(self):
        for value in ("https://user:secret@example.org/archive", "http://invalid.test/archive"):
            with self.subTest(value=value):
                self.data["weights"]["download_url"] = value
                self.assertFalse(MODULE.validate(self.data)["passed"])

    def test_unpublished_paper_need_not_have_fabricated_doi(self):
        self.assertIsNone(self.data["paper"]["doi"])
        self.assertTrue(MODULE.validate(self.data)["passed"])

    def test_paper_title_required(self):
        self.data["paper"]["title"] = ""
        self.assertFalse(MODULE.validate(self.data)["passed"])

    def test_wrong_core_count_rejected(self):
        self.data["weights"]["core_checkpoint_count"] = 1
        self.assertFalse(MODULE.validate(self.data)["passed"])


if __name__ == "__main__":
    unittest.main()
