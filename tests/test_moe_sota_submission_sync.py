from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_moe_sota_submission_sync import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    MANIFEST,
    check_sync,
)


class MoeSotaSubmissionSyncTest(unittest.TestCase):
    def make_repository_copy(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        manifest = json.loads((ROOT / MANIFEST).read_text())
        for section_name in ("formal_sota", "submission"):
            relative = Path(manifest[section_name]["source_path"])
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        manifest_path = root / MANIFEST
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / MANIFEST, manifest_path)
        return temporary, root

    def test_checked_in_sota_and_submission_are_synchronized(self):
        self.assertEqual(check_sync(ROOT), [])

    def test_rejects_formal_sota_change_without_submission_sync(self):
        temporary, root = self.make_repository_copy()
        self.addCleanup(temporary.cleanup)
        manifest = json.loads((root / MANIFEST).read_text())
        formal = root / manifest["formal_sota"]["source_path"]
        formal.write_text(formal.read_text() + "\n# unsynchronized formal SOTA change\n")
        errors = check_sync(root)
        self.assertTrue(any("formal_sota SHA mismatch" in error for error in errors), errors)

    def test_rejects_submission_change_without_manifest_update(self):
        temporary, root = self.make_repository_copy()
        self.addCleanup(temporary.cleanup)
        manifest = json.loads((root / MANIFEST).read_text())
        submission = root / manifest["submission"]["source_path"]
        submission.write_text(submission.read_text() + "\n# unsynchronized submission change\n")
        errors = check_sync(root)
        self.assertTrue(any("submission SHA mismatch" in error for error in errors), errors)

    def test_rejects_missing_adapted_mechanism(self):
        temporary, root = self.make_repository_copy()
        self.addCleanup(temporary.cleanup)
        manifest_path = root / MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["required_submission_snippets"] = {"T.assume(False)": 1}
        manifest_path.write_text(json.dumps(manifest))
        errors = check_sync(root)
        self.assertTrue(any("missing synchronized mechanism" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
