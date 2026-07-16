from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from archive_moe_generated_code import (  # noqa: E402
    FC1_MARKER,
    FC2_MARKER,
    stage_function_hashes,
)


class GeneratedCodeArchiveTest(unittest.TestCase):
    def test_stage_hashes_cover_exact_function_slices(self):
        source = f"prefix\n{FC1_MARKER}fc1-body\n{FC2_MARKER}fc2-body\n"
        hashes = stage_function_hashes(source)
        fc1_start = source.index(FC1_MARKER)
        fc2_start = source.index(FC2_MARKER)
        self.assertEqual(
            hashes["fc1"],
            hashlib.sha256(source[fc1_start:fc2_start].encode()).hexdigest(),
        )
        self.assertEqual(
            hashes["fc2"],
            hashlib.sha256(source[fc2_start:].encode()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
