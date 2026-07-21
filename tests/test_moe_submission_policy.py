from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_moe_submission import check_submission  # noqa: E402  # pyright: ignore[reportMissingImports]


class MoeSubmissionPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.submission = ROOT / "benchmarks" / "tilelang-moe" / "submission.py"
        cls.source = cls.submission.read_text()

    def check_mutation(self, source: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            path.write_text(source)
            return check_submission(path)

    def test_checked_in_submission_satisfies_policy(self):
        self.assertEqual(check_submission(self.submission), [])

    def test_rejects_uncontrolled_route_weight_annotation(self):
        mutated = self.source.replace(
            "routed_expert_weights: T.Tensor((route_rows,), route_dtype)",
            "routed_expert_weights: T.Tensor((route_rows,), T.int32)",
        )
        self.assertNotEqual(mutated, self.source)
        errors = self.check_mutation(mutated)
        self.assertTrue(
            any("routed_expert_weights must use" in error for error in errors),
            errors,
        )

    def test_rejects_pytorch_compute(self):
        mutated = self.source.replace("torch.empty(", "torch.matmul(", 1)
        self.assertNotEqual(mutated, self.source)
        errors = self.check_mutation(mutated)
        self.assertTrue(
            any("PyTorch compute call torch.matmul" in error for error in errors),
            errors,
        )

    def test_rejects_assume_in_conservative_online_profile(self):
        mutated = self.source.replace(
            "            T.clear(gate_local)",
            "            T.assume(0 <= expert_id)\n            T.clear(gate_local)",
            1,
        )
        self.assertNotEqual(mutated, self.source)
        errors = self.check_mutation(mutated)
        self.assertTrue(
            any("T.assume() is not allowed" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
