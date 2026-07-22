from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_moe_evaluation import resolve_evaluation_command  # noqa: E402


class MoeDefaultEntrypointTest(unittest.TestCase):
    def test_no_arguments_selects_remote_submission_exact_timing(self):
        target, command = resolve_evaluation_command(())
        self.assertEqual(target, "remote_submission")
        self.assertEqual(Path(command[0]).name, "test-moe-submission.sh")
        self.assertEqual(command[1:], ["--benchmark-remote"])

    def test_formal_guard_requires_named_opt_in(self):
        target, command = resolve_evaluation_command(("--local-proxy-guard",))
        self.assertEqual(target, "local_proxy_guard")
        self.assertEqual(Path(command[1]).name, "fusedmoe_benchmark.py")
        self.assertIn("--local-proxy-guard", command)

    def test_describe_target_is_gpu_free_and_does_not_invent_aggregate(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_moe_evaluation.py"), "--describe-target"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["evaluation_target"], "remote_submission")
        self.assertEqual(
            payload["case_order"],
            ["remote-case-1", "remote-case-2", "remote-case-3"],
        )
        self.assertEqual(payload["aggregate_scoring"], {"status": "unknown", "value": None})


if __name__ == "__main__":
    unittest.main()
