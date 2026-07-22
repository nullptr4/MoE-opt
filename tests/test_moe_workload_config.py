from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = ROOT / "benchmarks" / "tilelang-moe"
sys.path.insert(0, str(MOE_ROOT))

from moe_test_config_tools import (  # noqa: E402
    WorkloadConfigError,
    default_evaluation_target,
    load_workload_config,
    local_proxy_guard_configs,
    remote_submission_specs,
    validate_workload_config,
)
from remote_contract_tools import load_contract, remote_case_specs  # noqa: E402


class MoeWorkloadConfigTest(unittest.TestCase):
    def test_default_is_ordered_remote_submission_matrix(self):
        config = load_workload_config()
        self.assertEqual(config["schema_version"], "2.0")
        self.assertEqual(config["default_evaluation_target"], "remote_submission")
        remote = default_evaluation_target(config)
        self.assertEqual(remote["abi"]["argument_count"], 10)
        self.assertEqual(remote["abi"]["entrypoint"], "run_kernel")
        self.assertEqual(remote_submission_specs(config), remote_case_specs(load_contract()))
        self.assertEqual(
            [case[0] for case in remote_submission_specs(config)],
            ["remote-case-1", "remote-case-2", "remote-case-3"],
        )

    def test_remote_aggregate_is_unknown_and_has_no_value(self):
        scoring = load_workload_config()["remote_submission"]["aggregate_scoring"]
        self.assertEqual(scoring["status"], "unknown")
        self.assertIsNone(scoring["value"])

    def test_old_formal_cases_are_explicit_local_proxy_guard(self):
        config = load_workload_config()
        with self.assertRaisesRegex(WorkloadConfigError, "explicit local-proxy"):
            local_proxy_guard_configs(config)
        guard = local_proxy_guard_configs(config, explicit_opt_in=True)
        self.assertEqual(guard["role"], "local_proxy_guard")
        self.assertFalse(guard["remote_equivalent"])
        self.assertEqual(guard["abi"]["argument_count"], 11)
        self.assertEqual(len(guard["functional"]), 2)
        self.assertEqual(len(guard["performance"]), 2)

    def test_config_cannot_drift_from_canonical_contract(self):
        config = copy.deepcopy(load_workload_config())
        config["remote_submission"]["cases"][1]["iterations"] = 30
        with self.assertRaisesRegex(WorkloadConfigError, "cases/order/timing"):
            validate_workload_config(config)

    def test_legacy_shape_cannot_become_default(self):
        config = copy.deepcopy(load_workload_config())
        config["default_evaluation_target"] = "local_proxy_guard"
        with self.assertRaisesRegex(WorkloadConfigError, "must be remote_submission"):
            validate_workload_config(config)


if __name__ == "__main__":
    unittest.main()
