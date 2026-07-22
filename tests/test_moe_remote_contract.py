from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = ROOT / "benchmarks/tilelang-moe"
sys.path.insert(0, str(MOE_ROOT))

from remote_contract_tools import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    ContractError,
    contract_fingerprint,
    generated_code_fingerprint,
    load_contract,
    local_parity_summary,
    remote_case_specs,
    submission_abi_fingerprint,
)


class MoeRemoteContractTest(unittest.TestCase):
    def test_generated_code_fingerprint_retains_text_hashes_and_resource_markers(self):
        device = """extern __shared__ char buf_dyn_shmem[];\n__global__ void k0() __launch_bounds__(256, 1) { auto p = buf_dyn_shmem + 32768; }\n"""
        result = generated_code_fingerprint(
            device_source=device,
            host_source="launch k0",
            tir_source="@T.prim_func\ndef kernel(): pass",
        )
        self.assertEqual(
            result["sources"]["device"]["sha256"],
            hashlib.sha256(device.encode()).hexdigest(),
        )
        self.assertEqual(result["sources"]["device"]["text"], device)
        self.assertEqual(result["resource_markers"]["launch_bounds"], ["256, 1"])
        self.assertEqual(
            result["resource_markers"]["dynamic_shared_offsets_bytes"], [32768]
        )
        self.assertEqual(result["resource_markers"]["extern_shared_declarations"], 1)

    def test_canonical_fingerprints_and_published_cases_are_stable(self):
        contract = load_contract()
        self.assertEqual(
            contract_fingerprint(contract),
            "sha256:76213668b712444f98522210534bd01faba7b1d4c09bee17f44ff9fe76c6b1d8",
        )
        self.assertEqual(
            submission_abi_fingerprint(contract),
            "sha256:f4151e2be522b52490ef26bf2c204e1e0b66bb2fff69f84950de01be124f8fa2",
        )
        self.assertEqual(
            remote_case_specs(contract),
            (
                ("remote-case-1", 2048, 8192, 16, 2272, 5, 30),
                ("remote-case-2", 7168, 2048, 32, 4544, 5, 20),
                ("remote-case-3", 7168, 2048, 64, 9088, 5, 20),
            ),
        )

    def test_unknowns_keep_local_mirror_out_of_remote_equivalent_state(self):
        parity = local_parity_summary(load_contract())
        self.assertEqual(parity["status"], "LOCAL_PROXY_ONLY")
        self.assertFalse(parity["remote_equivalent"])
        self.assertEqual(parity["aggregate_status"], "unknown")
        self.assertIn("routing.assignment_distribution", parity["unresolved_remote_fields"])
        self.assertIn("toolchain.exact_online_inventory", parity["unresolved_remote_fields"])

    def test_unknown_claim_cannot_hide_a_local_assumption(self):
        raw = json.loads((MOE_ROOT / "remote_contract.json").read_text())
        mutated = copy.deepcopy(raw)
        mutated["fields"]["scoring.aggregate"]["value"] = "mean"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(mutated))
            with self.assertRaisesRegex(ContractError, "must be null"):
                load_contract(path)

    def test_incidents_retain_unknown_root_cause(self):
        incidents = load_contract()["incidents"]
        self.assertEqual(len(incidents), 2)
        self.assertEqual({item["exit_code"] for item in incidents}, {11})
        self.assertEqual({item["root_cause_status"] for item in incidents}, {"unknown"})

    def test_recorded_c500_report_preserves_samples_and_proxy_boundary(self):
        report_path = (
            ROOT
            / "data/benchmarks/c500-64g/remote-submission-parity-20260722.json"
        )
        report = json.loads(report_path.read_text())
        self.assertEqual(report["evidence_kind"], "hardware")
        self.assertEqual(report["parity"]["status"], "LOCAL_PROXY_ONLY")
        self.assertFalse(report["parity"]["remote_equivalent"])
        self.assertEqual(report["scoring"]["aggregate_status"], "unknown")
        self.assertIsNone(report["scoring"]["aggregate_value"])
        self.assertEqual(report["case_order"], ["remote-case-1", "remote-case-2", "remote-case-3"])
        expected_samples = {"remote-case-1": 30, "remote-case-2": 20, "remote-case-3": 20}
        for case in report["cases"]:
            self.assertEqual(case["correctness"]["mismatch_count"], 0)
            self.assertEqual(
                len(case["benchmark"]["samples_ms"]), expected_samples[case["case_id"]]
            )
            self.assertEqual(case["benchmark"]["outlier_policy"], "none")
            self.assertTrue(
                case["parameter_fingerprint"]["parameter_fingerprint"].startswith("sha256:")
            )

    def test_default_v2_hardware_report_binds_role_aware_workload_config(self):
        report_path = (
            ROOT
            / "data/benchmarks/c500-64g/remote-submission-default-v2-20260722.json"
        )
        report = json.loads(report_path.read_text())
        config_path = MOE_ROOT / "moe_test_configs.json"
        self.assertEqual(report["evidence_kind"], "hardware")
        self.assertEqual(report["parity"]["status"], "LOCAL_PROXY_ONLY")
        self.assertEqual(report["scoring"], {
            "per_case_metrics": True,
            "aggregate_status": "unknown",
            "aggregate_value": None,
        })
        self.assertEqual(report["environment"]["evaluation_target"], "remote_submission")
        self.assertEqual(
            report["environment"]["workload_config_sha256"],
            hashlib.sha256(config_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            [case["parameter_fingerprint"]["timing"] for case in report["cases"]],
            [
                {"warmup": 5, "iterations": 30},
                {"warmup": 5, "iterations": 20},
                {"warmup": 5, "iterations": 20},
            ],
        )


if __name__ == "__main__":
    unittest.main()
