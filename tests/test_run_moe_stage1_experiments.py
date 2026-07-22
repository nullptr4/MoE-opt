from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_moe_stage1_experiments import (  # noqa: E402
    aggregate_performance_records,
    compare_with_e0,
    expected_schedule,
    expected_functional_workload_keys,
    expected_performance_workload_keys,
    stage1_source_snapshot,
    validate_functional_record,
)


class Stage1AggregateTest(unittest.TestCase):
    def functional_record(self, configs):
        return {
            "experiment": "E0",
            "schedule": expected_schedule("E0"),
            "results": [
                {
                    "test_type": "functional",
                    "config": config,
                    "correct": True,
                }
                for config in configs
            ],
        }

    def official_functional_configs(self):
        import json

        return [json.loads(key) for key in expected_functional_workload_keys()]

    def test_functional_record_requires_each_official_workload_once(self):
        configs = self.official_functional_configs()
        results = validate_functional_record("E0", self.functional_record(configs))
        self.assertEqual(len(results), len(configs))

    def test_functional_record_rejects_missing_workload(self):
        configs = self.official_functional_configs()
        with self.assertRaisesRegex(ValueError, "official configs"):
            validate_functional_record("E0", self.functional_record(configs[:1]))

    def test_functional_record_rejects_duplicate_workload(self):
        configs = self.official_functional_configs()
        with self.assertRaisesRegex(ValueError, "duplicate workload"):
            validate_functional_record(
                "E0",
                self.functional_record([configs[0], configs[0]]),
            )

    def test_three_processes_are_summarized_for_a_canonical_experiment(self):
        import json

        configs = [json.loads(key) for key in expected_performance_workload_keys()]
        records = [
            {
                "experiment": "E1",
                "schedule": expected_schedule("E1"),
                "results": [
                    {
                        "test_type": "performance",
                        "config": config,
                        "latency_ms": latency + config_index,
                    }
                    for config_index, config in enumerate(configs)
                ],
            }
            for latency in (9.0, 10.0, 12.0)
        ]
        summary = aggregate_performance_records("E1", records)
        self.assertEqual(len(summary), len(configs))
        self.assertEqual(
            sorted(item["median_latency_ms"] for item in summary),
            [10.0, 11.0],
        )
        self.assertTrue(all(item["mad_latency_ms"] == 1.0 for item in summary))
        self.assertTrue(all(item["complete"] for item in summary))

    def test_noncanonical_schedule_is_rejected(self):
        schedule = expected_schedule("E4")
        schedule["s2_stages"] = 1
        with self.assertRaisesRegex(ValueError, "canonical schedule"):
            aggregate_performance_records(
                "E4",
                [{"experiment": "E4", "schedule": schedule, "results": []}],
            )

    def test_empty_process_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            aggregate_performance_records("E0", [])

    def test_duplicate_workload_in_one_process_is_rejected(self):
        config = {"shape": "large"}
        result = {
            "test_type": "performance",
            "config": config,
            "latency_ms": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "duplicate workload"):
            aggregate_performance_records(
                "E0",
                [
                    {
                        "experiment": "E0",
                        "schedule": expected_schedule("E0"),
                        "results": [result, dict(result)],
                    }
                ],
            )

    def test_missing_workload_in_one_process_is_rejected(self):
        import json

        large, small = [json.loads(key) for key in expected_performance_workload_keys()]

        def record(configs):
            return {
                "experiment": "E0",
                "schedule": expected_schedule("E0"),
                "results": [
                    {
                        "test_type": "performance",
                        "config": config,
                        "latency_ms": 1.0,
                    }
                    for config in configs
                ],
            }

        with self.assertRaisesRegex(ValueError, "workload set"):
            aggregate_performance_records(
                "E0",
                [record((large, small)), record((large,))],
            )

    def test_all_processes_missing_the_same_workload_are_rejected(self):
        import json

        config = json.loads(next(iter(expected_performance_workload_keys())))
        record = {
            "experiment": "E0",
            "schedule": expected_schedule("E0"),
            "results": [
                {
                    "test_type": "performance",
                    "config": config,
                    "latency_ms": 1.0,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "official configs"):
            aggregate_performance_records("E0", [record, record, record])

    def test_comparison_enforces_one_percent_and_regression_limit(self):
        config_a = {"shape": "large"}
        config_b = {"shape": "small"}
        baseline = [
            {"config": config_a, "median_latency_ms": 100.0},
            {"config": config_b, "median_latency_ms": 50.0},
        ]
        candidate = [
            {"config": config_a, "median_latency_ms": 98.0},
            {"config": config_b, "median_latency_ms": 49.0},
        ]
        comparison = compare_with_e0(baseline, candidate)
        self.assertAlmostEqual(comparison["combined_improvement_pct"], 2.0)
        self.assertTrue(comparison["meets_latency_policy"])


class Stage1CliTest(unittest.TestCase):
    TUNER = ROOT / "benchmarks" / "tilelang-moe" / "tune_moe.py"

    def test_experiment_label_rejects_noncanonical_nonstage_flags(self):
        for flags in (
            ("--threads", "128"),
            ("--block-token", "64"),
            ("--swizzle-panel", "16"),
            ("--gemm-policy-down", "square"),
            ("--no-single-weight-buffer",),
            ("--combine-gate-up",),
        ):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(self.TUNER),
                        "--evaluation-target",
                        "local-proxy-guard",
                        "--experiment",
                        "E0",
                        *flags,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("complete canonical schedule", result.stderr)

    def test_fixed_candidate_disables_dynamic_selection_before_kernel_import(self):
        source = self.TUNER.read_text()
        kernel_import = source.index("import fusedmoe_benchmark as benchmark")
        for assignment in (
            'os.environ["MOE_AUTOTUNE"] = "0"',
            'os.environ["MOE_AUTOHEURISTIC"] = "0"',
            'os.environ["MOE_RECORD_RESULTS"] = "0"',
        ):
            self.assertLess(source.index(assignment), kernel_import)

    def test_formal_tuner_requires_explicit_local_proxy_opt_in(self):
        result = subprocess.run(
            [sys.executable, str(self.TUNER), "--experiment", "E0"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--evaluation-target", result.stderr)

    def test_stage1_snapshot_includes_both_orchestration_runners(self):
        paths = {item["path"] for item in stage1_source_snapshot()}
        self.assertIn("scripts/run_moe_stage1_experiments.py", paths)
        self.assertIn("scripts/confirm_moe_stage1_winner.py", paths)


if __name__ == "__main__":
    unittest.main()
