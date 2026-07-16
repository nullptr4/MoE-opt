from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_moe_baseline import (  # noqa: E402
    BASELINE_SCHEDULE,
    SOURCE_SNAPSHOT_PATHS,
    _source_snapshot,
    aggregate_process_records,
    percentile,
    summarize_samples,
)


class StatisticsTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([10.0], 95), 10.0)
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0], 95), 29.0)

    def test_summary_reports_median_mad_and_p95(self):
        summary = summarize_samples([10.0, 12.0, 14.0])
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["median_latency_ms"], 12.0)
        self.assertEqual(summary["mad_latency_ms"], 2.0)
        self.assertAlmostEqual(summary["p95_latency_ms"], 13.8)

    def test_empty_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            summarize_samples([])


class AggregateTest(unittest.TestCase):
    def test_three_processes_are_aggregated_per_workload(self):
        functional_config = {"dhidden": 256, "dexpert": 128, "bs": 1}
        performance_config = {"dhidden": 256, "dexpert": 128, "bs": 4}
        records = []
        for latency in (10.0, 12.0, 14.0):
            records.append(
                {
                    "schedule": dict(BASELINE_SCHEDULE),
                    "results": [
                        {
                            "test_type": "functional",
                            "config": functional_config,
                            "correct": True,
                        },
                        {
                            "test_type": "performance",
                            "config": performance_config,
                            "latency_ms": latency,
                        },
                    ],
                }
            )

        aggregate = aggregate_process_records(records)

        self.assertTrue(aggregate["all_functional_passed"])
        self.assertTrue(aggregate["all_performance_complete"])
        self.assertEqual(aggregate["functional"][0]["passed_processes"], 3)
        self.assertEqual(aggregate["performance"][0]["median_latency_ms"], 12.0)

    def test_non_baseline_schedule_is_rejected(self):
        schedule = dict(BASELINE_SCHEDULE)
        schedule["threads"] = 128
        with self.assertRaisesRegex(ValueError, "fixed baseline schedule"):
            aggregate_process_records([{"schedule": schedule, "results": []}])


class ReproducibilityTest(unittest.TestCase):
    def test_checked_in_schedule_matches_runner(self):
        schedule = json.loads((ROOT / "config" / "moe_baseline_schedule.json").read_text())
        self.assertEqual(schedule, BASELINE_SCHEDULE)

    def test_source_snapshot_is_complete_and_self_consistent(self):
        snapshot = _source_snapshot(ROOT)
        self.assertEqual([item["path"] for item in snapshot], list(SOURCE_SNAPSHOT_PATHS))
        for item in snapshot:
            self.assertEqual(item["content"], (ROOT / item["path"]).read_text())


if __name__ == "__main__":
    unittest.main()
