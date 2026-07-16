from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_moe_stage2_profiler_table import (  # noqa: E402
    DEFAULT_OUTPUT,
    REQUIRED_VERSION_IDS,
    build_table,
    markdown_report,
    numeric,
    parse_torch_stage_durations,
)
from validate_moe_stage2_profiler import validate_table  # noqa: E402


class ProfilerValueTest(unittest.TestCase):
    def test_numeric_preserves_units_without_misreading_suffixes(self):
        self.assertEqual(numeric("137,458.68(Kcycles)"), 137458.68)
        self.assertEqual(numeric("64,631,275,520.0byte"), 64631275520.0)
        self.assertEqual(numeric("81.20%"), 81.2)

    def test_torch_profile_parser_reads_self_cuda_not_zero_cpu_columns(self):
        large = parse_torch_stage_durations(
            ROOT, "logs/moe-torch-profiler-large-20260711T144843Z.log"
        )
        small = parse_torch_stage_durations(
            ROOT, "logs/moe-torch-profiler-small-20260711T144218Z.log"
        )
        self.assertEqual(large, {"fc1": 115.040, "fc2": 81.633})
        self.assertEqual(small, {"fc1": 14.080, "fc2": 11.043})


class Stage2DecisionTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = build_table(ROOT)
        cls.by_id = {item["id"]: item for item in cls.table["versions"]}

    def test_required_versions_and_deterministic_decisions_are_present(self):
        self.assertEqual(
            tuple(item["id"] for item in self.table["versions"]),
            REQUIRED_VERSION_IDS,
        )
        self.assertEqual(
            self.by_id["E1"]["decision"]["primary_bottleneck"],
            "reduction_loop_overhead",
        )
        self.assertEqual(
            self.by_id["E2"]["decision"]["primary_bottleneck"],
            "cta_amplification",
        )
        self.assertEqual(
            self.by_id["E4"]["decision"]["primary_bottleneck"],
            "pipeline_overhead",
        )
        self.assertEqual(
            self.by_id["E5"]["decision"]["primary_bottleneck"],
            "register_spill",
        )

    def test_missing_metrics_are_explicit_and_targeted_reports_are_used(self):
        coverage = self.table["methodology"]["plan_metric_coverage"]
        self.assertEqual(coverage["active_blocks_per_sm"]["status"], "unavailable")
        self.assertEqual(coverage["dram_read_write_bandwidth"]["status"], "unavailable")
        for experiment in (f"E{index}" for index in range(6)):
            for stage in ("fc1", "fc2"):
                report = self.by_id[experiment]["stages"][stage]["counters"][
                    "targeted_report"
                ]
                self.assertTrue(Path(report).name.startswith("1_kernel_kernel"))
                self.assertNotEqual(Path(report).name, "report.txt.json")

    def test_async_pipeline_is_closed_and_stage4_is_selected(self):
        summary = self.table["summary"]
        self.assertFalse(summary["enter_fc1_async_pipeline"])
        self.assertEqual(summary["retained_default"], "E0")
        self.assertIn("Stage 4", summary["recommended_next_stage"])
        self.assertIn("下一步", markdown_report(self.table))

    def test_every_version_uses_a_documented_decision_rule(self):
        rules = {rule["id"] for rule in self.table["decision_rules"]}
        self.assertTrue(
            all(version["decision"]["rule"] in rules for version in self.table["versions"])
        )

    def test_checked_in_table_reconstructs_from_hashed_sources(self):
        table_path = ROOT / DEFAULT_OUTPUT
        self.assertTrue(table_path.is_file())
        table = json.loads(table_path.read_text())
        self.assertEqual(validate_table(table, ROOT), [])


if __name__ == "__main__":
    unittest.main()
