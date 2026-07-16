from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from confirm_moe_stage1_winner import (  # noqa: E402
    EXPERIMENTS,
    confirmation_execution_complete,
)


class ConfirmationCompletenessTest(unittest.TestCase):
    def complete_inputs(self):
        functional = {experiment: {"passed": True} for experiment in EXPERIMENTS}
        records = {experiment: [{}, {}, {}] for experiment in EXPERIMENTS}
        runs = {
            experiment: [
                {"returncode": 0, "result_recorded": True}
                for _ in range(3)
            ]
            for experiment in EXPERIMENTS
        }
        performance = {
            experiment: [{"complete": True}]
            for experiment in EXPERIMENTS
        }
        comparison = {"meets_latency_policy": False}
        return functional, records, runs, performance, comparison

    def test_complete_measurement_is_valid_even_when_candidate_is_rejected(self):
        functional, records, runs, performance, comparison = self.complete_inputs()
        self.assertTrue(
            confirmation_execution_complete(
                functional=functional,
                records=records,
                runs=runs,
                performance=performance,
                comparison=comparison,
                required_runs=3,
            )
        )

    def test_empty_performance_cannot_pass_vacuously(self):
        functional, records, runs, performance, comparison = self.complete_inputs()
        performance = {experiment: [] for experiment in EXPERIMENTS}
        self.assertFalse(
            confirmation_execution_complete(
                functional=functional,
                records=records,
                runs=runs,
                performance=performance,
                comparison=comparison,
                required_runs=3,
            )
        )

    def test_missing_comparison_is_incomplete(self):
        functional, records, runs, performance, _ = self.complete_inputs()
        self.assertFalse(
            confirmation_execution_complete(
                functional=functional,
                records=records,
                runs=runs,
                performance=performance,
                comparison=None,
                required_runs=3,
            )
        )


if __name__ == "__main__":
    unittest.main()
