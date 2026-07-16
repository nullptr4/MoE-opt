from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "tilelang-moe"))

from moe_schedule import (  # noqa: E402
    BASELINE_STAGE_SCHEDULE,
    EXPERIMENT_PRESETS,
    STAGE1_FIXED_SCHEDULE,
    canonical_experiment_schedule,
    experiment_stage_schedule,
    resolve_stage_schedule,
    validate_stage_schedule,
)


class StageScheduleResolutionTest(unittest.TestCase):
    def test_default_legacy_schedule_maps_to_both_stages(self):
        self.assertEqual(resolve_stage_schedule(), BASELINE_STAGE_SCHEDULE)

    def test_legacy_tiles_preserve_the_old_cross_stage_coupling(self):
        self.assertEqual(
            resolve_stage_schedule(block_dhidden=64, block_dexpert=256),
            {
                "s1_bn": 256,
                "s1_bk": 64,
                "s1_stages": 1,
                "s2_bn": 64,
                "s2_bk": 256,
                "s2_stages": 1,
            },
        )

    def test_explicit_fields_override_only_their_own_stage(self):
        self.assertEqual(
            resolve_stage_schedule(
                block_dhidden=96,
                block_dexpert=160,
                num_stages=3,
                num_stages_down=4,
                s1_bk=64,
                s2_bn=256,
                s2_stages=2,
            ),
            {
                "s1_bn": 160,
                "s1_bk": 64,
                "s1_stages": 3,
                "s2_bn": 256,
                "s2_bk": 160,
                "s2_stages": 2,
            },
        )

    def test_non_positive_or_boolean_fields_are_rejected(self):
        for kwargs in ({"s1_bn": 0}, {"s1_bk": -1}, {"s2_stages": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "positive integer"):
                resolve_stage_schedule(**kwargs)

    def test_unmasked_tile_tails_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            validate_stage_schedule(
                {**BASELINE_STAGE_SCHEDULE, "s1_bk": 24},
                d_hidden=7168,
                d_expert=2048,
            )
        with self.assertRaisesRegex(ValueError, "must divide"):
            validate_stage_schedule(
                {**BASELINE_STAGE_SCHEDULE, "s2_bn": 160},
                d_hidden=3584,
                d_expert=1024,
            )


class ExperimentPresetTest(unittest.TestCase):
    def test_e0_through_e5_are_single_variable_presets(self):
        expected = {
            "E0": {},
            "E1": {"s1_bk": 64},
            "E2": {"s1_bn": 64},
            "E3": {"s2_bk": 64},
            "E4": {"s2_bk": 64, "s2_stages": 2},
            "E5": {"s2_bn": 256},
        }
        self.assertEqual(EXPERIMENT_PRESETS, expected)
        for name, overrides in expected.items():
            with self.subTest(experiment=name):
                self.assertEqual(
                    experiment_stage_schedule(name),
                    {**BASELINE_STAGE_SCHEDULE, **overrides},
                )

    def test_unknown_experiment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown experiment"):
            experiment_stage_schedule("E6")

    def test_canonical_label_covers_the_complete_fixed_schedule(self):
        self.assertEqual(
            canonical_experiment_schedule("E3"),
            {
                **STAGE1_FIXED_SCHEDULE,
                **BASELINE_STAGE_SCHEDULE,
                "s2_bk": 64,
            },
        )


class KernelSignatureTest(unittest.TestCase):
    def test_autotune_factory_and_kernel_accept_all_explicit_stage_keys(self):
        source = (ROOT / "benchmarks" / "tilelang-moe" / "custom_fusedmoe.py").read_text()
        tree = ast.parse(source)
        functions = {
            node.name: {argument.arg for argument in node.args.args}
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        expected = {
            "s1_bn",
            "s1_bk",
            "s1_stages",
            "s2_bn",
            "s2_bk",
            "s2_stages",
        }
        self.assertTrue(expected <= functions["_moe_autoheuristic_configs"])
        self.assertTrue(expected <= functions["_moe_forward_tilelang_routed"])


if __name__ == "__main__":
    unittest.main()
