#!/usr/bin/env python3
"""Resolve the role-aware MoE target and execute its one authoritative entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = ROOT / "benchmarks" / "tilelang-moe"
sys.path.insert(0, str(MOE_ROOT))

from moe_test_config_tools import (  # noqa: E402
    default_evaluation_target,
    load_workload_config,
    local_proxy_guard_configs,
)


def resolve_evaluation_command(arguments: Sequence[str]) -> tuple[str, list[str]]:
    config = load_workload_config()
    if arguments and arguments[0] == "--local-proxy-guard":
        local_proxy_guard_configs(config, explicit_opt_in=True)
        return (
            "local_proxy_guard",
            [
                sys.executable,
                str(MOE_ROOT / "fusedmoe_benchmark.py"),
                "--local-proxy-guard",
                *arguments[1:],
            ],
        )

    default_evaluation_target(config)
    remote_arguments = list(arguments) or ["--benchmark-remote"]
    return (
        "remote_submission",
        [str(ROOT / "scripts" / "test-moe-submission.sh"), *remote_arguments],
    )


def main() -> None:
    arguments = sys.argv[1:]
    describe = arguments == ["--describe-target"]
    target, command = resolve_evaluation_command(() if describe else arguments)
    if describe:
        config = load_workload_config()
        print(
            json.dumps(
                {
                    "evaluation_target": target,
                    "command": command,
                    "case_order": [
                        case["case_id"]
                        for case in config["remote_submission"]["cases"]
                    ],
                    "aggregate_scoring": config["remote_submission"][
                        "aggregate_scoring"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
