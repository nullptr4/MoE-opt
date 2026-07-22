#!/usr/bin/env python3
"""Validate the target mirror of the versioned remote Routed-MoE contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = ROOT / "benchmarks/tilelang-moe"
sys.path.insert(0, str(MOE_ROOT))

from remote_contract_tools import (  # noqa: E402
    contract_fingerprint,
    load_contract,
    local_parity_summary,
    remote_case_specs,
    submission_abi_fingerprint,
)
from moe_test_config_tools import (  # noqa: E402
    load_workload_config,
    remote_submission_specs,
)


def main() -> int:
    contract = load_contract()
    workload_config = load_workload_config()
    print(
        json.dumps(
            {
                "contract_id": contract["contract_id"],
                "contract_version": contract["contract_version"],
                "contract_fingerprint": contract_fingerprint(contract),
                "submission_abi_fingerprint": submission_abi_fingerprint(contract),
                "published_cases": [list(item) for item in remote_case_specs(contract)],
                "workload_config": {
                    "schema_version": workload_config["schema_version"],
                    "default_evaluation_target": workload_config["default_evaluation_target"],
                    "remote_submission_cases": [
                        list(item) for item in remote_submission_specs(workload_config)
                    ],
                    "aggregate_scoring": workload_config["remote_submission"][
                        "aggregate_scoring"
                    ],
                    "legacy_role": workload_config["local_proxy_guard"]["role"],
                },
                "parity": local_parity_summary(contract),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
