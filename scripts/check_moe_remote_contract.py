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


def main() -> int:
    contract = load_contract()
    print(
        json.dumps(
            {
                "contract_id": contract["contract_id"],
                "contract_version": contract["contract_version"],
                "contract_fingerprint": contract_fingerprint(contract),
                "submission_abi_fingerprint": submission_abi_fingerprint(contract),
                "published_cases": [list(item) for item in remote_case_specs(contract)],
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
