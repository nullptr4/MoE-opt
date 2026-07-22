"""Validated, role-aware access to the Routed-MoE workload configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from remote_contract_tools import load_contract, remote_case_specs


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "moe_test_configs.json"
SCHEMA_VERSION = "2.0"
DEFAULT_EVALUATION_TARGET = "remote_submission"
LOCAL_PROXY_GUARD = "local_proxy_guard"


class WorkloadConfigError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkloadConfigError(f"cannot read workload config: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadConfigError("workload config must be an object")
    return value


def _case_tuple(case: Mapping[str, Any]) -> tuple[str, int, int, int, int, int, int]:
    keys = {
        "case_id", "hidden", "intermediate", "experts", "group_sum", "warmup",
        "iterations",
    }
    if set(case) != keys:
        raise WorkloadConfigError("remote case fields differ from schema v2")
    values = tuple(case[key] for key in keys - {"case_id"})
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise WorkloadConfigError("remote dimensions and timing must be positive integers")
    return (
        case["case_id"],
        case["hidden"],
        case["intermediate"],
        case["experts"],
        case["group_sum"],
        case["warmup"],
        case["iterations"],
    )


def validate_workload_config(
    config: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> None:
    required = {
        "schema_version", "config_id", "default_evaluation_target",
        "remote_submission", "local_proxy_guard",
    }
    if set(config) != required or config.get("schema_version") != SCHEMA_VERSION:
        raise WorkloadConfigError("workload config top-level fields differ from schema v2")
    if config.get("default_evaluation_target") != DEFAULT_EVALUATION_TARGET:
        raise WorkloadConfigError("default evaluation target must be remote_submission")

    remote = config.get("remote_submission")
    if not isinstance(remote, dict) or set(remote) != {
        "role", "contract_reference", "abi", "cases", "aggregate_scoring"
    }:
        raise WorkloadConfigError("remote_submission fields differ from schema v2")
    if remote["role"] != "remote_objective" or remote["contract_reference"] != "remote_contract.json":
        raise WorkloadConfigError("remote_submission role or contract reference is invalid")
    expected_abi = {
        "name": "submission-compact-ten-argument",
        "entrypoint": "run_kernel",
        "argument_count": 10,
    }
    if remote["abi"] != expected_abi:
        raise WorkloadConfigError("remote_submission must use the ten-argument submission ABI")
    if remote["aggregate_scoring"] != {"status": "unknown", "value": None}:
        raise WorkloadConfigError("remote aggregate scoring must remain explicitly unknown")
    if not isinstance(remote["cases"], list) or not remote["cases"]:
        raise WorkloadConfigError("remote_submission requires an ordered case list")

    canonical = load_contract() if contract is None else contract
    configured_cases = tuple(_case_tuple(case) for case in remote["cases"])
    if configured_cases != remote_case_specs(canonical):
        raise WorkloadConfigError(
            "remote_submission cases/order/timing differ from canonical remote_contract"
        )
    contract_fields = canonical["fields"]
    if remote["abi"]["entrypoint"] != contract_fields["abi.entrypoint"]["value"]:
        raise WorkloadConfigError("remote entrypoint differs from canonical remote_contract")
    if remote["abi"]["argument_count"] != contract_fields["abi.argument_count"]["value"]:
        raise WorkloadConfigError("remote argument count differs from canonical remote_contract")

    guard = config.get(LOCAL_PROXY_GUARD)
    if not isinstance(guard, dict) or guard.get("role") != LOCAL_PROXY_GUARD:
        raise WorkloadConfigError("old formal workloads must be labeled local_proxy_guard")
    if guard.get("remote_equivalent") is not False:
        raise WorkloadConfigError("local_proxy_guard cannot be remote-equivalent")
    if guard.get("abi") != {
        "name": "formal-compact-eleven-tensor",
        "entrypoint": "RoutedMoEKernel",
        "argument_count": 11,
    }:
        raise WorkloadConfigError("local_proxy_guard ABI is invalid")
    if set(guard.get("timing", {})) != {"warmup", "iterations"}:
        raise WorkloadConfigError("local_proxy_guard timing is incomplete")
    for role in ("functional", "performance"):
        cases = guard.get(role)
        if not isinstance(cases, list) or not cases:
            raise WorkloadConfigError(f"local_proxy_guard.{role} must be non-empty")
    scoring = guard.get("aggregate_scoring")
    if not isinstance(scoring, dict) or scoring.get("remote_applicability") is not False:
        raise WorkloadConfigError("local proxy aggregate must deny remote applicability")


def load_workload_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    validate_workload_config(config)
    return config


def default_evaluation_target(config: Mapping[str, Any]) -> Mapping[str, Any]:
    validate_workload_config(config)
    if config["default_evaluation_target"] != DEFAULT_EVALUATION_TARGET:
        raise WorkloadConfigError("legacy fallback cannot become the default objective")
    return config[DEFAULT_EVALUATION_TARGET]


def remote_submission_specs(
    config: Mapping[str, Any],
) -> tuple[tuple[str, int, int, int, int, int, int], ...]:
    remote = default_evaluation_target(config)
    return tuple(_case_tuple(case) for case in remote["cases"])


def local_proxy_guard_configs(
    config: Mapping[str, Any],
    *,
    explicit_opt_in: bool = False,
) -> Mapping[str, Any]:
    validate_workload_config(config)
    if not explicit_opt_in:
        raise WorkloadConfigError(
            "local_proxy_guard is not a default objective; pass an explicit local-proxy option"
        )
    return config[LOCAL_PROXY_GUARD]
