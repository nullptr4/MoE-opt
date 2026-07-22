"""Pure-Python validation and fingerprint helpers for the remote MoE contract mirror."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "remote_contract.json"
SCHEMA_VERSION = "1.0"
PROVENANCE = {
    "published", "remotely_observed", "derived", "local_assumption", "unknown"
}


class ContractError(ValueError):
    pass


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    return "sha256:" + _canonical_hash(contract)


def submission_abi_fingerprint(contract: Mapping[str, Any]) -> str:
    fields = contract["fields"]
    payload = {
        path: fields[path]["value"]
        for path in (
            "abi.entrypoint",
            "abi.argument_count",
            "abi.argument_order",
            "abi.output_semantics",
        )
    }
    return "sha256:" + _canonical_hash(payload)


def hash_integer_sequence(values: Sequence[int]) -> str:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ContractError("integer sequence contains a non-integer")
    return "sha256:" + _canonical_hash(list(values))


def generated_code_fingerprint(
    *,
    device_source: str,
    host_source: str,
    tir_source: str,
) -> dict[str, Any]:
    """Retain exact generated text plus stable hashes and resource markers."""

    sources = {
        "device": device_source,
        "host": host_source,
        "tir": tir_source,
    }
    if any(not isinstance(value, str) or not value for value in sources.values()):
        raise ContractError("generated code sources must be non-empty strings")

    def observation(value: str) -> dict[str, Any]:
        encoded = value.encode("utf-8")
        return {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
            "lines": len(value.splitlines()),
            "text": value,
        }

    launch_bounds = re.findall(
        r"__launch_bounds__\s*\(([^)]*)\)", device_source
    )
    shared_offsets = sorted(
        {
            int(value)
            for value in re.findall(
                r"buf_dyn_shmem\s*\+\s*([0-9]+)", device_source
            )
        }
    )
    device_functions = re.findall(
        r"__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)", device_source
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": {name: observation(value) for name, value in sources.items()},
        "resource_markers": {
            "launch_bounds": launch_bounds,
            "dynamic_shared_offsets_bytes": shared_offsets,
            "extern_shared_declarations": len(
                re.findall(r"extern\s+__shared__", device_source)
            ),
            "device_functions": device_functions,
        },
    }


def contiguous_strides(shape: Sequence[int]) -> list[int]:
    result: list[int] = []
    stride = 1
    for dimension in reversed(tuple(shape)):
        result.append(stride)
        stride *= dimension
    return list(reversed(result))


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read remote contract: {exc}") from exc
    required = {
        "schema_version", "contract_id", "contract_version", "accessed_at",
        "canonical_owner", "sources", "fields", "incidents",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ContractError("remote contract top-level fields differ from schema v1")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ContractError("unsupported remote contract schema")
    if not isinstance(value["sources"], dict) or not value["sources"]:
        raise ContractError("remote contract requires sources")
    if not isinstance(value["fields"], dict) or not value["fields"]:
        raise ContractError("remote contract requires fields")
    field_keys = {"status", "value", "source_ids", "note", "close_with"}
    for name, claim in value["fields"].items():
        if not isinstance(name, str) or not name or not isinstance(claim, dict):
            raise ContractError("remote contract has an invalid field")
        if set(claim) != field_keys or claim["status"] not in PROVENANCE:
            raise ContractError(f"remote field {name} violates the claim schema")
        if not isinstance(claim["source_ids"], list) or any(
            source not in value["sources"] for source in claim["source_ids"]
        ):
            raise ContractError(f"remote field {name} cites an unknown source")
        if claim["status"] == "unknown":
            if claim["value"] is not None or not claim["close_with"]:
                raise ContractError(f"unknown remote field {name} must be null and closable")
        elif not claim["source_ids"]:
            raise ContractError(f"supported remote field {name} requires provenance")
    case_ids = value["fields"]["workload.case_order"]["value"]
    if case_ids != ["remote-case-1", "remote-case-2", "remote-case-3"]:
        raise ContractError("published remote case order changed unexpectedly")
    return value


def unresolved_fields(contract: Mapping[str, Any]) -> list[str]:
    return sorted(
        name
        for name, claim in contract["fields"].items()
        if claim["status"] in {"unknown", "local_assumption"}
    )


def remote_case_specs(contract: Mapping[str, Any]) -> tuple[tuple[str, int, int, int, int, int, int], ...]:
    fields = contract["fields"]
    result = []
    for case_id in fields["workload.case_order"]["value"]:
        prefix = f"cases.{case_id}"
        result.append(
            (
                case_id,
                fields[f"{prefix}.hidden"]["value"],
                fields[f"{prefix}.intermediate"]["value"],
                fields[f"{prefix}.experts"]["value"],
                fields[f"{prefix}.group_sum"]["value"],
                fields[f"{prefix}.warmup"]["value"],
                fields[f"{prefix}.iterations"]["value"],
            )
        )
    return tuple(result)


def tensor_fingerprint(tensor: Any) -> dict[str, Any]:
    shape = [int(value) for value in tensor.shape]
    return {
        "shape": shape,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "strides": [int(value) for value in tensor.stride()],
        "contiguous": bool(tensor.is_contiguous()),
    }


def case_parameter_fingerprint(
    *,
    contract: Mapping[str, Any],
    case: Any,
    out: Any,
    workspace: Any,
    warmup: int,
    iterations: int,
    routing_generator: str,
    route_weight_generator: str,
    case_cache_lifecycle: str,
    compiler_path: str,
) -> dict[str, Any]:
    sizes = [int(value) for value in case.group_sizes_list]
    raw_offsets = [int(value) for value in case.group_offsets.cpu().tolist()]
    padded_offsets = [int(value) for value in case.group_padded_offsets.cpu().tolist()]
    block_experts = [int(value) for value in case.group_idx_for_bx.cpu().tolist()]
    tensors = {
        "stacked_expert_tokens": tensor_fingerprint(case.tokens),
        "gate_w": tensor_fingerprint(case.gate),
        "up_w": tensor_fingerprint(case.up),
        "down_w": tensor_fingerprint(case.down),
        "routed_expert_weights": tensor_fingerprint(case.route_weights),
        "group_sizes": tensor_fingerprint(case.group_sizes),
        "group_offsets": tensor_fingerprint(case.group_offsets),
        "group_padded_offsets": tensor_fingerprint(case.group_padded_offsets),
        "group_idx_for_bx": tensor_fingerprint(case.group_idx_for_bx),
        "out": tensor_fingerprint(out),
        "private_up_logits_workspace": tensor_fingerprint(workspace),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "workload_case_id": case.name,
        "remote_case_id": case.name,
        "workload_class": "published_dimensions_local_routing_fixture",
        "contract_fingerprint": contract_fingerprint(contract),
        "submission_abi_fingerprint": submission_abi_fingerprint(contract),
        "abi_name": "submission-compact-ten-argument",
        "entrypoint": "run_kernel",
        "argument_count": 10,
        "argument_order": contract["fields"]["abi.argument_order"]["value"],
        "hidden": int(case.hidden),
        "intermediate": int(case.intermediate),
        "experts": len(sizes),
        "group_sum": sum(sizes),
        "tensors": tensors,
        "group_sizes": {
            "length": len(sizes),
            "sum": sum(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "mean": sum(sizes) / len(sizes),
            "zero_count": sum(value == 0 for value in sizes),
            "hash": hash_integer_sequence(sizes),
            "generator": routing_generator,
        },
        "offsets": {
            "length": len(raw_offsets),
            "sentinel": "absent",
            "raw_hash": hash_integer_sequence(raw_offsets),
            "padded_hash": hash_integer_sequence(padded_offsets),
            "raw_formula": "exclusive_prefix_sum(group_sizes)",
            "padded_formula": "prefix_sum(ceil((group_size + 1) / 128) * 128)",
        },
        "metadata": {
            "block_token": 128,
            "block_count": len(block_experts),
            "block_count_formula": "ceil(group_sum / 128) + experts",
            "group_idx_for_bx_hash": hash_integer_sequence(block_experts),
            "group_idx_for_bx_formula": "last expert whose padded offset is <= bx * 128",
        },
        "workspace_policy": {
            "data_rows": "group_sum",
            "workspace_rows": "group_sum",
            "output_rows": "group_sum",
            "workspace_owner": "private_submission_cache",
            "route_weight_generator": route_weight_generator,
        },
        "timing": {"warmup": warmup, "iterations": iterations},
        "cache_compiler": {
            "case_cache_lifecycle": case_cache_lifecycle,
            "compiler_path": compiler_path,
        },
        "applicability": "local_proxy",
    }
    result["parameter_fingerprint"] = "sha256:" + _canonical_hash(result)
    return result


def local_parity_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    unknowns = unresolved_fields(contract)
    return {
        "status": "LOCAL_PROXY_ONLY" if unknowns else "REMOTE_PARITY_VERIFIED",
        "remote_equivalent": not unknowns,
        "mismatch_fields": [],
        "unresolved_remote_fields": unknowns,
        "aggregate_status": (
            "unknown"
            if contract["fields"]["scoring.aggregate"]["status"] == "unknown"
            else "published"
        ),
        "conclusion": (
            "Published dimensions and local fixture mechanics are exercised, but unresolved "
            "remote fields prohibit submit-ready or remote-SOTA claims."
        ),
    }
