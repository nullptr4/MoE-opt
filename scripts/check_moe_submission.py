#!/usr/bin/env python3
"""Fail-closed static policy check for the single-file MoE submission."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {"math", "tilelang", "torch"}
RUN_KERNEL_ARGUMENTS = [
    "stacked_expert_tokens",
    "gate_w",
    "up_w",
    "down_w",
    "routed_expert_weights",
    "group_sizes",
    "group_offsets",
    "group_padded_offsets",
    "group_idx_for_bx",
    "out",
]
BANNED_CALL_NAMES = {
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "open",
    "print",
    "setattr",
    "type",
}
BANNED_ATTRIBUTE_CALLS = {
    "cpu",
    "data_ptr",
    "item",
    "tolist",
}


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def is_tensor_annotation(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and dotted_name(node.func) == "T.Tensor"
        and len(node.args) == 2
        and not node.keywords
    )


def check_submission(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [f"cannot parse submission: {exc}"]

    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(
                        f"line {statement.lineno}: import root {root!r} is not allowed"
                    )
        elif isinstance(statement, ast.ImportFrom):
            if statement.level:
                errors.append(
                    f"line {statement.lineno}: relative imports are not allowed"
                )
            root = (statement.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                errors.append(
                    f"line {statement.lineno}: import root {root!r} is not allowed"
                )
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if value is None:
                errors.append(
                    f"line {statement.lineno}: module-level assignment needs a literal value"
                )
            else:
                try:
                    ast.literal_eval(value)
                except (ValueError, TypeError):
                    errors.append(
                        f"line {statement.lineno}: module-level assignment must be a literal"
                    )
        elif isinstance(statement, ast.Expr):
            if not (
                isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                errors.append(
                    f"line {statement.lineno}: executable module-level expression is not allowed"
                )
        elif not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            errors.append(
                f"line {statement.lineno}: unsupported module-level statement "
                f"{statement.__class__.__name__}"
            )

    run_kernels = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_kernel"
    ]
    if len(run_kernels) != 1:
        errors.append(f"expected exactly one module-level run_kernel, found {len(run_kernels)}")
    else:
        run_kernel = run_kernels[0]
        arguments = run_kernel.args
        positional = arguments.posonlyargs + arguments.args
        names = [argument.arg for argument in positional]
        if names != RUN_KERNEL_ARGUMENTS:
            errors.append(
                "run_kernel arguments differ from the required ABI: " + repr(names)
            )
        if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
            errors.append("run_kernel cannot use variadic or keyword-only arguments")
        if arguments.defaults or arguments.kw_defaults:
            errors.append("run_kernel cannot define default arguments")
        if run_kernel.decorator_list:
            errors.append("run_kernel cannot have decorators")
        if run_kernel.returns is not None:
            errors.append("run_kernel cannot have a return annotation")
        if any(argument.annotation is not None for argument in positional):
            errors.append("run_kernel arguments cannot have annotations")
        if any(
            isinstance(node, ast.Return) and node.value is not None
            for node in ast.walk(run_kernel)
        ):
            errors.append("run_kernel must write out in place, not return a result")
        if not any(
            isinstance(node, ast.Call)
            and any(
                isinstance(argument, ast.Name) and argument.id == "out"
                for argument in node.args
            )
            for node in ast.walk(run_kernel)
        ):
            errors.append("run_kernel does not pass the supplied out tensor to a call")

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            errors.append(f"line {node.lineno}: async functions are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                errors.append(
                    f"line {node.lineno}: private or dunder attribute {node.attr!r} is not allowed"
                )
            if node.attr == "data":
                errors.append(f"line {node.lineno}: tensor.data is not allowed")
        if isinstance(node, ast.Call):
            call_name = dotted_name(node.func)
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                errors.append(
                    f"line {node.lineno}: call to {node.func.id!r} is not allowed"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in BANNED_ATTRIBUTE_CALLS
            ):
                errors.append(
                    f"line {node.lineno}: call to .{node.func.attr}() is not allowed"
                )
            if call_name == "torch.cuda.synchronize":
                errors.append(
                    f"line {node.lineno}: torch.cuda.synchronize() is not allowed"
                )
            if call_name and call_name.startswith("torch.") and call_name != "torch.empty":
                errors.append(
                    f"line {node.lineno}: PyTorch compute call {call_name} is not allowed; "
                    "only torch.empty workspace allocation is permitted"
                )

    for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        if any(dotted_name(decorator) == "T.prim_func" for decorator in function.decorator_list):
            positional = function.args.posonlyargs + function.args.args
            for argument in positional:
                if not is_tensor_annotation(argument.annotation):
                    errors.append(
                        f"line {argument.lineno}: prim_func argument {argument.arg!r} "
                        "must use T.Tensor(shape, dtype)"
                    )
                elif argument.arg == "routed_expert_weights":
                    annotation = argument.annotation
                    if not isinstance(annotation, ast.Call):
                        errors.append(
                            f"line {argument.lineno}: routed_expert_weights annotation is invalid"
                        )
                        continue
                    route_dtype = annotation.args[1]
                    if not (
                        (isinstance(route_dtype, ast.Name) and route_dtype.id == "dtype")
                        or dotted_name(route_dtype) == "T.float16"
                    ):
                        errors.append(
                            f"line {argument.lineno}: routed_expert_weights must use "
                            "the FP16 submission dtype"
                        )

    return errors


def main() -> int:
    if len(sys.argv) > 2:
        print("usage: check_moe_submission.py [submission.py]", file=sys.stderr)
        return 2
    if len(sys.argv) == 2:
        path = Path(sys.argv[1]).resolve()
    else:
        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "tilelang-moe"
            / "submission.py"
        )
    errors = check_submission(path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"PASS submission policy: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
