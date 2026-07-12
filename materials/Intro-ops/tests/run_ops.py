from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _discover_ops() -> tuple[str, ...]:
    ops_dir = ROOT / "tests" / "op_tests"
    bench_dir = ROOT / "tests" / "bench"
    names: set[str] = set()

    for path in ops_dir.glob("test_*.py"):
        if path.name == "__init__.py":
            continue
        names.add(path.stem.removeprefix("test_"))

    for path in bench_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        names.add(path.stem)

    return tuple(sorted(names))


def _format_table(rows: list[list[str]]) -> str:
    headers = ["kind", "op", "backend", "status", "target"]
    widths = []
    for idx, header in enumerate(headers):
        content_width = max((len(row[idx]) for row in rows), default=0)
        widths.append(max(len(header), content_width))

    def fmt(cols: list[str]) -> str:
        return " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(cols))

    separator = "-+-".join("-" * width for width in widths)
    lines = [fmt(headers), separator]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def _format_bench_table(rows) -> str:
    headers = [
        "operator",
        "backend",
        "shape",
        "dtype",
        "runtime_ms",
        "torch_ms",
        "speedup",
    ]
    body = []
    for row in rows:
        speedup = "-" if row.speedup is None else f"{row.speedup:.2f}"
        torch_ms = "-" if row.torch_ms is None else f"{row.torch_ms:.4f}"
        body.append(
            [
                row.operator,
                row.backend,
                row.shape,
                row.dtype,
                f"{row.runtime_ms:.4f}",
                torch_ms,
                speedup,
            ]
        )

    widths = []
    for idx, header in enumerate(headers):
        content_width = max((len(r[idx]) for r in body), default=0)
        widths.append(max(len(header), content_width))

    def fmt(cols: list[str]) -> str:
        return " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(cols))

    separator = "-+-".join("-" * width for width in widths)
    lines = [fmt(headers), separator]
    lines.extend(fmt(cols) for cols in body)
    return "\n".join(lines)


def _run_pytest(path: str, backend: str) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", path, "-q", "-rs", "--backend", backend]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    output = result.stdout.strip()
    summary = output.splitlines()[-1] if output else f"exit={result.returncode}"
    if result.returncode == 0:
        return True, summary
    tail = result.stderr.strip() or output or f"exit={result.returncode}"
    return False, tail.splitlines()[-1]


def _run_bench_module(op: str, backend: str):
    module = __import__(f"tests.bench.{op}", fromlist=[f"bench_{op}"])
    fn = getattr(module, f"bench_{op}")
    return fn(backend)


def main() -> int:
    ops = _discover_ops()
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=[*ops, "all"], default="all")
    parser.add_argument("--backend", choices=["nvidia", "tilelang", "metax"], default="nvidia")
    parser.add_argument("--mode", choices=["test", "bench", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ("bench", "all") and not torch.cuda.is_available():
        print("CUDA is required for benchmark", file=sys.stderr)
        return 2

    selected_ops = ops if args.op == "all" else (args.op,)
    rows: list[list[str]] = []
    bench_rows = []
    failed = False

    for op in selected_ops:
        if args.mode in ("test", "all"):
            ok, detail = _run_pytest(f"tests/op_tests/test_{op}.py", args.backend)
            rows.append(["test", op, args.backend, "ok" if ok else "fail", detail])
            failed = failed or not ok
        if args.mode in ("bench", "all"):
            try:
                results = _run_bench_module(op, args.backend)
            except Exception as exc:  # pragma: no cover - CLI surface
                rows.append(["bench", op, args.backend, "fail", str(exc)])
                failed = True
            else:
                rows.append(["bench", op, args.backend, "ok", f"{len(results)} rows"])
                bench_rows.extend(results)

    print(_format_table(rows))
    if bench_rows:
        print()
        print(_format_bench_table(bench_rows))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
