#!/usr/bin/env bash
# Run the official task-1 preliminary-round Fused MoE functional/performance tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=activate-maca.sh
source "${ROOT}/scripts/activate-maca.sh"

cd "${ROOT}/benchmarks/tilelang-moe"
exec python fusedmoe_benchmark.py "$@"
