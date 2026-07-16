#!/usr/bin/env bash
# Run the standalone padded-token ABI check for the OJ submission.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=activate-maca.sh
source "${ROOT}/scripts/activate-maca.sh"

cd "${ROOT}/benchmarks/tilelang-moe"
exec python test_moe_submission.py "$@"
