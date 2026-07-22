#!/usr/bin/env bash
# Run the default ten-argument submission remote matrix, or an explicit formal guard.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=activate-maca.sh
source "${ROOT}/scripts/activate-maca.sh"

exec python "${ROOT}/scripts/run_moe_evaluation.py" "$@"
