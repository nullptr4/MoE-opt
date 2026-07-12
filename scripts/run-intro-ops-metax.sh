#!/usr/bin/env bash
# Optional TileLang training-camp launcher. Usage: ./run-intro-ops-metax.sh {env|configure|build|test|all|clean}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-env}"
# shellcheck source=activate-maca.sh
source "$ROOT/scripts/activate-maca.sh"
case "$mode" in
  test|all) "$ROOT/scripts/verify-maca.sh" ;;
  env|configure|build|clean) ;;
  *) echo "Usage: $0 {env|configure|build|test|all|clean}" >&2; exit 64 ;;
esac
export PYTHON_BIN=python
export CAMP_USE_TILELANG_METAX=1
export CAMP_TILELANG_SOURCE_ROOT="$TILELANG_REPO"
exec bash "$ROOT/materials/Intro-ops/scripts/build_metax.sh" "$mode"
