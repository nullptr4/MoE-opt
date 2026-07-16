#!/usr/bin/env bash
# Rebuild the pinned TileLang source after a MACA image/SDK change.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=activate-maca.sh
source "${ROOT}/scripts/activate-maca.sh"

FFI_DIR="${TILELANG_HOME}/3rdparty/tvm/3rdparty/tvm-ffi"
rm -rf "${TILELANG_HOME}/build"
(
  cd "${TILELANG_HOME}"
  python -m pip install -e . --force-reinstall --no-deps --no-build-isolation
)
rm -rf "${FFI_DIR}/build"
(
  cd "${FFI_DIR}"
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI=0.1.2 \
    python -m pip install -e . --force-reinstall --no-deps --no-build-isolation
)
exec "${ROOT}/scripts/verify-maca.sh"
