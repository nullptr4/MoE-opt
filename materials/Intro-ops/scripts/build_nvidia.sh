#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-nvidia}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
CAMP_ENABLE_CUTE="${CAMP_ENABLE_CUTE:-AUTO}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-native}"
CAMP_FORCE_RECONFIGURE="${CAMP_FORCE_RECONFIGURE:-0}"

MODE="${1:-build}"

CUTLASS_VERSION="${CUTLASS_VERSION:-v3.7.0}"
CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"

ensure_third_party() {
  local tp_dir="${ROOT}/third_party"
  if [[ -n "${CAMP_CUTLASS_ROOT:-}" ]]; then
    return
  fi
  if [[ -f "${tp_dir}/cutlass/include/cute/tensor.hpp" ]]; then
    CAMP_CUTLASS_ROOT="${tp_dir}/cutlass"
    return
  fi
  mkdir -p "${tp_dir}"
  echo "==> Fetching CUTLASS ${CUTLASS_VERSION} into third_party/cutlass ..."
  git clone --depth 1 --branch "${CUTLASS_VERSION}" "${CUTLASS_REPO}" "${tp_dir}/cutlass"
  CAMP_CUTLASS_ROOT="${tp_dir}/cutlass"
}

ensure_third_party

if [[ -z "${CAMP_CUTLASS_ROOT:-}" && -n "${CUTLASS_ROOT:-}" ]]; then
  CAMP_CUTLASS_ROOT="${CUTLASS_ROOT}"
fi

export CAMP_CUTLASS_ROOT="${CAMP_CUTLASS_ROOT:-}"

prepare_build_dir() {
  mkdir -p "${BUILD_DIR}"
  if [[ "${CAMP_FORCE_RECONFIGURE}" == "1" ]]; then
    rm -f "${BUILD_DIR}/CMakeCache.txt"
    rm -rf "${BUILD_DIR}/CMakeFiles"
  fi
}

configure() {
  prepare_build_dir
  local cmake_args=(
    -G "${CMAKE_GENERATOR}"
    "${ROOT}"
    -B "${BUILD_DIR}"
    -DCAMP_ENABLE_NVIDIA=ON
    -DCAMP_ENABLE_METAX=OFF
    -DCAMP_ENABLE_CUTE="${CAMP_ENABLE_CUTE}"
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
  )
  if [[ -n "${CAMP_CUTLASS_ROOT}" ]]; then
    cmake_args+=(-DCAMP_CUTLASS_ROOT="${CAMP_CUTLASS_ROOT}")
  fi
  "${PYTHON_BIN}" -m cmake "${cmake_args[@]}"
}

build() {
  "${PYTHON_BIN}" -m cmake --build "${BUILD_DIR}" -- -v
}

test_pytest() {
  env PYTHONPATH="${ROOT}/python:${ROOT}" CAMP_BUILD_DIR="${BUILD_DIR}" \
    "${PYTHON_BIN}" -m pytest "${ROOT}/tests" -v --backend nvidia
}

test_run_ops() {
  env PYTHONPATH="${ROOT}/python:${ROOT}" CAMP_BUILD_DIR="${BUILD_DIR}" \
    "${PYTHON_BIN}" "${ROOT}/tests/run_ops.py" --op all --backend nvidia --mode all
}

test_examples() {
  env PYTHONPATH="${ROOT}/python:${ROOT}" CAMP_BUILD_DIR="${BUILD_DIR}" \
    "${PYTHON_BIN}" "${ROOT}/examples/01_copy.py" --backend nvidia
  env PYTHONPATH="${ROOT}/python:${ROOT}" CAMP_BUILD_DIR="${BUILD_DIR}" \
    "${PYTHON_BIN}" "${ROOT}/examples/02_vector_add.py" --backend nvidia
  env PYTHONPATH="${ROOT}/python:${ROOT}" CAMP_BUILD_DIR="${BUILD_DIR}" \
    "${PYTHON_BIN}" "${ROOT}/examples/03_reduce_sum.py" --backend nvidia
  env PYTHONPATH="${ROOT}/python:${ROOT}" CAMP_BUILD_DIR="${BUILD_DIR}" \
    "${PYTHON_BIN}" "${ROOT}/examples/04_softmax.py" --backend nvidia
  env PYTHONPATH="${ROOT}/python:${ROOT}" CAMP_BUILD_DIR="${BUILD_DIR}" \
    "${PYTHON_BIN}" "${ROOT}/examples/06_relu.py" --backend nvidia
}

show_env() {
  echo "BUILD_DIR=${BUILD_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "CMAKE_GENERATOR=${CMAKE_GENERATOR}"
  echo "CAMP_ENABLE_CUTE=${CAMP_ENABLE_CUTE}"
  echo "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
  echo "CAMP_FORCE_RECONFIGURE=${CAMP_FORCE_RECONFIGURE}"
  echo "CAMP_CUTLASS_ROOT=${CAMP_CUTLASS_ROOT:-}"
}

case "${MODE}" in
  env)
    show_env
    ;;
  configure)
    configure
    ;;
  build)
    configure
    build
    ;;
  test)
    test_pytest
    test_run_ops
    test_examples
    ;;
  all)
    configure
    build
    test_pytest
    test_run_ops
    test_examples
    ;;
  clean)
    rm -rf "${BUILD_DIR}"
    ;;
  *)
    echo "usage: $0 {env|configure|build|test|all|clean}" >&2
    exit 2
    ;;
esac
