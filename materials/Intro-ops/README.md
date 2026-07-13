# Operator Runtime Training Camp

A nano GPU operator runtime for learning kernel development. The framework uses a backend-agnostic public C API with separate NVIDIA and MetaX build variants, while Python, tests, and benchmarks sit on top of the same runtime contract.

## Quick Start

```bash
# activate environment
conda activate py312

# build NVIDIA (auto-fetches CUTLASS on first run)
bash scripts/build_nvidia.sh build

# run all NVIDIA tests
bash scripts/build_nvidia.sh test

# build MetaX
bash scripts/build_metax.sh build

# run MetaX tests
bash scripts/build_metax.sh test

# clean build
bash scripts/build_nvidia.sh clean
bash scripts/build_metax.sh clean
```

## Common Commands

### Build Script Modes

```bash
bash scripts/build_nvidia.sh env          # show current build environment variables
bash scripts/build_nvidia.sh configure    # cmake configure only
bash scripts/build_nvidia.sh build        # configure + build
bash scripts/build_nvidia.sh test         # run pytest + run_ops + examples
bash scripts/build_nvidia.sh all          # build + test
bash scripts/build_nvidia.sh clean        # remove build directory

bash scripts/build_metax.sh env           # show current MACA build environment
bash scripts/build_metax.sh configure     # cmake configure only
bash scripts/build_metax.sh build         # configure + build
bash scripts/build_metax.sh test          # run pytest + run_ops + examples
bash scripts/build_metax.sh all           # build + test
bash scripts/build_metax.sh clean         # remove build directory
```

### Single Operator Testing

```bash
# correctness test for one operator
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_copy.py -v

# run_ops supports --mode: test, bench, all
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode test
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode bench
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode all

# all operators
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op all --backend nvidia --mode all

# MetaX
PYTHONPATH=python:. CAMP_BUILD_DIR=build-metax pytest tests/op_tests/test_copy.py -v --backend metax
PYTHONPATH=python:. CAMP_BUILD_DIR=build-metax python tests/run_ops.py --op all --backend metax --mode all
```

### Force Rebuild

```bash
# clear cmake cache and rebuild
CAMP_FORCE_RECONFIGURE=1 bash scripts/build_nvidia.sh build

# target specific GPU architecture
CMAKE_CUDA_ARCHITECTURES=89 bash scripts/build_nvidia.sh build
```

## Project Structure

```
include/operator_runtime/
  operator_runtime.h  <-- umbrella public runtime header
  ops/<op>.h          <-- backend-agnostic public C API
  detail/*.h          <-- shared internal helper layer
ops/<op>/nvidia/
  kernel.cuh          <-- implement your kernel here
  <op>_cuda.cu        <-- unified public symbol implementation
ops/<op>/metax/
  <op>_metax.maca     <-- MetaX kernel + launch implementation
ops/elementwise/<op>/
  nvidia/*.cu         <-- elementwise NVIDIA implementations
  metax/*.maca        <-- elementwise MetaX implementations
python/operator_runtime/
  ops/<op>.py         <-- Python wrapper over shared C API / TileLang
tests/
  op_tests/test_<op>.py   <-- correctness tests
  cases/<op>.py           <-- test cases
  bench/<op>.py           <-- benchmarks
```

## Architecture

- Public operator headers live under `include/operator_runtime/ops/*.h` and expose backend-agnostic symbols such as `oprt_create_copy_descriptor`.
- Backend-private code lives under `ops/.../<backend>/`.
- Python bindings use the same public symbol names for both compiled backends and select the active backend via `oprt_set_backend(...)` plus separate build outputs.

## Operators

| Operator | Difficulty | Key Concept |
| --- | --- | --- |
| `copy` | easy | grid-stride loop, vectorized memory access |
| `vector_add` | easy | elementwise computation |
| `reduce_sum` | medium | shared memory, warp reduction |
| `softmax` | hard | multi-pass reduction + normalization |
| `relu` | reference | elementwise framework (already implemented) |

## Workflow

1. Read the kernel scaffold in `ops/<op>/nvidia/kernel.cuh`
2. Implement the TODO kernel
3. Build: `bash scripts/build_nvidia.sh build`
4. Test: `PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_<op>.py -v`
5. Benchmark: `PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op <op> --backend nvidia --mode bench`

## Adding a New Operator

See [docs/how-to-add-an-operator.md](docs/how-to-add-an-operator.md) for the full guide. Minimal steps:

1. Create `ops/<op>/nvidia/` with kernel and cuda source
2. Add Python bindings in `python/operator_runtime/ops/<op>.py`
3. Add test cases, correctness tests, and benchmarks under `tests/`
4. Re-run `bash scripts/build_nvidia.sh configure` when adding new NVIDIA `.cu` sources
5. Build and verify

## MetaX Backend

The training runtime also supports a MetaX build variant through `bash scripts/build_metax.sh ...`.

```bash
# build MetaX variant
bash scripts/build_metax.sh build

# run MetaX correctness + benchmark flow
bash scripts/build_metax.sh test
```

## TileLang on MetaX

Stock pip `tilelang` is not sufficient for this machine. Use the source-built `/root/tilelang-metax` tree for TileLang-on-MetaX validation when needed.

```bash
# build /root/tilelang-metax separately with USE_MACA=ON

CAMP_USE_TILELANG_METAX=1 \
CAMP_TILELANG_SOURCE_ROOT=/root/tilelang-metax \
bash scripts/build_metax.sh test
```

This mode exports:
- `PYTHONPATH=/root/tilelang-metax`
- `LD_LIBRARY_PATH=/root/tilelang-metax/build/lib:/opt/maca/lib:...`

and validates `copy/vector_add/reduce_sum/softmax` with `backend=tilelang` on MetaX.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Target GPU arch (e.g. `89` for L40) |
| `CAMP_FORCE_RECONFIGURE` | `0` | Set `1` to clear CMake cache before configure |
| `CAMP_ENABLE_CUTE` | `AUTO` | CuTe/CUTLASS support: `AUTO`, `ON`, `OFF` |
| `BUILD_DIR` | `build-nvidia` | Build output directory |
| `CAMP_BUILD_DIR` | — | Tell Python where to find `libcamp_ops.so` |
| `CAMP_CUTLASS_ROOT` | — | Override CUTLASS path (skips auto-fetch) |
| `CAMP_USE_TILELANG_METAX` | `0` | Set `1` to run TileLang-on-MetaX validation in `build_metax.sh` |
| `CAMP_TILELANG_SOURCE_ROOT` | `/root/tilelang-metax` | Source-built TileLang tree used for MetaX TileLang validation |
