# Intro-ops 训练营

面向 kernel 开发的 nano 级 GPU 算子运行时。框架现在采用后端无关的公共 C API，并提供独立的 NVIDIA 与 MetaX 构建变体；Python、测试和 benchmark 都建立在同一套运行时契约之上。

## 快速开始

```bash
# 激活环境
conda activate py312

# 构建 NVIDIA（首次运行自动拉取 CUTLASS）
bash scripts/build_nvidia.sh build

# 运行全部 NVIDIA 测试
bash scripts/build_nvidia.sh test

# 构建 MetaX
bash scripts/build_metax.sh build

# 运行 MetaX 测试
bash scripts/build_metax.sh test

# 清理构建
bash scripts/build_nvidia.sh clean
bash scripts/build_metax.sh clean
```

## 常用命令

### 构建脚本模式

```bash
bash scripts/build_nvidia.sh env          # 显示当前构建环境变量
bash scripts/build_nvidia.sh configure    # 仅 cmake 配置
bash scripts/build_nvidia.sh build        # 配置 + 编译
bash scripts/build_nvidia.sh test         # 运行 pytest + run_ops + examples
bash scripts/build_nvidia.sh all          # 编译 + 测试
bash scripts/build_nvidia.sh clean        # 删除构建目录

bash scripts/build_metax.sh env           # 显示当前 MACA 构建环境
bash scripts/build_metax.sh configure     # 仅 cmake 配置
bash scripts/build_metax.sh build         # 配置 + 编译
bash scripts/build_metax.sh test          # 运行 pytest + run_ops + examples
bash scripts/build_metax.sh all           # 编译 + 测试
bash scripts/build_metax.sh clean         # 删除构建目录
```

### 单算子测试

```bash
# 单算子正确性测试
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_copy.py -v

# run_ops 支持 --mode: test, bench, all
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode test
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode bench
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op copy --backend nvidia --mode all

# 全部算子
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op all --backend nvidia --mode all

# MetaX
PYTHONPATH=python:. CAMP_BUILD_DIR=build-metax pytest tests/op_tests/test_copy.py -v --backend metax
PYTHONPATH=python:. CAMP_BUILD_DIR=build-metax python tests/run_ops.py --op all --backend metax --mode all
```

### 强制重新构建

```bash
# 清除 cmake 缓存并重新构建
CAMP_FORCE_RECONFIGURE=1 bash scripts/build_nvidia.sh build

# 指定 GPU 架构
CMAKE_CUDA_ARCHITECTURES=89 bash scripts/build_nvidia.sh build
```

## 项目结构

```
include/operator_runtime/
  operator_runtime.h  <-- 公共运行时总头
  ops/<op>.h          <-- 后端无关的公开 C API
  detail/*.h          <-- 共享内部 helper 分层
ops/<op>/nvidia/
  kernel.cuh          <-- 在这里实现你的 kernel
  <op>_cuda.cu        <-- 统一公共符号实现
ops/<op>/metax/
  <op>_metax.maca     <-- MetaX kernel + launch 实现
ops/elementwise/<op>/
  nvidia/*.cu         <-- elementwise NVIDIA 实现
  metax/*.maca        <-- elementwise MetaX 实现
python/operator_runtime/
  ops/<op>.py         <-- 基于共享 C API / TileLang 的 Python 包装
tests/
  op_tests/test_<op>.py   <-- 正确性测试
  cases/<op>.py           <-- 测试用例
  bench/<op>.py           <-- 性能基准
```

## 架构说明

- 公开算子头文件位于 `include/operator_runtime/ops/*.h`，暴露统一的后端无关符号，例如 `oprt_create_copy_descriptor`。
- 后端私有实现位于 `ops/.../<backend>/`。
- Python 绑定对 NVIDIA 和 MetaX 复用同一套公开符号，通过 `oprt_set_backend(...)` 和独立构建产物选择后端。

## 算子列表

| 算子 | 难度 | 核心概念 |
| --- | --- | --- |
| `copy` | 入门 | grid-stride loop、向量化内存访问 |
| `vector_add` | 入门 | 逐元素计算 |
| `reduce_sum` | 中等 | shared memory、warp 归约 |
| `softmax` | 进阶 | 多趟归约 + 归一化 |
| `relu` | 参考实现 | elementwise 框架（已实现） |

## 开发流程

1. 阅读 `ops/<op>/nvidia/kernel.cuh` 中的 kernel 骨架
2. 实现 TODO kernel
3. 构建：`bash scripts/build_nvidia.sh build`
4. 测试：`PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_<op>.py -v`
5. Benchmark：`PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op <op> --backend nvidia --mode bench`

## 新增算子

完整指南见 [docs/how-to-add-an-operator.md](docs/how-to-add-an-operator.md)。最小步骤：

1. 创建 `ops/<op>/nvidia/`，包含 kernel 和 cuda 源文件
2. 在 `python/operator_runtime/ops/<op>.py` 添加 Python 绑定
3. 在 `tests/` 下添加测试用例、正确性测试和 benchmark
4. 新增 NVIDIA `.cu` 文件后，重新运行 `bash scripts/build_nvidia.sh configure`
5. 构建并验证

## MetaX 后端

训练营同时支持 MetaX 构建变体，可通过 `bash scripts/build_metax.sh ...` 使用。

```bash
# 构建 MetaX 变体
bash scripts/build_metax.sh build

# 运行 MetaX 正确性与 benchmark 流程
bash scripts/build_metax.sh test
```

## MetaX 上的 TileLang

本机不能直接使用 pip 官方版 `tilelang`。只有在需要验证 MetaX 上的 TileLang 时，才使用源码构建的 `/root/tilelang-metax`。

```bash
# 先单独把 /root/tilelang-metax 以 USE_MACA=ON 构建完成

CAMP_USE_TILELANG_METAX=1 \
CAMP_TILELANG_SOURCE_ROOT=/root/tilelang-metax \
bash scripts/build_metax.sh test
```

该模式会导出：
- `PYTHONPATH=/root/tilelang-metax`
- `LD_LIBRARY_PATH=/root/tilelang-metax/build/lib:/opt/maca/lib:...`

并在 MetaX 上验证 `copy/vector_add/reduce_sum/softmax` 的 `backend=tilelang`。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | 目标 GPU 架构（如 L40 用 `89`） |
| `CAMP_FORCE_RECONFIGURE` | `0` | 设为 `1` 在配置前清除 CMake 缓存 |
| `CAMP_ENABLE_CUTE` | `AUTO` | CuTe/CUTLASS 支持：`AUTO`、`ON`、`OFF` |
| `BUILD_DIR` | `build-nvidia` | 构建输出目录 |
| `CAMP_BUILD_DIR` | — | 告诉 Python 在哪里找 `libcamp_ops.so` |
| `CAMP_CUTLASS_ROOT` | — | 覆盖 CUTLASS 路径（跳过自动拉取） |
| `CAMP_USE_TILELANG_METAX` | `0` | 设为 `1` 时在 `build_metax.sh` 中执行 MetaX 上的 TileLang 验证 |
| `CAMP_TILELANG_SOURCE_ROOT` | `/root/tilelang-metax` | 用于 MetaX TileLang 验证的源码版 TileLang 根目录 |
