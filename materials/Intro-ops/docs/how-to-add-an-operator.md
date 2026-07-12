# 如何开发一个新算子

## 目标

新增一个算子的最小闭环包括：

1. 在 `ops/<算子名>/nvidia/` 下实现 kernel 和 descriptor 生命周期。
2. 在 `python/operator_runtime/ops/` 下补 Python API。
3. 在 `tests/` 下补正确性测试和 benchmark。
4. 重新构建并验证。

训练营支持两条并行路径：
- **自定义算子**：`ops/<op>/nvidia/`，适合 kernel 教学、特殊 layout、reduce/softmax 等非逐元素结构。
- **elementwise 复用**：`ops/elementwise/<op>/nvidia/`，适合 unary/binary/broadcast 逐元素算子。

## Step 1：明确算子接口

开始前确认：

1. 几个输入、几个输出
2. 输出 shape 是否和输入一致
3. 是否要求 dtype 一致
4. 是否只支持 contiguous tensor
5. 是否需要额外参数（`dim`、`scalar` 等）
6. 是否需要 workspace

## Step 2：创建算子目录

最小文件集：

```
ops/<算子名>/nvidia/
  kernel.cuh              # kernel 实现
  <算子名>_cuda.cu        # descriptor 生命周期 + launch
include/operator_runtime/ops/<算子名>.h   # C API 头文件
python/operator_runtime/ops/<算子名>.py   # Python 绑定
tests/cases/<算子名>.py                   # 测试数据
tests/op_tests/test_<算子名>.py           # 正确性测试
tests/bench/<算子名>.py                   # benchmark
```

### include 目录结构与创建原则

```
include/operator_runtime/
  api.h               # 公共类型定义（status、dtype、tensor_view 等）
  descriptor.h        # descriptor 基类
  tensor_view.h       # tensor view 结构体
  operator_runtime.h  # 汇总头文件
  ops/
    <算子名>.h        # 每个算子一个头文件，声明四个生命周期 C 函数
  detail/
    cuda_helpers.h    # CUDA 工具函数（blocks_for、stream 转换等）
    elementwise.h     # elementwise descriptor helper
    tensor_checks.h   # tensor 校验工具
    operation.h       # operation 基类
```

创建原则：

- `ops/<算子名>.h` 是算子的公开 C API 契约，只声明四个生命周期函数（create/workspace/execute/destroy），不暴露实现细节。
- 所有函数使用 `extern "C"` + `OPRT_EXPORT`，保证 Python FFI 可以按符号名 dlsym。
- 参数类型只使用 `api.h` 中定义的公共类型（`oprt_status_t`、`oprt_tensor_view_t`、`oprt_operator_descriptor_t`、`oprt_stream_t`）。
- `detail/` 下放内部实现工具，不对外暴露，算子实现可以 include 但用户代码不应依赖。
- 新增算子只需在 `ops/` 下加一个头文件，不需要修改其他头文件。

## Step 3：实现 NVIDIA 后端

一个算子需要四个生命周期接口：

1. `oprt_create_<op>_descriptor` — 检查输入合法性，保存运行信息
2. `oprt_get_<op>_workspace_size` — 返回临时内存大小
3. `oprt_execute_<op>` — 按 dtype dispatch 并 launch kernel
4. `oprt_destroy_<op>_descriptor` — 释放 descriptor

参考 `copy` 的实现：kernel 放 `kernel.cuh`，生命周期放 `<op>_cuda.cu`。

如果是 elementwise 算子，复用 `ops/common/elementwise/nvidia/elementwise_nvidia.cuh` 和 `include/operator_runtime/detail/elementwise.h`，只需提供 device functor。参考 `relu`。

## Step 4：补 Python 绑定

Python 入口放在 `python/operator_runtime/ops/<算子名>.py`，通常暴露三个接口：

- `<op>` — out-of-place，自动分配输出
- `<op>_` — out-variant，调用方提供输出 tensor
- `prepare_<op>` — 创建 descriptor，支持多次执行复用

新增后在 `ops/__init__.py` 和 `operator_runtime/__init__.py` 中导出。

## Step 5：补测试

`tests/cases/<算子名>.py` 组织测试数据，分三类：

- `correctness_cases()` — 正确性用例（shape、dtype、tolerance）
- `api_error_cases()` — 异常输入用例
- `benchmark_cases()` — 性能测试规模

`tests/op_tests/test_<算子名>.py` 负责：

- 正确性对比（vs PyTorch）
- API contract 检查（shape/dtype 不匹配、非 contiguous）
- prepared 执行复用检查

`tests/bench/<算子名>.py` 负责性能入口。

## Step 6：构建和验证

```bash
# 新增 .cu 后必须重新 configure
bash scripts/build_nvidia.sh configure
bash scripts/build_nvidia.sh build

# 单算子验证
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_<算子名>.py -v --backend nvidia

# 正确性 + benchmark
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op <算子名> --backend nvidia --mode all

# 确认没有破坏已有算子
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op all --backend nvidia --mode test
```

## 现有模板参考

| 模板 | 适用场景 |
| --- | --- |
| `copy` | 最简单的单输入单输出自定义算子 |
| `vector_add` | 双输入 contiguous 自定义算子 |
| `relu` | elementwise 框架复用，含标量参数 |
| `reduce_sum` | 带 reduce 维度的算子 |
| `softmax` | 多步归约 + 归一化 |
