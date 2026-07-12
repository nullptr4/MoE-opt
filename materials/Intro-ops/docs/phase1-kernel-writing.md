# 第一阶段训练目标：写 Kernel

## 目标

本阶段主要需要做一件事：

**给现有算子写 kernel 文件。**

其余所有代码（descriptor 生命周期、Python 绑定、测试、构建）都已经写好，剩下需要补全计算逻辑本身。

默认要求是：

1. **优先把 kernel 写对、跑通。**
2. 不需要一开始就追求最优性能。
3. 先完成 `kernel.cuh` / `kernel.py` 里的 TODO，再考虑更激进的优化。

如果你已经完成基础目标，并且学有余力，希望进一步冲性能，那么可以继续修改对应算子的包装层文件（例如 NVIDIA 后端的 `.cu` 文件），调整 launch policy、切换不同 kernel、加入更激进的 specialization。这些都属于进阶优化内容，不属于第一阶段的必做要求。

---

## 需要写的文件

每个算子有两种后端，对应两个 kernel 文件：

| 算子 | NVIDIA kernel | TileLang kernel |
|------|--------------|-----------------|
| `copy` | `ops/copy/nvidia/kernel.cuh` | `ops/copy/tilelang/kernel.py` |
| `vector_add` | `ops/vector_add/nvidia/kernel.cuh` | `ops/vector_add/tilelang/kernel.py` |
| `reduce_sum` | `ops/reduce_sum/nvidia/kernel.cuh` | `ops/reduce_sum/tilelang/kernel.py` |
| `softmax` | `ops/softmax/nvidia/kernel.cuh` | `ops/softmax/tilelang/kernel.py` |

建议按从简到难的顺序：copy → vector_add → reduce_sum → softmax。

---

## NVIDIA Kernel（`.cuh` 文件）

### 写什么

一个 `__global__` 函数，放在对应算子的命名空间下。函数只负责计算逻辑，输入输出是裸指针，不涉及任何 descriptor 或 API。

本阶段默认**不需要修改**同目录下的 `.cu` 文件。仓库已经提供了兼容性优先的 launch 样板，这一阶段只要求把 `kernel.cuh` 写对、跑通。

如果你已经完成基础目标，并且想继续冲性能，可以把 `.cu` 当作进阶优化层：在那里调整线程数、切换不同 kernel、加入更激进的 specialization。但这些不属于第一阶段的必做内容。

### 四个算子的难度递进

**copy / vector_add**：逐元素操作，用 grid-stride loop 模式。每个线程负责若干个独立的元素，线程之间没有通信。

**reduce_sum**：行规约。每行一个 block，block 内线程先各自累加自己负责的列，再通过 shared memory 做树形归约，最终 `smem[0]` 是整行的结果。需要用 `__syncthreads()` 同步。

**softmax**：基础版本使用三段流程：第一遍求行最大值（数值稳定性），第二遍求 `exp(x - max)` 的和并把中间结果写到输出，第三遍再除以 sum。每一遍都需要 block 内同步。

### 需要理解的概念

- **grid-stride loop**：为什么这样写可以处理任意大小的 tensor
- **shared memory 规约**：树形归约的每一步在做什么，为什么需要 `__syncthreads()`
- **softmax 减 max**：为什么直接算 `exp(x)` 会出问题，减去行最大值为什么不改变结果

---

## TileLang Kernel（`.py` 文件）

### 写什么

一个用 `@tilelang.jit` 装饰的 Python 函数，用 TileLang DSL 描述 tile 粒度的计算逻辑。TileLang 会把它编译成真正的 CUDA kernel。

和 NVIDIA 一样，本阶段默认只需要填写 `kernel.py` 里的 TODO，不需要修改外层适配代码。

### 四个算子的难度递进

**copy**：直接用内置 `T.copy` 在两个 tile 之间搬数据，无需手写循环。

**vector_add**：用 `T.Parallel` 循环在 tile 内逐元素计算，理解 `T.Parallel` 与 `T.Serial` 的区别。

**reduce_sum**：需要分块累加。外层用 `T.Serial` 顺序遍历列方向的分块（因为要累积状态），内层用 `T.reduce_sum` 对 fragment 做规约。

**softmax**：两遍扫描，使用 online softmax 算法。第一遍滚动维护 log-sum-exp 状态，第二遍用最终的 lse 归一化。计算中用 `exp2` / `log2` 替代 `exp` / `log`。

### 需要理解的概念

- **`T.Parallel` vs `T.Serial`**：什么情况下循环内的迭代可以并行，什么情况下必须顺序
- **`T.alloc_fragment`**：tile 级别的局部 buffer，对应寄存器或 shared memory
- **`T.copy`**：把全局内存的一块搬到 fragment，不是逐元素赋值
- **online softmax**：为什么一遍扫描就能算出正确的归一化，log-sum-exp 的滚动更新逻辑
- **`exp2` / `log2`**：为什么 TileLang 里用这两个而不是 `exp` / `log`

---

## 验证方式

每写完一个 kernel，用对应的测试验证：

```bash
# NVIDIA kernel（需要先重新编译）
bash scripts/build_nvidia.sh build
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_<算子名>.py -v --backend nvidia

# TileLang kernel（不需要编译）
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia pytest tests/op_tests/test_<算子名>.py -v --backend tilelang

# 同时跑正确性 + benchmark
PYTHONPATH=python:. CAMP_BUILD_DIR=build-nvidia python tests/run_ops.py --op <算子名> --backend nvidia --mode all
```

四个算子两种后端全部通过，阶段一完成。
