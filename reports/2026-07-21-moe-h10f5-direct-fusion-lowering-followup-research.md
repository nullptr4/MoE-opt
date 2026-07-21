# H10F5 direct-fusion/lowering 增量调研

日期：2026-07-21（UTC）

## 边界与基线

本轮唯一不可变基线是 H10F5：目标证据提交
`a6fab680739d0026d44a27c24ad16ea8155d81fd`，内核起源提交
`e60f9b3216d008e342d285cd24f8b6ede1142387`，源码 SHA-256
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。schedule 是 combined
Gate/Up，FC1/FC2 `BN128/BK64/stage1`，FC1 stage-1 `T.Pipelined`，FC2 A shared，route
FP16 fragment，以及重用 256×64 FC1 weight shared buffer 的两个 affine/interleaved
128×64 FP16 Up 半块。更旧的 A1b、H4、H8F1 和临时候选都不是正式对照。

## 新增第一手来源

Web Search 搜索词、访问状态、URL、license 和 hypothesis 映射的完整机器记录在
[`h10f5-direct-fusion-lowering-followups-20260721.json`](../data/benchmarks/c500-64g/h10f5-direct-fusion-lowering-followups-20260721.json)。
搜索摘要未被当作证据。

- N51：TileLang 官方
  [TIR elementwise op 源码](https://raw.githubusercontent.com/tile-ai/tilelang/main/tilelang/language/tir/op.py)
  和 [MIT license](https://github.com/tile-ai/tilelang/blob/main/LICENSE)。源码提供可向量化的
  `T.sigmoid`，并将 `exp` 与 `exp2` 作为不同路径；只证明接口合法，不证明 C500 性能。
- N52：Triton 官方
  [SwiGLU 核心实现](https://raw.githubusercontent.com/triton-lang/triton/main/python/triton_kernels/triton_kernels/swiglu_details/_swiglu.py)、
  [wrapper](https://raw.githubusercontent.com/triton-lang/triton/main/python/triton_kernels/triton_kernels/swiglu.py)
  和 [MIT license](https://github.com/triton-lang/triton/blob/main/LICENSE)。只借鉴直接计算
  Gate sigmoid×Gate×Up 和一次 masked store；CUDA inline assembly、launch geometry、
  `maxnreg` 不迁移到 C500。
- N53：vLLM 官方
  [activation kernel](https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/activation_kernels.cu)
  和 [Apache-2.0 license](https://github.com/vllm-project/vllm/blob/main/LICENSE)。其
  ACT(gate)×up 不物化 activated-Gate tensor；CUDA packed width 不复制，生成 MACA 代码决定向量宽度。
- N54：CUTLASS 官方 [efficient GEMM epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)、
  [predicated epilogue components](https://docs.nvidia.com/cutlass/4.3.1/media/docs/cpp/implicit_gemm_convolution.html)
  和 [BSD-3-Clause license](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt)。只借鉴
  shared exchange、elementwise work 和有界 global store 的结构，不导入 CUDA iterator 或 tensor-core 映射。
- L09：安装版 TileLang-MACA `ec48829b` 源码，MIT/Apache-2.0；`T.sigmoid` 的默认
  legalization 是 `1/(1+exp(-x))`，MACA backend 同时注册 `exp`/`exp2`。
- M17：H10F5/H11 C500 profiler 与生成源码，project-owned measured evidence；确认现有 weight
  copy/output store 已经是 `uint4`，兼容 profile 的 FC1 private read/write 为
  `921984/921984`。

## Hypothesis 与静态审查

- H12F1：保留 manual `exp2`、FP32 Gate 和 FP16 Up rounding，仅去掉 activated-Gate
  accumulator rewrite，直接在 shared product 中使用 Gate×sigmoid(Gate)。
- H12F2：不改 proven interleaved shared layout 和算术，将两个 128×64 output loop 改成一个
  128×128 predicated logical traversal，生成源码决定是否被合法合并。
- H12F3：只把 manual `exp2` sigmoid 换成安装版 `T.sigmoid`；由于实际走 `exp`，数值和性能
  风险均由 official correctness 和真实 C500 决定。
- H12S1：现有 MACA 已生成 `uint4`，H11F4 也未改变 lowering，重复 vector hint 静态排除。
- H12S2：正式 routing 只有约 0.4% padding rows，且 H5F1 已拒绝 tail specialization，静态排除。
- H12S3：fixed ABI 无 workspace/cross-CTA sync；FC1/FC2 融合需重算或 128×2048
  intermediate 加 output accumulators，静态排除。

三条实际候选都在 agent 隔离 worktree 中实现，并绑定 campaign、source IDs、parent H10F5、
candidate commit 与 diff SHA-256。外部事实、工程推断和 C500 结论分别保存在来源条目、上述
hypothesis 和结果报告中。
