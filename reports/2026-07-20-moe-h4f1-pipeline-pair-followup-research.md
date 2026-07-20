# H4F1 pipeline/pair-layout 后续：增量第一手调研与假设

日期：2026-07-20（UTC）

## 唯一基线与去重

本轮硬件对照只有 H4F1：target evidence commit
`ceed231329dd04321546b477afe181f85e2aee88`，kernel-origin commit
`61fc0bbdb4a08c0a4ba44af278606ce362b794ce`，源码 SHA-256
`2be93d4210d7d3bdda857887c70540c6c4da964946033fa95e58b114dd1fe47e`。其 schedule 为
combined Gate/Up，FC1/FC2 `BN128/BK64/stage1`，FC2 A shared，route weight FP16 fragment。
A1b、H4、E3+E7 和 H1–H7 只作历史边界，不作正式性能对照。

硬件前已审计 agent 的 40 项 E1–H7S1 tried-direction ledger 和 measured knowledge。本轮不重做
FC2 stage2/serial、H6 half-major fragment view、fragment `T.copy`、route predicate、column
swizzle、metadata broadcast、BK/BN/occupancy 网格或 persistent CTA。

## 新增第一手来源

检索使用原生 Web Search，并在 2026-07-20 实际打开正文或源码；搜索摘要不是证据。精确 query、
访问状态、许可证与 hypothesis 映射位于机器可读结果。

| ID | 第一手来源 / 许可证 | 外部事实 | H4F1/C500 适用边界 |
|---|---|---|---|
| N28 | [TileLang 0.1.12 `Pipelined` API](https://tilelang.com/autoapi/tilelang/language/loop/index.html)，MIT | `T.Pipelined` 可给 copy/GEMM 语句建立 pipeline，并支持 `num_stages` 或显式 order/stage。 | 本机安装 MACA fork 有同一 API；只测试 FC1 stage1 和一个 copy-order 点，不重开 stage 网格。 |
| N29 | [NVIDIA MoE fusion article](https://developer.nvidia.com/blog/boosting-moe-training-throughput-with-advanced-fusion-kernels/)，NVIDIA website terms | interleaved Gate/Input columns 可让同一 CTA 消费两半并融合 GLU epilogue。 | 固定 ABI 不允许离线重排；只独立尝试 on-chip pair-last，未复制代码或 NVIDIA 数值。 |
| N30 | [CUDA Programming Guide async/prefetch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)，NVIDIA documentation terms | 把未来 global read 提前到独立计算前可隐藏部分延迟；真 async copy 是 target-specific。 | 不向 C500 转移 CUDA 指令或延迟假设；H8F2 只移动已有同步 route load。 |
| N31 | [CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)，BSD-3-Clause | accumulator 最优布局与 elementwise/global epilogue 布局可不同。 | 只支持审查 pair-last 方向；不复制 CUDA layout。 |
| N32 | [MetaX developer portal](https://developer.metax-tech.com/) | 增量查询未获得可访问的 C500/MACA pipeline/prefetch 正文。 | 无正文不构成能力证据；async 和 resident-grid 继续排除。 |

安装源码 L05 是 `/opt/tilelang-metax@ec48829bb61fcd55a366407c69355581733cceef`
（MIT）的 `loop.py`、pipeline planning/injection、layout checks 与生成内核。它证明 API 和
baseline dataflow，不证明性能。measured source M09 记录：把 FC2 stage1 pipeline 改为 serial
会使 mean/median 回退 `0.3721%/0.4810%`；最新 H4F1 FC1 约 `126M cycles`，FC2 约
`51–53M cycles`，baseline private traffic 为 0。

## 外部事实、工程推断与待测假设

- 外部事实：TileLang 有 stage-aware `Pipelined`；通用 GEMM/MoE 实现会考虑 prefetch 与
  accumulator/epilogue layout 分离。
- 工程推断：H4F1 的 dominant FC1 仍是 serial，而同一安装后端在 FC2 的 stage1 form 已有
  正向 measured evidence，因此只对 FC1 做精确逆向 follow-up；route load 提前与 pair-last
  需要本机编译器和 profiler 判定。
- C500 结论：只有本轮 correctness、10/100 endpoint 和独立 mcProfiler 完成后形成，见结果报告。

| ID | 机制 | 数值/资源审查 | 状态 |
|---|---|---|---|
| H8F1 | combined FC1 的 `T.serial` 改为同 buffer、同语句顺序的 `T.Pipelined(..., num_stages=1)` | K/GEMM/精度顺序不变；必须检查 private spill 与 FC1 cycles | 实测 |
| H8F2 | 把已有 predicated FP16 route fragment load 移到 FC2 GEMM 前 | value/predicate/product 不变；fragment lifetime 变长，必须无新增 spill | 实测 |
| H8F3 | Gate/Up on-chip pair-last shared/accumulator，直接 FP32 SwiGLU，移除 auxiliary fragment | ABI 不变但 layout inference 敏感；build 先于 correctness | 实测 build |
| H8F4 | existing FC2 stage1 loop 内先 copy down weight，再 copy activation | 仅 copy issue order；作为 H8F3 硬失败后的不同机制替补 | 实测 |
| H8S1 | `prefer_async` / target async global-to-shared | TileLang 文档只给 PTX hint，本机无已证 MACA 原语 | 静态排除 |
| H8S2 | 离线永久 interleave Gate/Up | submission ABI 给出两个独立 tensor | 静态排除 |

全部实际候选绑定 campaign、hypothesis、source IDs、H4F1 parent 和 diff hash，在隔离 worktree
生成；保持 RoutedMoEKernel、十参数 submission ABI、官方 seed/shape、FP16 input、FP32
accumulate、`atol=rtol=0.01` 不变。只有 Large/Small 0 mismatch 后才进入 warmup 10 / repeat
100 / outlier none 和四-launch mcProfiler。
