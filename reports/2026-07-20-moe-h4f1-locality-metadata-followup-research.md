# H4F1 locality/metadata 后续：增量第一手调研与静态审查

日期：2026-07-20（UTC）

## 范围与基线

本轮唯一不可变基线仍是最新正式胜者 H4F1：kernel-origin commit
`61fc0bbdb4a08c0a4ba44af278606ce362b794ce`，测量时 target evidence commit
`69bd0ec5421aebbf697f99ef6798f47e378ec487`，源码 SHA-256
`2be93d4210d7d3bdda857887c70540c6c4da964946033fa95e58b114dd1fe47e`。schedule 是
combined Gate/Up；FC1 `BN128/BK64/stage1`；FC2 `BN128/BK64/stage1`；FC2 A shared；
route weight FP16 fragment cache。A1b、E3+E7、H4 及 H1–H6 只作历史边界，不作性能对照。

检索使用本会话原生 Web Search，并实际打开官方正文/源码；访问日期均为 2026-07-20。
搜索摘要不作证据，CUDA/Triton 的定量结论不外推到 C500。精确查询、URL、访问状态、
许可证和 hypothesis 映射也保存在机器可读结果中。

## 新增第一手来源

| ID | 查询、正文、许可证 | 外部事实 | H4F1/C500 适用边界 |
|---|---|---|---|
| N24 | `site:github.com/tile-ai/tilelang rasterization2DRow use_swizzle threadblock swizzle source`；[TileLang 官方仓库](https://github.com/tile-ai/tilelang)，MIT | 官方 GEMM 示例把 `T.use_swizzle` 描述为可选的 L2 locality rasterization。 | 只借鉴 locality 原则；实际顺序以安装 MACA 源码 L04 为准。 |
| N25 | `site:github.com/triton-lang/triton grouped matmul GROUP_SIZE_M L2 cache tutorial source`；[Triton 官方 matmul tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py)，MIT | grouped program ordering 可在切换另一输出维前保持一侧 operand tile 的局部性。 | 不转移 Triton program-ID、CUDA cache 或 A100 数值；只支持 exact row/column 因果比较。 |
| N26 | `site:github.com/tile-ai/tilelang T.copy fragment fragment vectorized source copy op`；[TileLang 官方 copy_op.py](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/copy_op.py)，MIT | `T.copy` 接受匹配的 `BufferRegion`，循环和 vector lowering 交给后端。 | 安装 MACA 提供相同 API；是否改善只能由生成代码与 C500 endpoint 确认。 |
| N27 | `site:github.com/tile-ai/tilelang shared metadata tx == 0 T.sync_threads source`；[TileLang 官方 DeepSeek top-k 示例](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py)，MIT | 示例合法使用 `tx==0` 更新 shared scalar，再 `T.sync_threads` 供 CTA 消费。 | 只证明语法/同步合法，不预言 barrier 与 metadata read 的 C500 代价。 |

## 安装后端与 measured follow-up

本机 `/opt/tilelang-metax@ec48829bb61fcd55a366407c69355581733cceef`（MIT）中的
`src/tl_templates/maca/threadblock_swizzle.h` 证明：当前二维 grid 中 row16 在固定 logical M
时遍历 N，优先保留 FC2 shared A locality；column16 在固定 N 时遍历 M，优先保留 expert
weight locality。`copy_op.py` 接受 fragment region。生成的 H4F1 源码还显示 routing metadata
标量由每个 thread 重复加载。以上是 capability 事实，不是性能结论。

Measured source M08 是 H4F1 profile 与历史 pre-H4 swizzle 实验。历史 FC2 column16 已比
row16 慢，继续视为 rejected；仅因 H4 把 FC2 A 改到 shared、H4F1 又改变 route reads，允许
精确复审 column16 一点，并明确标成 resource-changing follow-up，不重扫 panel/warp/grid。

## Hypothesis 与静态/数值审查

| ID | 机制 | 来源 | 风险与处理 |
|---|---|---|---|
| H7F1 | thread0 一次加载五个 routing metadata int32，写入 20-byte shared tile，单次 CTA sync 后复用 | N27/L04/M08 | 整数和值/地址算术不变；必须由 profiler 证明 read 收益大于 barrier/shared 代价；实测。 |
| H7F2 | 仅把 FC1 combined accumulator 的 Up half 手写 fragment loop 换成匹配 region 的 `T.copy` | N26/L04/M08 | auxiliary 仍为 FP32，不重复 H6 alias 或 A1c FP16；实测。 |
| H7F3 | H4F1 后只复审 FC2 row16→column16 | N24/N25/L04/M08 | 只改 launch order；历史方向 follow-up，不作 grid；实测。 |
| H7F4 | H7F2 证明 spill-free copy lowering 后，只把完整 FC2 route global→FP16 fragment load 换成 `T.copy`，尾块保留 predicate | N26/L04/M08 | 不同于 H5F5 的 predicate removal；route value、scale/store 不变；实测。 |
| H7S1 | 一个 CTA 计算两个 FC2 M/N output tile 以复用一个 operand | N25/L04/M08 | 两个活跃 128×128 FP32 accumulator 重复已测 BN256 spill；串行计算则重复 persistent serialization，静态排除。 |

H7F1–H7F4 均保持 RoutedMoEKernel API、十参数 submission ABI、shape、seed、dtype、
FP32 accumulate 与 `atol=rtol=0.01` 不变。每个实际候选只能在官方 Large/Small 0 mismatch
后进入 warmup 10 / timing 100 / outlier none 与独立四-launch mcProfiler。
