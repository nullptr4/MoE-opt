# H8F1 epilogue/register follow-up：增量调研与假设

日期：2026-07-20（UTC）

## 唯一基线与去重

本轮唯一正式对照是已经接受并落地的 H8F1：目标提交
`0ae63ec7f61e29a3da6494d33c20cde3365da0e0`，内核 SHA-256
`6543398edf4e48c95c896b5a703c1a58f3f49a8a7dec543cb892cb0467aea7a3`。schedule 是
combined Gate/Up FC1 的 same-buffer stage-1 `T.Pipelined`，FC2 A shared、route weight FP16
fragment，FC1/FC2 均 BN128/BK64/stage1。A1b、H4、H4F1 和临时候选不作本轮正式对照。

硬件测量前已审计 46 项 tried-direction ledger、measured knowledge、H8F1 两次独立测量与
ABBA，并将 agent 控制面提交为 `0bd27faba2fd1b896e7ab92467c754813a47f772`。H8F1 编译元数据
显示 combined FC1 为 256 vector / 38 scalar registers、20 B/thread private、32768 B dynamic
shared；这使本轮聚焦 epilogue lifetime、copy/liveness 和 residue predicate，而不重扫 BK/BN/stage。

## 新增第一手来源

原生 Web Search 的精确 query、访问状态、许可证和 hypothesis 映射同时保存在机器可读文件。
以下页面均在 2026-07-20 打开正文；搜索摘要不作为证据。

| ID | 来源 / 许可证 | 外部事实 | H8F1/C500 适用边界 |
|---|---|---|---|
| N33 | [TileLang official repository and annotated GEMM](https://github.com/tile-ai/tilelang)，MIT | 官方 GEMM 示例把 copy、GEMM、epilogue 显式写在 kernel schedule 中。 | 只借鉴可表达的 dataflow；实际 lowering 由本机 MACA fork 与 C500 profiler 判定。 |
| N34 | [TileLang `loop.py`](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/loop.py)，MIT | `Pipelined` 是带 stage/order hint 的语言原语。 | H8F1 已用 stage1；没有 MACA async 证据时不推断 CUDA 异步行为。 |
| N35 | [TileLang MLA implementation notes](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md)，MIT | 官方实现说明 tile/layout 与 backend-specific tuning 共同决定性能。 | 只支持小型因果候选，不把 NVIDIA 数值外推到 C500。 |
| N36 | [MetaX C500 软件栈发布说明](https://developer.metax-tech.com/api/client/document/file/222/preview/?file_type=pdf)，MetaX proprietary documentation | 正文列出 B16 register allocation、SLP、Post-RA/pipeline scheduling，以及 grouped GEMM / fused MoE Triton 优化。 | 可确认编译器确有 register/scheduling 优化类别；不提供 resident-grid、bank map 或 async-copy 常量。 |
| N37 | [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)，NVIDIA documentation terms | 通用原则包括 coalescing、shared-memory conflict 和 occupancy/resource trade-off。 | 只作机制词汇和审查清单，不复制 CUDA warp/bank/occupancy 常量。 |
| N38 | [CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)，BSD-3-Clause | GEMM epilogue 会改变 accumulator lifetime、交换布局与 global-store dataflow。 | 支持隔离 epilogue lifetime 候选；不复制 CUDA-specific epilogue。 |
| N39 | [CUTLASS CuTe predication tutorial](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0y_predication.html)，BSD-3-Clause source | residue-free tile 可避免不必要 predicate，而 edge tile 必须保持保护。 | official workload 的 expert hidden dimensions 是完整 tile；route/token predicate 不可移除。 |
| N40 | [MetaX compiler guide](https://developer.metax-tech.com/api/client/document/file/151/preview/?file_type=pdf)，访问失败 | 页面返回 internal error，未获得正文。 | 不据此启用任何未文档化编译器开关。 |

本地来源 L06 是安装的 TileLang-MACA `ec48829b` 源码与编译产物；M10 是 H8F1 的 256
vector-register、20 B/thread private 元数据和四个 workload-matched profiler；M11 是本轮逐候选
C500 测量。外部事实、工程推断和 C500 实测结论在报告中分开陈述。

## 假设与静态审查

| ID | 机制 | 来源 | 静态风险 / 状态 |
|---|---|---|---|
| H9F1 | 在同一个 `T.Parallel` 中计算并 store FC1 `Up×SiLU(Gate)`，删除随后一次 epilogue traversal | N33, N38, M10 | 运算和 FP32→FP16 store 顺序不变；可能缩短 product lifetime；实测。 |
| H9F2 | H8F1 follow-up：combined FC1 中将 activation copy 放到 Gate/Up copy 之后、GEMM 前 | N34, N36, M10 | 只改 issue/liveness order；因 H8F1 新 spill 条件重审，不是 H8F4 重跑；实测。 |
| H9F3 | 对 residue-free hidden dimension 的 FC1 activation 使用完整 tile copy，移除该维 predicate | N33, N39, M10 | 不移除 token/expert predicate；需观察编译器寄存器/私有内存；实测。 |
| H9F4 | 保留 Gate SiLU fragment writeback，但将 Up×Gate 结果直接 store，避免 product fragment writeback | N38, M10, M11 | 与历史 A1g 的全面 fused epilogue 不同；作为 H9F1 profiler follow-up 实测。 |
| H9F5 | H9F4 与 residue-free epilogue predicate removal 组合 | N39, M11 | 仅在 H9F4 为正但未过门时执行；H9F4 回退，条件不成立，静态关闭。 |
| H9S1 | async/deeper/manual pipeline | N34, N36, L06 | 当前 MACA lowering 没有可验证 async primitive，且 H8F1 已有 spill；排除。 |
| H9S2 | undocumented compiler flags | N36, N40 | N40 无正文，不能启用；排除。 |
| H9S3 | 64 KiB shared epilogue staging | N37, N38, M10 | 会把现有 32 KiB dynamic shared 翻倍且无 bank/residency 证据；排除。 |

所有实际候选均在 agent 隔离 worktree 生成，绑定 campaign、H8F1 parent、source IDs、candidate
commit 和 diff hash。RoutedMoEKernel API、submission ABI、shape、seed `81394`、FP16 input、FP32
accumulate、`atol=rtol=0.01` 均保持不变。门禁为 official Large/Small correctness 0 mismatch，
其后 warmup 10 / timing 100 / outlier none 与独立四-launch mcProfiler；只有 mean 和 median 都至少
提升 0.5% 才进入第二轮和 ABBA。
