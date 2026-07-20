# H4F1 后续：增量第一手调研与 C500 假设

日期：2026-07-20（UTC）

## 范围和基线

本轮只研究上一轮 H4F1 落地后仍未解决、且 fixed ABI 可实现的机制。唯一不可变基线是
目标提交 `61fc0bbdb4a08c0a4ba44af278606ce362b794ce` 中的 H4F1，内核 SHA-256
`2be93d4210d7d3bdda857887c70540c6c4da964946033fa95e58b114dd1fe47e`。它组合 H4 的
shared FC2 A tile 与 H2 的 FP16 route-weight fragment cache；A1b、H4 和所有临时候选只作
历史证据，不作正式对照。

检索使用本会话原生 Web Search，并实际打开正文或源码。访问日期均为 2026-07-20。
搜索摘要不作为证据；CUDA/ROCm 的指令、warp、TMA 和 occupancy 假设不直接外推到 C500。

## 新增第一手来源

| ID | 查询、正文和许可证 | 正文机制 | H4F1/C500 边界 |
|---|---|---|---|
| N06 | `site:tilelang.com documentation T.gemm clear_accum layout shared memory pipeline`；[TileLang data movement instructions](https://tilelang.tile-ai.cn/programming_guides/instructions.html)，MIT | `T.copy` 支持 Fragment/Register 到 Global 的同步 tile copy，并交给后端选择 lowering。 | 支持测试 full-tile epilogue store；不证明 MACA 有 CUDA 异步 copy。 |
| N07 | `repo:tile-ai/tilelang clear_accum`；[TileLang GEMM operation source](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/gemm_op.py)，MIT | `T.gemm` 公共接口提供 `clear_accum`。 | 必须再核对本机 MACA lowering；接口存在不等于生成不同代码。 |
| N08 | `site:github.com/tile-ai/tilelang make_swizzled_layout GEMM`；[TileLang DeepSeek MLA layout notes](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md)，MIT | TileLang layout inference 可自动给 shared GEMM operand 分配 swizzle。 | 是反对无证据手工叠加 padding/swizzle 的来源，不是 C500 bank 数证明。 |
| N09 | `site:github.com/NVIDIA/cutlass residue tile predication full tile epilogue source`；[CUTLASS Efficient GEMM epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)，BSD-3-Clause | 单独 epilogue 阶段做 elementwise scaling，并将 accumulator fragment 映射到 cooperative global store。 | 只借鉴 epilogue locality；不复制 NVIDIA warp/shared 映射。 |
| N10 | `site:docs.nvidia.com/cutlass predication residue tiles GEMM epilogue`；[CUTLASS CuTe predication guide](https://docs.nvidia.com/cutlass/4.3.3/media/docs/cpp/cute/0y_predication.html)，BSD-3-Clause | residue tile 必须 predication，完整 tile 可以走独立路径。 | 支持只特化 `actual_rows == 128`，partial expert tile 保留原始边界保护。 |
| N11 | `site:github.com/triton-lang/triton grouped gemm persistent full tiles mask epilogue source`；[Triton persistent matmul tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py)，MIT | 教程用 epilogue subtiling 缩短 live resource 生命周期。 | 只保留资源生命周期概念；TMA、warp specialization、`NUM_SMS` 是 NVIDIA 特定边界。 |
| N12 | `site:github.com/tile-ai/tilelang pull 1820 fusedmoe coalesced_width`；[TileLang PR 1820](https://github.com/tile-ai/tilelang/pull/1820)，仓库 MIT | 页面正文无法由 Web Search 打开（cache miss）。 | 明确不使用搜索摘要；且上一轮 H4F4 已用 C500 排除显式 `coalesced_width=8`。 |
| N13 | `site:developer.metax-tech.com MACA shared memory bank conflict C500 GPU programming` 与 `site:developer.metax-tech.com mcProfiler C500 shared memory bank conflict occupancy`；[MetaX developer portal](https://developer.metax-tech.com/) | 未找到新的可访问第一手 C500 bank/occupancy 正文。 | 不允许据此编造 bank mapping、resident grid 或 async primitive；只能依赖本机源码和实测。 |

## 本机直接能力证据

`/opt/tilelang-metax/tilelang/maca/op/gemm/gemm_mma.py`（本机 TileLang-MACA revision
`ec48829bb61fcd55a366407c69355581733cceef`，MIT）显示：shared GEMM operands 已自动调用
`make_swizzled_layout`；`clear_accum` 在该后端仅对相同 accumulator 发出与当前显式代码相同的
`T.clear`。因此 H5S1（换一种 clear 写法）和 H5S2（无 bank 证据的手工 swizzle/padding）在静态
审查中关闭，不浪费 C500 正式测量。

## 外部事实、工程推断和实测结论的分界

- 外部事实：TileLang 有同步 `T.copy` 和 `clear_accum` 接口；CUTLASS 明确区分 full/residue
  epilogue；Triton 展示 epilogue resource-lifetime 思路。
- 工程推断：官方 shape 中每个 expert 大多是 128 行完整 tile，只有 residue 需要谓词，因此
  full-tile route load/store specialization、tile store、route fragment representation 值得在原 ABI
  下隔离测试。
- C500 结论：只能来自本轮 correctness、10/100 和独立 mcProfiler artifact；外部资料不计性能
  支持。实际结论见结果报告和 `data/benchmarks/c500-64g/h4f1-followups-20260720.json`。

## 假设和静态风险审查

| ID | 机制 | 来源 | 数值/ABI 风险审查 | 处理 |
|---|---|---|---|---|
| H5F1 | 完整 FC2 tile 同时去掉 route-load 和 scaled-store 行谓词，residue 原样保留 | N09/N10/M03 | 同一 FP16 load、FP32 product、FP16 store 地址；uniform CTA branch | 实测 |
| H5F2 | 完整 FC1 intermediate tile 用 `T.copy` 写 Global，residue 仍用原 predicated store | N06/N10/M04 | value、目标、dtype 不变，只改变 lowering | 实测 |
| H5F3 | H4F1 的 route fragment 改为 FP32，仍从原 FP16 tensor 读一次 | N09/M03 | FP16 到 FP32 是精确提升；需检查 register/private traffic | 实测 |
| H5F4 | 完整 FC2 accumulator 在 fragment 内 FP32 缩放后 `T.copy`，tail 原表达式 | N06/N09/N10/M03 | 仍是一乘一写，不前移 FP16 rounding | 实测 |
| H5F5 | 仅去掉完整 FC2 tile 的 route-load 谓词，store 原样 | N10/M03/M05 | H5F1 profiler 支持的测量分解；关闭组合中唯一未隔离组件 | 实测 follow-up |
| H5S1 | 用 `clear_accum` 替换显式 clear | N07/L01 | 本机 MACA 生成语义相同代码，无独立机制 | 静态排除 |
| H5S2 | 手工 shared padding/swizzle | N08/L01/N13 | 当前 lowering 已自动 swizzle，且没有 C500 bank counter/rule | 静态排除 |
| H5S3 | async copy、persistent/subtile、TMA/warp-specialized 路径 | N06/N11/N13 | 缺少当前 MACA primitive/resident-grid 证据，历史深 pipeline/单 CTA persistent 已失败 | 静态排除 |

M03 是上一轮已接受 H4F1 的硬件 evidence；M04 是 H4F4 copy-width 拒绝；M05 是本轮 H5F1
完整 artifact。它们是 measured follow-up，不冒充外部来源。完整 source 和
hypothesis-to-source 映射保存在 agent 的 `kernels/moe/external_research.json`。
