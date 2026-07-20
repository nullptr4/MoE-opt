# A1b 后续：外部机制调研与 C500 候选筛选

日期：2026-07-20（UTC）

目标：`RoutedMoEKernel`，分支 `feature/c500-a1b-followups`

不可变基线：目标提交 `347f17bf1875698db67e2df2fe2a398b0892ad6c` 中的 A1b；内核 SHA-256 `581a00710cfe04b75a7da2bda8aa9fd273f9abfad9030fcf8c00191bc6fe8a27`

agent 起点：`42e777ff294e742a71e43e6d24e5553a103fd033`

## 证据边界

本文把三类结论严格分开：

- **外部事实**：来自下列官方仓库、官方文档、论文或明确标注的社区材料；CUDA/ROCm
  上的结果不是 C500 性能事实。
- **工程推断**：依据外部机制、当前 A1b 源码、已安装 TileLang-MACA 能力和上一轮
  mcProfiler 计数，提出可证伪的候选。
- **C500 实测结论**：必须由本轮同一环境的官方 Large/Small correctness、warmup 10、
  timing 100、无剔除原始样本及 mcProfiler 共同支持。本文初始提交不预填任何性能结论；
  实验完成后由 campaign 数据和最终报告补充。

机器可读的完整查询、来源和 hypothesis-to-source 映射位于
`data/benchmarks/c500-64g/a1b-external-research-20260720.json`。

## 检索过程

本轮使用原生 Web Search 做了十轮定向检索，共 40 条查询。范围包括 TileLang 最新
GEMM/grouped GEMM/persistent 实现、TensorRT-LLM/CUTLASS grouped MoE、Triton、
FlashInfer、DeepSpeed、Megatron、MegaBlocks、Composable Kernel，以及 MetaX/MACA/C500
公开资料。每个作为证据的页面均打开正文或源码；搜索摘要只用于导航。完整查询逐字保存在
机器可读清单中。

无法访问项也保留：TileLang PR #1923 页面在本次 Web 工具中返回 cache miss，故未将其
PR 描述当成证据；改用 TileLang 官方 release 页面、主仓库与本机安装树核验 ptr-table
grouped GEMM 和 persistent primitive。FlashInfer PR #2944 的独立 PR 页面同样未稳定
返回；只采用官方仓库/发布信息中可核验的 grouped-GEMM/combine-fusion 描述。

## 第一手来源与可借鉴边界

| ID | 来源（2026-07-20 访问） | 关键外部事实 | 许可与借鉴边界 | 对当前 C500/TileLang 的适用性 |
|---|---|---|---|---|
| S01 | [TileLang 官方仓库](https://github.com/tile-ai/tilelang) 与 [GEMM 示例](https://github.com/tile-ai/tilelang/tree/main/examples/gemm) | 官方示例覆盖 GEMM、pipeline、parallel copy、L2 swizzle，并包含 persistent GEMM。 | MIT；可借鉴实现思想。官方已列出的主要验证后端是 NVIDIA/AMD，不能外推 C500 性能。 | 本机 TileLang-MACA `ec48829` 确有通用 `T.Persistent`，可直接做 C500 候选。 |
| S02 | [TileLang persistent GEMM 源码](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm_persistent.py) | 用设备 SM 数限制 grid，并让每个 program 迭代多个 tile。 | MIT；API 形态可借鉴，需以本机 fork 编译结果为准。 | 支持 H1；不依赖 CUDA TMA。 |
| S03 | [TileLang releases](https://github.com/tile-ai/tilelang/releases) | 新版本公开 grouped GEMM ptr-table、grouped compilation 和 shared-memory reuse 控制。 | MIT；当前安装 fork 未必含最新 API，不能直接调用未安装功能。 | 支持 grouped/persistent 方向，但 ptr-table 方案在固定 ABI 下先做静态排除。 |
| S04 | [Triton Group GEMM 教程](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) | device tensor 保存各 group 大小、stride、指针；persistent program 遍历 problem/tile。 | MIT；只借鉴调度机制。 | 当前 ABI 已给 packed expert token 和固定权重数组，不应复制 host/device descriptor ABI。 |
| S05 | [Triton Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | grid 为 `min(NUM_SMS, num_tiles)`，program 以固定步长遍历 tile；subtile epilogue 可降低 shared-memory。 | MIT；TMA、warp specialization、Hopper/Blackwell 专用部分不可移植。 | 通用 persistent tile-loop 可测；TMA/WGMMA 排除。 |
| S06 | [NVIDIA MoE fusion 技术博客](https://developer.nvidia.com/blog/boosting-moe-training-throughput-with-advanced-fusion-kernels/) | 融合 activation/scaling 可删除中间读写；设备端 token/group metadata 避免 CPU 同步；动态调度处理不规则 expert workload。 | 博客事实与 CUDA/CuTe 代码受 NVIDIA 条款约束；不复制代码，只借鉴抽象机制。 | 支持 H1、H2、H3；性能数字和 GB200 行为不可外推 C500。 |
| S07 | [NVIDIA grouped GEMM API 博客](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) | 多个不同尺寸 GEMM 可由一个 grouped launch 执行，MoE 是目标场景。 | NVIDIA API/结果；只作机制证据。 | 当前算子已在单 kernel 内按 expert/batch 调度，不能直接替换为 cuBLAS。 |
| S08 | [CUTLASS 官方仓库](https://github.com/NVIDIA/cutlass) | grouped/persistent scheduler 和 expert-wise problem descriptor 是公开实现方向。 | 主 C++ 为 BSD-3-Clause；`python/CuTeDSL` 为 NVIDIA EULA。不得复制 CuTeDSL 代码。 | 支持 H1；SM90/SM100 MMA、TMA 和 occupancy 常量不可用于 C500。 |
| S09 | [TensorRT-LLM：MoE as Dense GEMM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html) | 极低 token、Blackwell FP4 场景下执行 dense experts 可能更快。 | NVIDIA 文档；机制依赖特定精度/硬件。 | 会改变工作量和精度，且是 SM100/103 场景；H7 静态排除。 |
| S10 | [FlashInfer 官方仓库](https://github.com/flashinfer-ai/flashinfer) | 提供 fused MoE、grouped GEMM、routing，并公开 grouped-GEMM/combine fusion 方向。 | Apache-2.0；可借鉴概念，不能导入 CUDA-only kernel。 | 支持 routing metadata 与 combine/scale 融合假设 H2/H3。 |
| S11 | [FlashInfer occupancy issue](https://github.com/flashinfer-ai/flashinfer/issues/3031) | NVIDIA 后端在 profiling 前查询 tactic occupancy 并过滤零 occupancy tactic。 | GitHub issue，社区/工程证据，不是 C500 官方规则。 | 支持在硬件测量前做 register/shared/occupancy 静态审查。 |
| S12 | [Megatron-LM token dispatcher](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py) | token/expert metadata 留在 device，支持 fused permute/unpermute，避免 D2H sync。 | 仓库主许可为 BSD 类且含混合许可；仅借鉴 metadata reuse。 | 当前 ABI 输入已 packed、无通信，dispatcher 重写不适用；支持 H2 的局部 metadata cache。 |
| S13 | [MegaBlocks 仓库](https://github.com/databricks/megablocks) 与 [论文](https://arxiv.org/abs/2211.15841) | block-sparse dropless MoE 避免 padding/drop 的质量—效率权衡；后续路径使用 grouped GEMM。 | Apache-2.0；论文机制可引用。 | 官方 workload 已提供 padded expert stack，改变 packing/drop 会改变 workload；H6 排除。 |
| S14 | [DeepSpeed MoE 教程](https://www.deepspeed.ai/tutorials/mixture-of-experts/) 与 [NLG MoE 教程](https://www.deepspeed.ai/tutorials/mixture-of-experts-nlg/) | expert parallelism、capacity、token drop/no-drop 是系统级 MoE 机制。 | Apache-2.0 项目；不复制无关代码。 | 当前单卡固定 shapes/seed/输出语义，capacity/drop/通信改造均越过 ABI；排除。 |
| S15 | [CK Tile grouped GEMM 文档](https://rocm.docs.amd.com/projects/composable_kernel/en/docs-7.1.1/doxygen/html/structck__tile_1_1_grouped_gemm_kernel.html) | 公开 `UsePersistentKernel`、`MaxOccupancyGridSize`、device group lookup、K-loop/tail pipeline 选择和 Split-K offset。 | MIT；只借鉴调度抽象。 | 强支持 H1；Split-K 需额外 reduction/atomic，需结合 C500 证据筛选。 |
| S16 | [CK Tile 概念文档](https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/index.html) | tile 编程以协作加载、coalescing 和可组合 pipeline 为核心。 | MIT。 | 支持 H4 的 memory-space/occupancy 实验，但 AMD wave/occupancy 数不能外推。 |
| S17 | [MetaX C500 官方产品页](https://www.metax-tech.com/en/goods/prod.html?cid=107&id=21) | C500 提供 64GB、ECC、高带宽并使用 MXMACA 软件栈。 | MetaX 官方内容；只引用公开规格。 | 确认目标设备/软件栈，不提供可臆测的 CUDA 等价常数。 |
| S18 | [MetaX MXMACA release note PDF](https://developer.metax-tech.com/api/client/document/file/222/preview/?file_type=pdf) | 公开版本加入 Group GEMM API、优化 MoE/vLLM Triton kernel、改进 FP16 MMA/分支、提供 mcProfiler；部分原子能力/指标采集有版本限制。 | MetaX 官方文档；不复制闭源实现。 | grouped/MoE 是栈内受支持方向；Split-K 原子路径风险升高；所有候选必须实际 profiler。 |
| S19 | [MetaX 开发者文档索引](https://developer.metax-tech.com/doc) 与 [快速入门](https://developer.metax-tech.com/api/client/document/preview/459/index.html) | 给出 MXMACA 编程模型与当前文档入口。 | MetaX 官方文档。 | 用于核对 API/版本，不把未公开硬件常数填入 schedule。 |
| S20 | [MetaX 异步拷贝论坛帖](https://developer.metax-tech.com/forum/t/ru-he-shi-yong-yi-bu-kao-bei-yi-da-dao-you-hua-de-xiao-guo-ni/215/) | 一份社区 C500 示例中 async copy 慢于同步版本。 | 社区、未验证，不能当通用性能事实。 | 只支持“必须实测，不能假设 async 更快”；不据此接受/拒绝候选。 |
| S21 | [公开 C500 TileLang 赛题分享 PDF](https://developer.metax-tech.com/forum/media/attachments/c7/ec/p921bRvdtxGdMGVlejqB3MWt5vRdjo5Vd9Mdb21bQ3YTuk9JSnNrARaoU14kszzE/yu-han-tilelang-.pdf) | 建议结合 latency、grid/block、register/shared/occupancy 分析，不照搬 NCU stall 模型；async copy 需测量。 | 论坛附件，标注 community，不代表 MetaX 官方承诺。 | 支持本轮以 mcProfiler + resource footprint 判定 H1/H4。 |

许可证原文核对：TileLang、Triton、CK 为 MIT；CUTLASS 主 C++ 为 BSD-3-Clause、CuTeDSL
另受 NVIDIA EULA；FlashInfer、MegaBlocks、DeepSpeed 为 Apache-2.0；Megatron-LM 含混合
许可。本轮计划只写小型、独立的 TileLang schedule 变更，不复制外部 kernel 代码。

## 从来源到候选

### H1 / P1：原生 persistent CTA 调度

- 外部事实：TileLang、Triton 和 CK Tile 都提供“以有限 resident programs 遍历更多
  GEMM tile”的 persistent 调度。
- 工程推断：A1b 的 FC2 有大量逻辑 workgroups；用本机已存在的 `T.Persistent` 将物理
  grid 限制为设备 SM 数，可能降低 launch/scheduler 开销并在 device 端连续取 tile。
- 实现：先只改 FC2，保持 tile shape、MMA、加载、epilogue、输出和 ABI 不变；根据
  profiler 再决定是否扩展到 FC1 或双阶段。
- 数值风险：低；逻辑 tile 与每 tile 的算术顺序不变。性能风险：中，物理 grid 过小或
  tile 顺序不合适会降低并行度。
- 来源：S01、S02、S04、S05、S06、S08、S15。

### H2 / P2：FC2 route-weight 显式缓存与广播

- 外部事实：NVIDIA/FlashInfer 的 MoE epilogue/combine fusion 和 Megatron metadata reuse
  通过复用 device metadata、融合 scaling 来减少重复访存/同步。
- 工程推断：A1b 当前在 FC2 每个输出 tile 的二维 epilogue 中引用同一行的
  `routed_expert_weights[m+i]`。显式装入每行 local fragment 后广播给 128 个输出列，可能
  降低 global load 指令或改善生成代码；若编译器已做同样优化，profiler 应显示无变化。
- 实现：仅增加每行局部 cache，乘法位置/类型/顺序保持原样。
- 数值风险：低；性能风险：低到中，额外 local 值可能增加寄存器压力。
- 来源：S06、S10、S11、S12。

### H3 / P3：把 route scaling 前移到 FC1 中间 epilogue

- 外部事实：MoE 实现通过 activation/scale/epilogue fusion 删除中间读写或重复 combine
  工作；线性 down projection 允许在数学上移动标量缩放。
- 工程推断：`Down(w * SwiGLU(x)) == w * Down(SwiGLU(x))`。A1b 在每个 FC2 输出 tile
  缩放；前移后可在较少的 FC1 中间 tile 完成缩放并移除 FC2 epilogue 乘法/metadata load。
- 实现：ABI、routing、shape 不变，只移动标量乘法。
- 数值风险：**中**。FP16 中间张量会在缩放后提前舍入，虽数学等价但浮点结果不逐位等价；
  必须先用官方 Large/Small `atol=rtol=1e-2` correctness 决定，不能降低容差。
- 来源：S06、S10。

### H4 / P4：FC2 A tile 改用 shared memory

- 外部事实：TileLang/CK Tile 将协作加载、shared-memory footprint、register/occupancy 作为
  GEMM schedule 的核心权衡；FlashInfer 在 profiling 前先排除 occupancy 不可行 tactic。
- 工程推断：A1b 的 FC2 A tile 当前位于 fragment/register 路径，而 profiler 未见 private
  spill。把 A tile 放到 shared 可能降低 register pressure、改善 load reuse，也可能把
  shared footprint 提高到约 32 KiB/block 并降低 occupancy；这是单一机制、需 profiler 判定。
- 实现：只改变 FC2 A tile 的 memory scope，保持 tile shape/MMA/epilogue 不变。
- 数值风险：低；性能风险：中。
- 来源：S01、S02、S11、S16、S18、S21。

## 静态排除与延后方向

- **Split-K**（S15、S18）：当前 FC2 已有大量 M×N tile；Split-K 需要额外 workspace 或
  atomic reduction，并受 MetaX release note 所述原子版本限制影响。当前 ABI 没有合法
  workspace，故第一批不实现；只有 persistent profiler 证明 K 维并行是主瓶颈时再审议。
- **ptr-table grouped GEMM**（S03、S04、S07、S08）：最新实现以 problem descriptor/pointer
  table 为接口；当前 submission ABI 不允许增加 descriptor，且当前 kernel 已把 expert
  维并入 grid。保留来源，但不改变 ABI。
- **block-sparse、dropless、capacity/token drop**（S13、S14）：会改变官方 padding、routing
  或 workload，违反固定 shape/seed/语义。
- **dense experts/FP4**（S09）：改变工作量和精度，且依赖 Blackwell SM100/103。
- **TMA/WGMMA/warp specialization**（S05、S08）：CUDA/Hopper/Blackwell 专用，当前
  TileLang-MACA 无对应能力。
- **盲目增加 pipeline/async copy**（S20、S21）：没有 C500 普适收益证据；上一轮已有相关
  失败。本轮只在新的 profiler 证据指向 memory-latency 且 resource 可行时提出定向候选。

## 正式实验协议

所有实际候选仅与 A1b 比较，且按如下顺序执行：

1. 静态 API/ABI/source-diff 检查与隔离编译；
2. 官方 functional Large、Small，原 shape/seed，`atol=rtol=1e-2`；
3. 同一环境下 Large+Small、warmup 10、timing 100、无样本剔除，保存全部有序样本；
4. 每个 correctness-passing 候选执行 mcProfiler；工具失败也保存 invocation、日志和 partial
   artifacts；
5. mean 和 median 均至少提升 0.5%、相对标准差稳定且 profiler 与机制一致，才进入第二次
   完整 correctness+10/100+profiler；
6. 两次通过后做相邻 A/B 或 A1b–候选–候选–A1b（ABBA）。任何不可复现、均值/中位数
   分歧、异常方差或 sub-0.5% 结果都拒绝。

本轮不会把外部平台性能数字当作 C500 结论，也不会把一个 profile 代替另一个 workload。
