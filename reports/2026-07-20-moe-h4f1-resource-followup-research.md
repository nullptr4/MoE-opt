# H4F1 资源后续：增量第一手调研与静态审查

日期：2026-07-20（UTC）

## 范围与证据边界

本轮唯一不可变基线是 H4F1：kernel-origin commit
`61fc0bbdb4a08c0a4ba44af278606ce362b794ce`，测量时 target evidence commit
`d08a78262f8b3f0d8e3d6829d277684a2ec46f02`，源码 SHA-256
`2be93d4210d7d3bdda857887c70540c6c4da964946033fa95e58b114dd1fe47e`。A1b、H4、H1–H5
及临时候选仅作历史/失败证据，不作正式性能对照。

检索使用本会话原生 Web Search，并实际打开正文、源码或 PDF；访问日期均为 2026-07-20。
搜索摘要不作证据。外部 CUDA/ROCm 机制只有在本机 TileLang-MACA `ec48829b` 存在对应 lowering，
且 C500 profiler 有因果依据时才成为候选。

## 新增第一手来源

| ID | 查询、正文、许可证 | 外部事实 | H4F1/C500 边界 |
|---|---|---|---|
| N14 | `site:tilelang.com layout shared memory swizzle bank conflict pipeline GEMM`；[TileLang swizzle API](https://tilelang.com/autoapi/tilelang/layout/swizzle/index.html)，MIT | `make_swizzled_layout` 支持 full/half/quarter bank 以及 `allow_pad`。 | 本机 MACA 已自动给 128×64 FP16 GEMM operand 选择 full-bank；只有物理 stride 改变才是非等价候选。 |
| N15 | `site:github.com/tile-ai/tilelang min_blocks_per_sm launch_bounds`；[TileLang PR #1979](https://github.com/tile-ai/tilelang/pull/1979)，MIT | `T.annotate_min_blocks_per_sm(n)` 映射到第二个 `__launch_bounds__` 参数。 | 只测试 H4F1 改变资源后的 FC2 `n=2`；不能从 CUDA occupancy 公式推断 C500 收益。 |
| N16 | `site:developer.metax-tech.com TileLang MACA async copy shared memory swizzle MetaX C500`；[MetaX Task6 C500 training PDF](https://developer.metax-tech.com/forum/media/attachments/c7/ec/p921bRvdtxGdMGVlejqB3MWt5vRdjo5Vd9Mdb21bQ3YTuk9JSnNrARaoU14kszzE/yu-han-tilelang-.pdf)，MetaX All Rights Reserved | 材料给出 104 AP、64 KiB group memory、每 AP 32 waves；其 GEMM 中 stage2+swizzle 胜过无 swizzle，async/更深 stage 更慢，stage3 BK64 超 64 KiB。 | 只作为 C500 方向与资源上限，不复制代码，不预测本 MoE 时延。 |
| N17 | `site:developer.metax-tech.com C500 shared memory bank conflict swizzle padding occupancy`；[MetaX C-series optimization training](https://developer.metax-tech.com/forum/media/attachments/39/d1/El1ADiOoQlsq7kyPEIFB1u2IHcoF2nGVICxS6Yi14awRCnfdinB6F8j1R3CRlVk9/ying-ran-2026052.pdf)，MetaX All Rights Reserved | 64-thread wave、32 waves/AP、32 blocks/AP、64 KiB shared/AP；register/shared 限制 active blocks，padding/swizzle 可处理 bank conflict。 | 不假设 CUDA bank index；用 mcProfiler 和实际 endpoint 决定。 |
| N18 | `site:github.com/tile-ai/tilelang T.view alloc_fragment epilogue`；[TileLang instructions](https://www.tilelang.com/programming_guides/instructions.html)，MIT | `T.reshape`/`T.view` 是共享底层存储的 no-copy alias。 | 支持 FC1 auxiliary fragment elimination 的编译审查；MACA layout inferencer 是硬门。 |
| N19 | `TileLang GitHub __ldg builtin`；[TileLang builtin.py](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/builtin.py)，MIT | `T.__ldg` 接受 flattened `BufferLoad` 并请求只读 cache；上游文档把非 CUDA 后端描述为普通 load fallback。 | 本机 MACA fork 有显式 `__ldg(&(buffer_ref))` codegen，但后续 pass 是否仍保留裸 BufferLoad 必须真实编译。 |
| N20 | `TileLang latest release persistent pipeline vectorized loading min blocks`；[TileLang releases](https://github.com/tile-ai/tilelang/releases)，MIT | release history 提供 pipeline、vectorized loading 与 launch-bound 功能线索。 | 仅用于增量发现；必须回到安装 revision 源码核对。 |
| N21 | `site:developer.metax-tech.com mcProfiler manual C500 events export`；[MetaX mcProfiler manual](https://developer.metax-tech.com/api/client/document/file/211/preview/?file_type=pdf)，MetaX All Rights Reserved | 说明 native capture/export 工作流。 | 支持保留 DB、per-kernel JSON、event catalog 与 invocation；正式性能仍只用 10/100 event timing。 |
| N22 | `site:developer.metax-tech.com C500 async copy sync copy TileLang`；[C500 forum reproduction](https://developer.metax-tech.com/forum/t/ru-he-shi-yong-yi-bu-kao-bei-yi-da-dao-you-hua-de-xiao-guo-ni/215/)，公开社区帖子 | 帖子中的 C500 reproduction 报告 async 慢于 sync。 | 仅 community 支持；与 N16 一起用于“不盲做 async”，不外推到所有 kernel。 |
| N23 | `site:tilelang.com annotate_l2_hit_ratio TileLang`；[TileLang annotations](https://tilelang.com/autoapi/tilelang/language/annotations/index.html)，MIT | 公共 API 有 L2 hit-ratio、layout 与 min-block 注解。 | 安装树中 L2 persistence transform 仅位于 `src/cuda`，当前 MACA 无 lowering，故静态排除。 |

## 本机源码与生成代码证据

安装 revision 是 `/opt/tilelang-metax@ec48829bb61fcd55a366407c69355581733cceef`（MIT）。审查结果：

- `tilelang/maca/op/gemm/gemm_mma.py` 已对 shared GEMM operands 调用 `make_swizzled_layout`；
  128×64 FP16 的默认和 no-pad selector 都选择 full-bank。
- H4F1 生成的 FC1/FC2 均为 `__launch_bounds__(256,1)`；FC2 动态 shared 是 32 KiB，
  FC1/FC2 都没有 private spill。
- H4F1 每个 kernel 对 `group_padded_offsets` 生成两次标量 load；routing metadata 仍是普通 load。
- `src/maca/codegen/codegen_maca.cc` 有 `T.__ldg` 专用 lowering，但只接受代码生成时仍为裸
  flattened `BufferLoad` 的节点。
- H4F1 FC2 `Pipelined(num_stages=1)` 会生成 prologue/steady-state 次序，不与 `T.serial` 等价。
- `annotate_l2_hit_ratio` 只有 CUDA transform；没有 MACA 实现。

## Hypothesis 与静态风险审查

| ID | 机制 | 来源 | 静态/数值审查 | 处理 |
|---|---|---|---|---|
| H6F1 | H4F1 FC2-only `min_blocks_per_sm=2` | N15/N16/N17/L02/M06 | 只改 launch bound；FC2 32 KiB × 2 正好等于文档 64 KiB 上限；不得出现 private spill | 实测 |
| H6F2 | FC2 shared A/B 物理 stride 64→65，logical GEMM 仍 128×64 | N14/N16/N17/L02/M06 | 地址/值不变，物理 layout 改变；不假设 bank geometry | 实测 |
| H6F3 | 用 no-copy view 切分 combined FC1 accumulator，消除 128×128 FP32 auxiliary fragment | N18/L02/M06 | FP32 SiLU/product 次序不变；layout inferencer 为硬门 | 实际 build |
| H6F4 | 仅把 H4F1 FC2 stage1 改为 stage2 | N14/N16/M06 | H4F1 改变了历史 E4 的 A scope/route cache；只复审一个点，不重扫 grid | 实测 |
| H6F5 | 仅把 H4F1 FC2 stage1 pipeline form 改为 `T.serial` | N16/L03/M07 | 同一 GEMM K 次序和 FP32 accumulate；生成代码确实非等价 | 实测 |
| H6F6 | 对两 kernel 的 immutable int32 routing metadata 使用 `T.__ldg` | N19/L03/M06 | 相同整数值；必须生成目标 `__ldg` 才可继续 | 实际 build |
| H6F7 | 两 kernel 各复用一次 padded offset，删除重复 metadata load | L03/M06 | 相同 immutable int32 值；不改变 cache policy | 实测 |
| H6S1 | async copy | N16/N22 | 新 C500 来源均无正向支持，H4F1 profiler 也无反转依据 | 静态排除 |
| H6S2 | 显式写出与 inferencer 相同的 swizzle | N14/L02 | 机器代码机制等价 | 静态排除 |
| H6S3 | 猜 104-AP resident grid 重试 persistent | N16/N17/M06 | H1 已回退约 22%，且无可靠 resident-CTA metric | 静态排除 |
| H6S4 | `annotate_l2_hit_ratio` | N19/N23/L03 | 当前 MACA 没有 lowering | 静态排除 |

外部事实、工程推断和 C500 结论严格分开：上表来源只能提出/排除候选，性能判断只来自结果报告
引用的真实 C500 experiment、100 个有序样本和 workload-matched mcProfiler。
