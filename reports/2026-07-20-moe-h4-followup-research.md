# H4 后续：增量外部调研与静态可行性审查

日期：2026-07-20（UTC）

本轮只增量研究上一轮 H4 尚未收敛的内存放置、copy lowering 和资源调度问题。
搜索结果摘要不作为证据；下列结论均来自实际打开的官方源码、官方文档/PDF，或上一轮
C500 原始 artifact。外部事实、工程推断和 C500 实测结论严格分开记录。

## 搜索与来源

使用本会话原生 Web Search，随后打开第一手页面。新增查询为：

1. `site:github.com/tile-ai/tilelang coalesced_width T.copy shared memory pipeline GEMM`
2. `site:developer.metax-tech.com shared memory MACA bank conflict vectorization`
3. `site:developer.metax-tech.com TileLang C500 occupancy shared_memeory_occupancy`
4. `site:github.com/tile-ai/tilelang fused moe shared activation fragment T.alloc_shared`

| ID | 第一手来源 | 访问/许可证边界 | 外部事实 | 对 H4/C500 的适用性 |
|---|---|---|---|---|
| N01 | [TileLang `copy_op.py`](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/copy_op.py) | 已打开源码；MIT | `T.copy` 接受按元素计的 `coalesced_width`；通用 `T.async_copy` 文档语义是 CUDA `cp.async` | 本机 TileLang-MACA 有相同 annotation 和 lowering，可安全试显式 FP16 width-8；不能据此假定 CUDA async-copy 可移植 |
| N02 | [TileLang 官方 GEMM 示例](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py) | 已打开源码；MIT | 典型 GEMM 在 `T.Pipelined` 内把 A/B tile 放入 shared 后做 `T.gemm` | 支持隔离测试 FC1 activation fragment→shared，但不预言 C500 性能 |
| N03 | [MetaX GPU Core Architecture and Hardware Based Performance Enhancements](https://developer.metax-tech.com/forum/media/attachments/39/d1/El1ADiOoQlsq7kyPEIFB1u2IHcoF2nGVICxS6Yi14awRCnfdinB6F8j1R3CRlVk9/ying-ran-2026052.pdf) | 已打开 PDF；标注 MetaX Confidential / All Rights Reserved；只引用事实，未复制代码 | C-series wave 为 64 threads，每 AP shared memory 64 KiB；active blocks 受资源限制；推荐合并 global access，并用 padding/swizzle 处理 bank conflict | 给出 H4 tile scope/BK 的容量与 occupancy 上界；精确 resident grid 仍需编译器或 profiler 证据 |
| N04 | [TileLang-MetaX Task6 operator tuning and validation](https://developer.metax-tech.com/forum/media/attachments/c7/ec/p921bRvdtxGdMGVlejqB3MWt5vRdjo5Vd9Mdb21bQ3YTuk9JSnNrARaoU14kszzE/yu-han-tilelang-.pdf) | 已打开 PDF；未声明复用许可证；只引用事实，未复制代码 | 公开 C500 数据给出 104 AP、64 KiB group memory、MACA async-copy barrier 约束、BK32 可行；其中同步 stage-2 为 0.464 ms，async 为 0.494 ms | 支持只测一个有因果依据的 BK32；反证“async/deeper stage 必然更快” |
| N05 | [TileLang AMD FlashMLA architecture notes](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/amd/README.md) | 已打开源码说明；MIT | register/shared 放置依赖架构与 footprint；示例中的 shared swizzle 由 TileLang 自动处理 | 支持隔离 FC1 scope，但不应把 CUDA/AMD 手工 swizzle 假设直接搬到 MACA |

[TileLang LICENSE](https://raw.githubusercontent.com/tile-ai/tilelang/main/LICENSE) 已单独打开核对为
MIT。上一轮两个本地 measured follow-up 也参与推导：M01 为 H2 route-weight cache
artifact `experiment-651372a539eaaa5938a3ae85.json`，M02 为 H4 shared FC2 A-tile
artifact `experiment-68a5259b5a15dd0356e1a28e.json`；它们不是外部来源。

## 假设与静态审查

| ID | 机制 | 来源 | ABI/数值风险审查 | 决定 |
|---|---|---|---|---|
| H4F1 | 在 H4 shared FC2 A tile 上组合 route-weight fragment cache，每 row 只加载一次权重 | M01、M02 | fixed ABI；同一 FP32 权重仍在原 FC2 epilogue 乘入，不重排算术 | 实测 |
| H4F2 | FC1 input activation fragment→shared，同时保留 combined Gate/Up | N02、N03、N05、M02 | FC1 A 16 KiB + combined weights 32 KiB = 48 KiB，低于 64 KiB；FP32 accumulator/epilogue 不变 | 实测 |
| H4F3 | 仅将 H4 FC2 BK 64→32，使每 block shared 32 KiB→16 KiB | N03、N04、M02 | validator 合法；FP32 K 顺序不变；显式代价是 K-loop 翻倍 | 只测 BK32，不重扫网格 |
| H4F4 | H4 FC2 两个 global→shared FP16 copy 显式 `coalesced_width=8` | N01、N03、M02 | 16-byte vector；tile extent 可整除；只影响 copy lowering | 实测 |

## 未执行方向与硬边界

- async copy / 更深 pipeline：本机原语存在，但 N04 的 C500 数据显示同步 stage-2 更快，
  且当前 kernel 的 alias/barrier 结构没有新的重叠证据；不机械实现。
- 手工 padding/swizzle：外部原则成立，但当前安装版 MACA 高层 TileLang 未证明有可安全控制
  的显式 layout/padding API，N05 又说明相关 swizzle 可能自动完成；没有能力证据，不实施。
- H1 persistent 多 CTA/SM：H4 的 32 KiB shared 容量上限允许至多 2 blocks/AP，但缺少可靠
  register/resident-grid 结果；`annotate_min_blocks_per_sm` 曾导致 spill，不能只凭 shared 容量重测。
- FC2 stage2、A1d–A1g、单 CTA/SM H1、单独 H2、H3 均已有明确反证，不重复。
- Split-K workspace、ptr-table descriptor、block-sparse/drop、FP4、TMA/WGMMA 在 fixed ABI、
  精度或当前 TileLang-MACA 能力边界内仍不可用；本轮没有发现新能力来源。

以上是外部事实和工程推断。真实 C500 数值与最终判断见
[`2026-07-20-moe-h4-followup-results.md`](2026-07-20-moe-h4-followup-results.md)。
