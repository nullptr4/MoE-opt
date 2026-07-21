# H10F5 wave/resource/epilogue 增量调研

日期：2026-07-21（UTC）

## 唯一基线与边界

本轮唯一正式基线是 H10F5：目标证据提交
`f233965325c087a0c0d66c4943b3b00484a379a9`，内核来源提交
`e60f9b3216d008e342d285cd24f8b6ede1142387`，源码 SHA-256
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。其 schedule 为
combined Gate/Up、FC1/FC2 BN128/BK64/stage1、FC1 `T.Pipelined`、FC2 A shared、route-weight
FP16 fragment，以及把死亡的 FC1 256x64 weight shared allocation 显式复用为两个
affine/interleaved 128x64 Up half。A1b、H4、H4F1、H8F1 都只是历史证据。

审计了 agent 的 81 条 E1–H13S3 ledger、H10F5 接受/ABBA、H11–H13 experiments/profiler
与 measured knowledge；没有更晚的正式胜者，也没有已有 FullCol、FC1 512-thread 或 FC2
dead-shared epilogue 实验。

## 新增第一手来源

| ID | 正文、许可证与访问状态 | 外部事实 | H10F5/C500 适用边界 |
|---|---|---|---|
| N58 | [TileLang MLA notes](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md)、[MIT](https://github.com/tile-ai/tilelang/blob/main/LICENSE)，2026-07-21 打开正文 | 大 accumulator 在 worker 不足时可能 spill；`FullCol` 可按输出列分工。 | Hopper 的 warp-group 数值/性能不外推；只借用 TileLang 支持的机制。 |
| N59 | [TileLang GemmWarpPolicy source](https://github.com/tile-ai/tilelang/blob/main/tilelang/tileop/base.py)，MIT，2026-07-21 打开源码 | `FullRow`、`FullCol`、`Square` 分别做 row、column、balanced partition，GEMM 算术不变。 | 精确 MACA lowering 另行审查，只测试一个有因果依据的 policy。 |
| N60 | [CK hardware guide](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/index.html)、[GEMM guide](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/gemm_optimization.html)，ROCm docs terms，2026-07-21 打开正文 | register pressure 影响 occupancy；tile/worker 在状态量与复用间权衡，小 tile 会重复 global load。 | AMD 常量和结果不是 C500 事实；仅支持小型、定向的 worker/tile 实验。 |
| N61 | [TileLang AMD MLA notes](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/amd/README.md)、MIT，2026-07-21 打开正文 | shared/register placement 与 dimension partition 必须 backend-aware。 | 不照搬 ROCm bank/async 假设；精确 TileLang-MACA 源码和 C500 profiler 优先。 |
| N62 | [TileLang GDN URL](https://github.com/tile-ai/tilelang/blob/main/examples/gdn/example_wy_fast.py)，MIT repo | 原生 Web Search 未返回精确正文，GitHub/raw direct open 为 cache-miss。 | 明确记录不可访问，不把搜索摘要当证据；H14F3 由安装版 L11 与实测 M19 支持。 |

查询包括 `site:github.com/tile-ai/tilelang GemmWarpPolicy FullRow FullCol source`、
`site:github.com/tile-ai/tilelang block_M register spilling threads TileLang`、
`site:rocm.docs.amd.com Composable Kernel blockwise GEMM thread cluster tile size register pressure`
和 `site:github.com/tile-ai/tilelang reduce U2RU instructions`。完整 query/URL/title/license/access
记录在 agent `kernels/moe/external_research.json`。

## 安装版与 C500 实测事实

安装的 TileLang-MACA `ec48829bb61fcd55a366407c69355581733cceef` 使用 64-lane worker：
H10F5 FC1 的 256-thread FullRow 是 4x1，FullCol 是 1x4，512-thread 是 8 workers；MACA
GEMM inferencer 已自动给 shared operands 使用 swizzled layout。安装版 GDN 示例确有
fragment→shared→global staging 用于减少 U2RU。H10F5 匹配 profiler 的 FC1 是 16,464
workgroups、65,856 waves、921,984 private read/write；FC2 没有 private traffic。上述是安装版
事实或本机实测，不是外部 GPU 性能推断。

## hypothesis 与静态风险

- H14F1：仅 FC1 `FullRow→FullCol`；dtype、tile、K reduction、FC2、ABI 均不变。
- H14F2：仅 FC1 `256→512` threads，保留 BM128/N256/BK64 和 FC2 256 threads。
- H14F3：FC2 mainloop 后复用两个死亡的 128x64 shared operands，存放 route-scaled FP16
  output halves；route multiplication 仍为 FP32，仍只做一次 FP16 rounding，不新增 shared。
- H14F4：若首批无胜者，单独把 FC1 BM128 拆成每 metadata tile 两个 BM64 CTA，同时保持
  FC2 BM128；预期减少 accumulator 状态，但会重复 weight load/workgroup。
- H14S1：generic manual swizzle 被静态排除，因为精确 MACA 已自动实现；物理 padding 已由
  H6F2 证明回退，不能改名重跑。

工程推断与外部事实明确分开；接受与否只由结果报告中的真实 C500 correctness、10/100 和
mcProfiler 决定。
