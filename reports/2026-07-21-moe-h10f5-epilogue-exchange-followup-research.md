# H10F5 epilogue/exchange 增量调研

日期：2026-07-20—21（UTC）

## 边界与基线

本轮唯一不可变基线是 H10F5：目标证据提交
`4c061a28addd6978b78502b95fd1510c93ad624c`，内核起源提交
`e60f9b3216d008e342d285cd24f8b6ede1142387`，源码 SHA-256
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。schedule 是 combined
Gate/Up，FC1/FC2 `BN128/BK64/stage1`，FC1 stage-1 `T.Pipelined`，FC2 A shared，route
FP16 fragment，以及 FC1 K mainloop 后重用 256×64 weight shared buffer 的两个
affine/interleaved 128×64 FP16 Up 半块。A1b、H4、H8F1 与临时候选都未用作本轮正式对照。

## 新增第一手来源

Web Search 搜索词、访问状态、URL、license 和 hypothesis 映射的完整机器记录在
[`h10f5-epilogue-exchange-followups-20260720.json`](../data/benchmarks/c500-64g/h10f5-epilogue-exchange-followups-20260720.json)。
搜索摘要未被当作证据。

- N47：TileLang 官方
  [instructions](https://tilelang.tile-ai.cn/programming_guides/instructions.html)、
  [language basics](https://tilelang.tile-ai.cn/programming_guides/language_basics.html)、
  [copy API](https://tilelang.tile-ai.cn/autoapi/tilelang/language/copy_op/index.html)、
  [copy source](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/copy_op.py) 和
  [MIT license](https://github.com/tile-ai/tilelang/blob/main/LICENSE)。实际打开正文；只借鉴
  `T.copy`/`T.Parallel` 的 region、coalescing 与 layout 接口，不假定 CUDA async 能力。
- N48：TileLang 主论文
  [arXiv:2504.17577](https://arxiv.org/abs/2504.17577)；论文条款，只使用公开机制描述。
  `T.copy`、`T.Parallel`、layout inference 和 shared-layout swizzle 支持检查 affine layout 和
  epilogue lifetime，但不导入 NVIDIA bank/register 常数。
- N49：CUTLASS 官方
  [vectorized epilogue source](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp)
  和 [BSD-3-Clause license](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt)。只借鉴“shared
  repartition + 有界 output fragment + vector store”的高层机制，未移植 CuTe/copy atom。
- N50：[MetaX developer index](https://developer.metax-tech.com/doc) 及搜索到的官方托管 PDF。
  PDF 明示 confidential/proprietary，因此未摘录、未总结、未用于候选。
- L08：本机安装的 TileLang-MACA `ec48829b`，MIT；核对了 `T.Cast`、`T.copy`、
  `T.Parallel`、`coalesced_width` 和 `loop_layout`，并确认裸 async 描述是 PTX-specific。
- M15/M16：H10F5 已接受 profiler/generated source 与本轮首批 C500 实测，属于
  project-owned measured follow-up，不是外部事实。

## Hypothesis 与静态审查

- H11F1：显式 FP16 cast 后直接从 accumulator 写 SwiGLU；会重现 H10F4 的同 fragment
  双 offset `StructuralEqual` 失败，新 fragment 则重复 A1c/H9，静态排除。
- H11F2：不改算术和 FP16 rounding，将 even/odd shared Up 行改为两个连续半块。
- H11F3：按 64 列半块串流执行 Up shared write→product→global store，缩短 lifetime。
- H11F4：仅给两个既有 predicated output `T.Parallel` 加 `coalesced_width=8`；开始用文档的
  Python int，安装版要求 `IntImmNode` 后只做 `T.int32(8)` 语法修正。
- H11F5：保持 proven interleaved arena，写 FP16 activated Gate，再与 FP32 Up 相乘；数值
  rounding 位置改变，必须由官方 Large/Small tolerance 决定。
- H11S1：dynamic/whole-tile `T.copy` 不能证明 residue-safe，且 full-tile branch 重复 H5F1。
- H11S2：cp.async/TMA/WGMMA 在当前 MACA backend 无可用原语。
- H11S3：没有 resident-CTA counter，H10F5 仍为 32 KiB shared 且 private traffic 高；
  FC1 min-blocks=2 已在 A1f 失败，无新因果依据，不重跑。

外部事实、工程推断和 C500 实测结论分别保存：N47–N50 是外部来源，L08 是
安装能力事实，hypothesis 是工程推断，M15/M16 和结果报告才是 C500 实测结论。
