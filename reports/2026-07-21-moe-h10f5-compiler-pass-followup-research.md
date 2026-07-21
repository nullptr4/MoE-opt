# H10F5 compiler-pass 增量调研

日期：2026-07-21（UTC）

## 边界与基线

本轮唯一不可变基线是 H10F5：目标证据提交
`ffb0a484da636274d5ff0e13a85da9f83fa93bfc`，内核起源提交
`e60f9b3216d008e342d285cd24f8b6ede1142387`，源码 SHA-256
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。schedule 是 combined
Gate/Up，FC1/FC2 `BN128/BK64/stage1`，FC1 stage-1 `T.Pipelined`，FC2 A shared，route
FP16 fragment，以及重用 256x64 FC1 weight shared buffer 的两个 affine/interleaved
128x64 FP16 Up 半块。A1b、H4、H8F1 与 H11/H12 临时候选都不是正式对照。

## 新增第一手来源

Web Search 查询、URL、访问状态、license、机制和 hypothesis 映射保存在同名机器记录中；
搜索摘要没有作为证据。

- N55：TileLang 官方 [PR 1089](https://github.com/tile-ai/tilelang/pull/1089) 与
  [MIT license](https://github.com/tile-ai/tilelang/blob/main/LICENSE)。合入的默认关闭开关
  `tl.storage_rewrite_detect_inplace` 允许 StorageRewrite 仅在分析证明安全时重用临时存储；
  upstream test 验证其可改变生成源码。是否作用于 H10F5 只能由本机生成源码与 C500 决定。
- N56：TileLang 官方 [pass config 文档](https://tilelang.com/autoapi/tilelang/transform/pass_config/index.html)、
  [源码](https://github.com/tile-ai/tilelang/blob/main/tilelang/transform/pass_config.py) 与 MIT
  license。默认关闭的 LowerLDGSTG 可把连续 global load/store 降成 32/64/128/256-bit helper；
  通用文档不证明 MACA 支持，因此先审查安装版源码。
- N57：TileLang 官方 [LetStmt 指南](https://tilelang.com/compiler_internals/letstmt_inline.html)、
  [pipeline API](https://tilelang.com/autoapi/tilelang/backend/pass_pipeline/pipeline_utils/index.html)
  与 MIT license。eager LetInline 保守替换纯 scalar/index 绑定并保护 buffer-definition 变量。
- L10：安装版 TileLang-MACA
  `ec48829bb61fcd55a366407c69355581733cceef`，TileLang MIT/TVM Apache-2.0。精确源码证明
  MACA pipeline 运行 LetInline、StorageRewrite 与 LowerLDGSTG；LowerLDGSTG 接受 `maca`，
  MACA codegen 实现条件/非条件 32/64/128/256-bit helper。相同源码也证明 predicated store
  不会把 shared/local-backed value 移出分支，且 MACA pipeline 没有 warp specialization。
- M18：H10F5/H11/H12 的项目自有 C500 实测和生成源码。它把本轮限定为不改 ABI、算术、
  schedule 的 compiler-lowering follow-up，并要求 no-op 也必须拒绝。

精确 pass key 的第一轮 Web Search 无结果，随后直接打开上述官方正文。LowerLDGSTG/MACA
GitHub raw 页面返回 cache-miss/internal error，已记录为不可访问；未用摘要替代，而是读取部署中
完全相同版本 L10 的源码正文。

## Hypothesis 与静态审查

- H13F1：只开启 analysis-proven StorageRewrite in-place detection，数值风险低，生成源码必须
  真正变化才算机制。
- H13F2：只开启 non-predicated LowerLDGSTG；它是全内核 post-vectorization lowering，区别于
  历史上只改 route metadata scalar load 的 H6F6。
- H13F3：只开启 eager conservative LetInline，目标是重复 routing/index/address IR；需同时检查
  生成源码体积、编译开销、C500 cycles。
- H13S1：predicated LDG/STG 被安装版 shared/local-backed value 安全检查阻断，静态排除。
- H13S2：fast math 改变数值契约且公开开关是 CUDA/nvcc 路径，静态排除。
- H13S3：安装版 MACA 无 warp specialization/async pass，既有 TMA/WGMMA 边界未改变。

三个实际候选均由 agent 隔离 worktree 生成，并绑定 campaign、source IDs、H10F5 parent、
candidate commit、diff SHA-256。外部事实、工程推断与真实 C500 结论分别保存在来源、静态审查
与结果报告中。
