# H8F1 shared-resource follow-up：增量调研与假设

日期：2026-07-20（UTC）

## 基线与去重

本轮唯一正式基线是 H8F1，目标提交
`9bff88ea9f0e78c40613111bc32ee027346513aa`，内核 SHA-256
`6543398edf4e48c95c896b5a703c1a58f3f49a8a7dec543cb892cb0467aea7a3`。schedule 为
combined Gate/Up FC1 same-buffer stage-1 `T.Pipelined`，FC2 A shared、route weight FP16
fragment，FC1/FC2 均 BN128/BK64/stage1。A1b、H4、H4F1 以及历史临时候选不作对照。

硬件测量前审计了 agent 的 54 项 tried-direction ledger、measured knowledge、最新报告、完整
experiment/profiler 与 Git 历史。控制面先绑定目标 evidence commit 和 H8F1 hash，再提交为
`7599637b69daf34230b6f56d040abf6d1c243113`；H10F4/H10F5 由新的编译证据分别在
`6b5df074873620b3bb1d288115cd23f92686d5ea` 和
`8ff8a9c95089d463a1b460c95b57639a78c29a03` 注册后才采集硬件数据。每次 control commit
改变后都重采同指纹 H8F1 baseline，没有跨 fingerprint 混用。

## 新增第一手来源

以下页面均由原生 Web Search 在 2026-07-20 实际打开正文；搜索摘要不作证据。精确 query、
访问状态、许可证和 hypothesis 映射保存在机器可读结果中。

| ID | 第一手来源 / 许可证 | 外部事实 | H8F1/C500 边界 |
|---|---|---|---|
| N41 | [TileLang releases](https://github.com/tile-ai/tilelang/releases)，MIT | 发布记录把 shared-memory reuse、alias 保留和 pipeline 作为显式编译机制。 | 只用于能力发现；不能把上游 CUDA 性能外推到 C500。 |
| N42 | [MergeSharedMemoryAllocations source](https://github.com/tile-ai/tilelang/blob/main/src/transform/merge_shared_memory_allocations.cc)，Apache-2.0 file | 该 pass 做 shared-buffer liveness 与 linear-scan arena packing。 | 必须由安装的 MACA fork 和生成 launch 证明实际复用。 |
| N43 | [MetaX-hosted Task 6 TileLang training](https://developer.metax-tech.com/forum/media/attachments/c7/ec/p921bRvdtxGdMGVlejqB3MWt5vRdjo5Vd9Mdb21bQ3YTuk9JSnNrARaoU14kszzE/yu-han-tilelang-.pdf)，MetaX training material | 正文列出 BN/BK/stage、register、dynamic shared、occupancy/grid 等 C500 变量及 64 KiB group-memory 边界。 | 仅支持小型定向实验和资源核对；不推断 CUDA 常量。 |
| N44 | [Triton matmul tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py)，MIT | output N tile、K depth、warps/stages 与 launch/cache 顺序耦合。 | 只测试一个 profiler 指向的 FC2 BN64 点，不复制 Triton 参数表。 |
| N45 | [CUTLASS GEMM API 3.x](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md)，BSD-3-Clause | tile/stage 是资源特化点，mainloop 与 epilogue 可组合。 | 仅借用资源与组合概念，不复制 CUDA stage/occupancy 公式。 |
| N46 | [CUTLASS grouped GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/24_gemm_grouped/gemm_grouped.cu)，BSD-3-Clause | grouped scheduler 可使用 host-precomputed problem list 和 sorting。 | fixed submission ABI 没有 problem list/workspace，因此 H10S1 静态排除。 |

本地 L07 是安装的 TileLang-MACA `ec48829b` 的 `maca/pipeline.py`、pass config 和 shared merge
源码：MACA pipeline 确实调用 lifetime-based merge。M12–M14 是 H8F1 资源、第一批 H10 endpoint、
生成 host launch 与 H10F4 编译错误。它们是 measured follow-up，不是外部事实。

## 假设、数值风险与状态

| ID | 机制 | 来源 | 静态审查 / 状态 |
|---|---|---|---|
| H10F1 | FC2 BN128→64，保持 BK64/stage1/shared A/route fragment 和 FC1 不变 | N43, N44, N45, L07, M12 | FP32 K reduction 不变；可能降低 accumulator/shared，但 FC2 CTA 加倍；实测。 |
| H10F2 | 组合未曾一起测试的 H9F1 same-pass FC1 product store 与 H5F5 full-tile FC2 route predicate removal | N43, N45, M12 | 两个独立低风险正向机制、不同 kernel；组合完整验证。 |
| H10F3 | 用 post-mainloop FP16 shared Up tile 替换 FP32 auxiliary fragment，尝试自动复用死去的 FC1 weight arena | N41, N42, N43, L07, M12 | 有一次 FP16 rounding；必须 0 mismatch，且生成 FC1 shared 不得超过 32 KiB。 |
| H10F4 | 不新增 allocation，显式把 256×64 weight buffer 当两个 128×64 Up 半块 | N42, N43, L07, M13 | 针对 H10F3 未重叠的 measured follow-up；编译器因单个 Parallel 中双 affine offset 拒绝。 |
| H10F5 | 将 H10F4 每个 64-wide 半块拆到独立 affine `T.Parallel` | N42, L07, M13, M14 | 只修正精确编译原因，不改 allocation、schedule 或算术；实测并胜出。 |
| H10S1 | host-sorted grouped problem descriptors | N46, M12 | 缺 fixed ABI pointer table/workspace，静态排除。 |

所有实际候选均由 agent 隔离 worktree 生成并绑定 campaign、source IDs、parent baseline、candidate
commit 与 diff hash。`RoutedMoEKernel` API、submission ABI、shape、seed `81394`、dtype、FP32
accumulate、`atol=rtol=0.01` 均不变。每个可编译候选先过 official Large/Small correctness，再跑
warmup 10 / timing 100 / outlier none 和四 workload-matched mcProfiler。H10F4 build 失败，因此按
协议不伪造 correctness、timing 或 profiler。
