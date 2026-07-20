# MoE 激进优化：Gate/Up 宽 GEMM 与 INT8 准入门

## 结论

按 `METAX_MOE_AGGRESSIVE_OPTIMIZATION_SCHEMES.md` 的优先级完成了 R1，并对 R2 做了
真实 C500 lowering、精度和吞吐准入测试：

- 文档原始 A1（logical `BN64`、physical `BN128`、`BK64`）在官方 Large / Small 上
  合计退化 **20.9701%**，拒绝。
- 占用率修正版 A1b（logical `BN128`、physical `BN256`、`BK64`）五进程复验在
  Large / Small 分别稳定提升 **0.5991% / 0.4619%**，合计 **0.5805%**。它满足
  0.3%–1.0% 的可组合候选区间，但未达到 1% 默认升级门槛，因此保留候选开关，当前默认
  仍是 E3+E7。
- C500 能正确 lower INT8 GEMM，生成 `__builtin_mxc_mma_16x16x16i8`，int32 真值逐元素
  完全一致；但代表性 GEMM 的 INT8/FP16 实测加速仅 **1.1663x**。
- 普通 W8A8 的代表性 FC2 tile 有 **44.3752%** 元素超出官方
  `atol=1e-2, rtol=1e-2`。单路 R2 过不了精度门，而双 residual 至少需要两次 INT8
  GEMM，吞吐又远未达到文档要求的约 2x，因此 R2–R4 不进入 submission 实现。
- R5–R7 依赖前述低精度或 persistent producer-consumer 前置条件；当前 ABI 没有预量化
  权重/scale，C500 跨 CTA semaphore 与 memory-ordering 路径也没有安全证据，故本轮按
  文档的顺序门停止，没有用未验证同步或在线全权重量化冒险。

此外，官方默认入口验证时发现 promoted E3+E7 已完全指定，却仍在每次新建 kernel 时经过
autotuner 包装器。改为直接 JIT 后，同一个 `scripts/run-moe.sh` 单进程从
`271.2803 / 69.7196 ms` 恢复到 `239.3611 / 37.9232 ms`。这是 host launch gap 修复，
不改变本报告 A1/A1b 的 GPU schedule 对照口径。

## R1 实现

`custom_fusedmoe.py` 新增内部 `combine_gate_up` specialization，不改变
`RoutedMoEKernel.__init__`、`__call__` 或 submission ABI：

1. 保留两个独立的 FP16 Gate/Up 公共权重 tensor；每个 K tile 将二者复制到一个
   `2*BN × BK` physical shared tile。
2. 一次 `T.gemm` 写入一个 `BM × 2*BN` FP32 accumulator。
3. 将 Up half 复制到独立 epilogue fragment，再对 Gate half 执行 SiLU 并相乘，最终仍写
   原 FP16 `up_logits`。
4. `tune_moe.py --combine-gate-up` 只允许 manual candidate，规范 E0–E5 experiment 标签
   不能混用这个额外变量。

最初直接在同一个 `T.Parallel` 表达式中读取 fragment 的 `j` 与 `j+BN`，MACA layout
inference 报 structural-layout 不一致。修正版先拆出 Up half，避免在一个 Parallel op 中
混合两种 fragment affine access；两组官方 functional 随后均通过。

可复现命令：

```bash
python benchmarks/tilelang-moe/tune_moe.py \
  --candidate-id A1b-gate-up-wide-bn128 \
  --s1-bn 128 --s1-bk 64 --s1-stages 1 \
  --s2-bn 128 --s2-bk 64 --s2-stages 1 \
  --compact-metadata-grid --combine-gate-up \
  --mode functional --shape all --warmup 0 --iteration 1
```

功能结果：
`data/benchmarks/c500-64g/aggressive-a1b-20260720T052900Z-functional.json`，2/2 PASS。

## 端到端性能

筛选统一使用官方 seed/workload、10 warm-up、100 iterations：

| 方案 | Large (ms) | Small (ms) | 合计相对当前 E3+E7 |
| --- | ---: | ---: | ---: |
| E3+E7 control | 238.7697 | 37.4381 | — |
| A1：logical BN64 / physical BN128 | 290.9851 | 43.1437 | **-20.9701%** |
| A1b：logical BN128 / physical BN256 | 237.1454 | 37.0826 | **+0.7167%** |

A1b 随后按 control/candidate 交替顺序运行五个独立进程，每进程 20 warm-up、100
iterations：

| workload | control median / MAD / P95 (ms) | A1b median / MAD / P95 (ms) | A1b 相对 control |
| --- | ---: | ---: | ---: |
| Large | 238.8625 / 0.0392 / 238.9624 | 237.4315 / 0.0515 / 237.4967 | **+0.5991%** |
| Small | 37.4644 / 0.0132 / 37.4932 | 37.2913 / 0.0174 / 37.3065 | **+0.4619%** |
| 合计 | 276.3269 | 274.7228 | **+0.5805%** |

候选五轮均为正向；Large / Small 的 CV 分别为 0.0699% / 0.1607%。原始进程 JSON：

```text
data/benchmarks/c500-64g/aggressive-a1b-20260720T050500Z-control-run-1..5.json
data/benchmarks/c500-64g/aggressive-a1b-20260720T050500Z-candidate-run-1..5.json
```

机器可读总表：
`data/benchmarks/c500-64g/aggressive-r1-r2-20260720T052642Z-summary.json`。

## mcProfiler

从 `/opt/mcProfiler-ubuntu18.04` 启动 mcProfiler，统一使用：

```text
perf_exec --per-kernel --single-pass
--kernelname kernel_kernel
--kernelnames kernel_kernel kernel_kernel_1
```

`1_kernel_kernel.txt.json` 是 FC1，`2_kernel_kernel_1.txt.json` 是 FC2。新增原始目录：

- `mcProfilerOutput/success/aggressive-a1-wide-bn64-large-20260720T052056Z-targeted/`
- `mcProfilerOutput/success/aggressive-a1b-wide-bn128-large-20260720T051739Z-targeted/`
- `mcProfilerOutput/success/aggressive-a1b-wide-bn128-small-20260720T051921Z-targeted/`

对照沿用相同口径的当前 E3+E7：

- `mcProfilerOutput/success/e3e7-combined-large-20260720T034000Z-targeted/`
- `mcProfilerOutput/success/e3e7-combined-small-20260720T034100Z-targeted/`

| workload / FC1 | workgroups | waves | total cycles (Kcycles) | 相对 control cycles |
| --- | ---: | ---: | ---: | ---: |
| Large control | 16,464 | 65,856 | 127,993.62 | — |
| Large A1 | 32,928 | 131,712 | 187,340.46 | **+46.3670%** |
| Large A1b | 16,464 | 65,856 | 127,420.65 | **-0.4477%** |
| Small control | 4,112 | 16,448 | 16,785.66 | — |
| Small A1b | 4,112 | 16,448 | 16,576.08 | **-1.2486%** |

A1 把 logical N tile 减半，恰好使 FC1 workgroups/waves 翻倍；宽 GEMM 减少了一次输入
装载，却没有抵消两倍 CTA 调度，解释了 20.97% 的端到端退化。A1b 恢复原 workgroup
数量，FC1 cycle 才出现小幅下降，因此端到端上限只有约 0.6%。FC2 未改：A1b 的 Large /
Small FC2 cycle 相对 control 为 `+0.0368% / +0.2959%`，属于单次 profiler 波动。

single-pass 的部分 instruction/global-traffic 是推断值，本次跨 run 出现不合理跳变；它们
不用于晋级结论。只使用可由 tile 数严格预测的 workgroups/waves、同向 FC1 cycles，以及
五进程端到端统计。

## R2 INT8 准入门

### Lowering 与真值

TileLang 通用 INT8 GEMM 在 C500 上生成：

```text
__builtin_mxc_mma_16x16x16i8
```

`128×128×128`、INT8 输入、INT32 accumulation/output 与 CPU INT32 matmul 逐元素完全一致，
`max_abs_diff=0`。TileLang 自带测试把 INT8 输入先转 FP32 作 reference，因 FP32 累加舍入
出现 1 LSB 差异；这不是 C500 INT8 MMA 的数值错误。

### 吞吐

代表性 `M=128, N=128, K=3584` tile，五轮交替 event 测量，每轮 20 warm-up / 100 timed：

| dtype | median (ms) | MAD (ms) |
| --- | ---: | ---: |
| FP16 input / FP32 accum | 0.141824 | 0 |
| INT8 input / INT32 accum | 0.121600 | 0.000256 |

INT8 仅为 FP16 的 **1.1663x**；这还没有计入动态 scale reduction、quantize/pack、dequantize
和 route-weight epilogue。

### W8A8 精度

按官方随机初始化尺度构造代表性 Small FC2 tile（`M128, H3584, E1024`），activation
per-row、weight per-output-channel 对称 INT8：

| 指标 | 结果 |
| --- | ---: |
| max absolute error | 0.207459 |
| mean absolute error | 0.020139 |
| relative L2 | 2.1023% |
| 超出 `atol=rtol=1e-2` 的元素 | **44.3752%** |

这只是 R2 准入筛选，不冒充完整 MoE functional；它已经足以否决普通 W8A8。当前公开调用
还只提供 FP16 `routed_expert_down`，没有 prepacked INT8 weight 或 scale。若在 timed
`custom_kernel` 内每次量化全部专家权重，会把模型加载期工作错误地计入算子，并违背文档
要求的 offline prepack。R3 的两次 residual GEMM 虽可能改善精度，但在 1.1663x 的 INT8
吞吐下，即使忽略全部量化/合并开销也无法击败一次 FP16 GEMM。

## 默认入口兼容修复

exact metadata 新增了内部 JIT specialization，而 autotune config factory 原先不接收
`metadata_m`，正常 `scripts/run-moe.sh` 会在 kernel 前报 unexpected keyword。修复包括：

- config factory 接收并忽略 `metadata_m` / `combine_gate_up`；
- autoheuristic variants 保留调用方的 stage schedule，避免把 promoted FC2 `BK64` 重置为
  历史 `BK128`；
- 完全指定的 promoted/manual schedule 直接走 `_moe_forward_tilelang_routed` JIT，不再
  每次穿过无效 autotuner wrapper。

同一 C500、同一官方脚本、10 warm-up / 100 iterations 的单进程诊断：

| 默认入口 | Large (ms) | Small (ms) | 合计 |
| --- | ---: | ---: | ---: |
| 修复前无效 wrapper | 271.2803 | 69.7196 | 340.9999 |
| direct JIT | 239.3611 | 37.9232 | 277.2842 |
| 改善 | **11.7661%** | **45.6061%** | **18.6849%** |

这组数据用于说明 wrapper 问题，不与五进程 A1/A1b GPU schedule 晋级统计混合。

## 环境与验证

- Device：MetaX C500，64 GiB
- MACA driver `3.8.30`，MACA `3.7.1.5`
- Python `3.12.11`，PyTorch `2.8.0+metax3.7.1.3`
- TileLang `0.1.12+maca.gitec48829b`
- mcProfiler `3.8.1.4`（`575f5a9f6d`）
- 起始 commit：`b60f94e8c779be560e0a8832aa65d25fca0f6f1a`，数据对应本报告的 modified worktree

验证结果：

- A1b 官方 functional Large / Small：2/2 PASS。
- `scripts/run-moe.sh` 默认 functional：2/2 PASS；performance 正常完成。
- `scripts/verify-maca.sh`：PASS，确认 C500 / MACA target。
- `scripts/test-moe-submission.sh --public-shape --fuzz`：6/6 PASS。
- 相关 pytest：36 PASS。
- `python scripts/check_repository.py`：PASS。
- `git diff --check`：PASS。
