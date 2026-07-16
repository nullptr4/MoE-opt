# MoE 阶段 1：FC1/FC2 参数解耦与 E0–E5 结果

## 结论

FC1/FC2 的 BN、BK、stages 已解耦，旧参数仍按原来的交叉关系解析，
`RoutedMoEKernel.__call__` 与 OJ `run_kernel` ABI 未变。E0–E5 六组全部编译成功并通过
两个公开 functional shape。

E3（仅 FC2 BK 128→64）是唯一稳定变快的候选，但在最终 E0/E3 交错顺序确认中
综合中位数只改善 **0.8816%**，低于计划规定的 1% 晋级线。因此不修改
默认 schedule；最终独立配置仍为 FC1 BN128/BK128/stage1 与
FC2 BN128/BK128/stage1。E3 作为资源特性更好但未过性能门槛的可复现 preset 保留。

## 参数解耦

新的 compile-time schedule 字段是：

```text
s1_bn, s1_bk, s1_stages
s2_bn, s2_bk, s2_stages
```

兼容解析关系为：

```text
block_dexpert -> s1_bn, s2_bk
block_dhidden -> s1_bk, s2_bn
num_stages -> s1_stages
num_stages_down -> s2_stages
```

显式 `s1_*` / `s2_*` 只覆盖对应阶段。由于当前 N/K load 无 tail mask，解析器同时
要求 tile 16 对齐且整除对应维度。

## E0–E5 正式矩阵

原始产物：
[stage1-20260715T141647Z.json](../data/benchmarks/c500-32g/stage1-20260715T141647Z.json)。
每组先运行 functional，再使用 3 个独立性能进程，warmup=10、iteration=100。

| 实验 | 独立变量 | Large median / MAD (ms) | Small median / MAD (ms) | 综合变化 | 判定 |
|---|---|---:|---:|---:|---|
| E0 | 无 | 243.6845 / 0.0088 | 38.6199 / 0.0174 | 0.0000% | 对照 |
| E1 | FC1 BK64 | 251.3341 / 0.2910 | 39.1100 / 0.2463 | -2.8833% | 拒绝 |
| E2 | FC1 BN64 | 295.4857 / 0.0964 | 44.5748 / 0.0679 | -20.4588% | 拒绝 |
| E3 | FC2 BK64 / stage1 | 241.3234 / 0.0594 | 38.0468 / 0.0344 | +1.0394% | 进入交错确认 |
| E4 | FC2 BK64 / stage2 | 246.6621 / 0.0478 | 38.6520 / 0.0009 | -1.0662% | 拒绝 |
| E5 | FC2 BN256 | 266.4236 / 0.0200 | 41.5835 / 0.0422 | -9.1046% | 拒绝 |

E1 说明 FC1 K 循环翻倍的成本超过资源收益；E2 的 N 方向 CTA 翻倍更明显退化。
E4 验证了“BK64 给 stage2 留出 shared-memory 预算”但 pipeline 开销仍未收回；
E5 的 64 KiB dynamic shared 与更大 accumulator 使 occupancy 方向不利。

## E0/E3 交错顺序确认

固定 E0→E1→…→E5 顺序可能引入 GPU 状态偏差，因此附加执行 3 轮独立进程，
每轮交替 E0/E3 先后顺序。权威产物：
[stage1-confirm-20260715T152651Z.json](../data/benchmarks/c500-32g/stage1-confirm-20260715T152651Z.json)。
该 schema v2 产物明确记录 `status=completed`、`promotion_status=rejected`，内嵌
kernel/benchmark 源码快照，并链接包含两个 Stage1 runner 的 `f0dd3e3` 检查点。
当前 runner 会把自身也写入未来产物的快照。早先的 `stage1-confirm-20260715T150421Z.json`
没有源码快照，现已
标记 `status=superseded`，仅保留为历史原始样本，不作为晋级状态来源。

| 配置 | Large median / MAD / P95 (ms) | Small median / MAD / P95 (ms) |
|---|---:|---:|
| E0 | 243.3780 / 0.1594 / 243.5214 | 38.5905 / 0.0144 / 38.6035 |
| E3 | 241.4574 / 0.0078 / 241.4644 | 38.0251 / 0.0033 / 38.4576 |

E3 的 Large 改善 0.7891%，Small 改善 1.4651%，综合改善 0.8816%。两个规格都
没有退化，但综合收益低于 1%，按预先规则视为不足以晋级。

## Profiler 与静态资源

对 Large shape 的 FC1 `kernel_kernel` 与 FC2 `kernel_kernel_1` 分开采集。每个实验
只重新采集变化的阶段；未变化阶段仅在生成函数 SHA256 完全相同时复用 E0 对照。
这样覆盖 E0–E5 的两个阶段，同时避免重复采集字节级相同的 kernel。原始归档：

- [E0 FC1](../data/profiler/c500-32g/stage1-E0-large-fc1/metadata.json)
- [E0 FC2](../data/profiler/c500-32g/stage1-E0-large-fc2/metadata.json)
- [E1 FC1](../data/profiler/c500-32g/stage1-E1-large-fc1/metadata.json)
- [E2 FC1](../data/profiler/c500-32g/stage1-E2-large-fc1/metadata.json)
- [E3 FC1](../data/profiler/c500-32g/stage1-E3-large-fc1/metadata.json)
- [E3 FC2](../data/profiler/c500-32g/stage1-E3-large-fc2/metadata.json)
- [E4 FC2](../data/profiler/c500-32g/stage1-E4-large-fc2/metadata.json)
- [E5 FC2](../data/profiler/c500-32g/stage1-E5-large-fc2/metadata.json)
- [资源汇总 JSON](../data/profiler/c500-32g/stage1-resource-usage.json)
- [E0–E5 生成内核源码档案](../data/profiler/c500-32g/stage1-generated-code/metadata.json)

生成源码档案同时保留每组 `device_kernel.cu`、`host_kernel.cu`、文件哈希和阶段函数
哈希；即使原始编译缓存被清理，也可复核静态资源与“未变化阶段复用”的依据。

### FC1 对照

| 指标 | E0 BN128/BK128 | E1 BK64 | E2 BN64 |
|---|---:|---:|---:|
| dynamic shared | 32 KiB | 16 KiB | 16 KiB |
| MT / ST registers | 256 / 38 | 208 / 30 | 158 / 30 |
| stack frame | 28 B | 0 B | 0 B |
| static max warps/PEU | 2 | 2 | 3 |
| private read / write | 330,240 / 330,240 | 0 / 0 | 0 / 0 |
| Total Cycles | 128,635.81 K | 137,458.68 K | 187,449.16 K |
| Total Instructions | 2.469 B | 3.144 B | 3.757 B |
| AP MMA duty | 28.79% | 27.11% | 19.68% |
| Global read bytes | 63.754 GB | 64.631 GB | 88.962 GB |

E1/E2 都降低了静态资源压力，E2 甚至把静态 warp 上限从 2 提高到 3；但 E1 的
K 循环翻倍、E2 的 N 方向 CTA 翻倍分别带来更多指令与显存读取，最终时延仍退化。

### FC2 对照

| 指标 | E0 BK128 | E3 BK64/stage1 | E4 BK64/stage2 | E5 BN256 |
|---|---:|---:|---:|---:|
| dynamic shared | 32 KiB | 16 KiB | 32 KiB | 64 KiB |
| MT / ST registers | 194 / 30 | 142 / 30 | 138 / 30 | 256 / 38 |
| stack frame | 0 B | 0 B | 0 B | 340 B |
| static max warps/PEU | 2 | 3 | 3 | 2 |
| private read / write | 0 / 0 | 0 / 0 | 0 / 0 | 130,725,504 / 9,246,720 |
| Total Cycles | 92,604.21 K | 90,411.86 K | 96,729.56 K | 119,048.79 K |
| Total Instructions | 2.013 B | 1.752 B | 2.828 B | 1.958 B |
| AP MMA duty | 19.91% | 20.46% | 19.14% | 15.45% |
| L2 hit rate | 94.13% | 87.70% | 86.06% | 79.35% |
| Global read bytes | 7.358 GB | 17.539 GB | 20.072 GB | 23.529 GB |

E3 确实改善了 FC2 资源压力与局部 cycles，但 K 循环翻倍使 L2 命中下降且读取流量
显著增加。E4 的第二级流水没有回收这部分成本，反而增加指令并将 shared access
efficiency 降至 50%。E5 出现 340 B stack frame 和大量 private read/write，明确触发
计划中的 spill 停止条件。mcProfiler 使用 single-pass，这些计数用于方向判断，正常
benchmark latency 仍是晋级依据。

## 验证与最终决策

- E0–E5：functional 12/12 通过；性能样本 36/36 完整。
- E3 OJ 移植试验：public ABI 2/2、fuzz 4/4 通过；独立证据见
  [stage1-e3-oj-validation-20260716T003958Z.json](../data/benchmarks/c500-32g/stage1-e3-oj-validation-20260716T003958Z.json)，
  候选可由其中记录的补丁从检查点精确重建，源码哈希已在隔离 worktree 再验证。
- 交错确认未过 1% 门槛后，已恢复默认 custom factory 与 OJ submission 的 FC2 BK128。
- 恢复后的最终 E0 submission 再次通过 public ABI 2/2 与 fuzz 4/4，且源码哈希与
  检查点 HEAD 一致。
- FC1 最优候选仍是 E0，因此计划中的 E6（FC1 最优 + FC2 最优）与 E3 完全等价，
  不重复执行同一 schedule。

阶段 1 在此停止扩大 tile 搜索。下一阶段应转向 profiler 决策表与手工异步流水，
而不是将 E3 设为默认。
