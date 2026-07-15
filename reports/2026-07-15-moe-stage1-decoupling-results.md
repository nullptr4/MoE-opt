# MoE 阶段 1：FC1/FC2 参数解耦与 E0–E5 结果

## 结论

FC1/FC2 的 BN、BK、stages 已解耦，旧参数仍按原来的交叉关系解析，
`RoutedMoEKernel.__call__` 与 OJ `run_kernel` ABI 未变。E0–E5 六组全部编译成功并通过
两个公开 functional shape。

E3（仅 FC2 BK 128→64）是唯一稳定变快的候选，但在 E0/E3 交错顺序确认中
综合中位数只改善 **0.7682%**，低于计划规定的 1% 晋级线。因此不修改
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
每轮交替 E0/E3 先后顺序。原始产物：
[stage1-confirm-20260715T150421Z.json](../data/benchmarks/c500-32g/stage1-confirm-20260715T150421Z.json)。

| 配置 | Large median / MAD / P95 (ms) | Small median / MAD / P95 (ms) |
|---|---:|---:|
| E0 | 243.4605 / 0.2107 / 243.6501 | 38.5262 / 0.0276 / 38.5510 |
| E3 | 241.7594 / 0.0292 / 241.7857 | 38.0610 / 0.0146 / 38.1426 |

E3 的 Large 改善 0.6987%，Small 改善 1.2074%，综合改善 0.7682%。两个规格都
没有退化，但综合收益低于 1%，按预先规则视为不足以晋级。

## Profiler 与静态资源

对 Large shape 的 FC1 `kernel_kernel` 与 FC2 `kernel_kernel_1` 分开采集。原始归档：

- [E0 FC1](../data/profiler/c500-32g/stage1-E0-large-fc1/metadata.json)
- [E0 FC2](../data/profiler/c500-32g/stage1-E0-large-fc2/metadata.json)
- [E3 FC1](../data/profiler/c500-32g/stage1-E3-large-fc1/metadata.json)
- [E3 FC2](../data/profiler/c500-32g/stage1-E3-large-fc2/metadata.json)
- [资源汇总 JSON](../data/profiler/c500-32g/stage1-resource-usage.json)

| FC2 指标 | E0 BK128 | E3 BK64 | 观察 |
|---|---:|---:|---|
| dynamic shared | 32 KiB | 16 KiB | 减半 |
| MT registers | 194 | 142 | -26.8% |
| ST registers | 30 | 30 | 不变 |
| static max warps/PEU | 2 | 3 | 上限提高 |
| stack frame | 0 B | 0 B | 无新增 stack |
| private read / write | 0 / 0 | 0 / 0 | 未见 spill |
| Total Cycles | 92,604.21 K | 90,411.86 K | -2.37% |
| Total Instructions | 2.013 B | 1.752 B | -12.99% |
| AP MMA duty | 19.91% | 20.46% | 略升 |
| L2 hit rate | 94.13% | 87.70% | 降低 |
| Global read bytes | 7.358 GB | 17.539 GB | K 循环翻倍的读流量代价 |

FC1 的 dynamic shared、registers、stack 与 static max warps/PEU 在 E0/E3 完全一致，
证明实验只改变 FC2。E3 确实改善了 FC2 资源压力和局部 cycles，但更低
L2 hit rate 与更高全局读流量抵消了大部分收益。mcProfiler 使用 single-pass，
这些计数用于方向判断，正常 benchmark latency 仍是晋级依据。

## 验证与最终决策

- E0–E5：functional 12/12 通过；性能样本 36/36 完整。
- E3 OJ 移植试验：public ABI 2/2、fuzz 4/4 通过，说明算法与 ABI 可行。
- 交错确认未过 1% 门槛后，已恢复默认 custom factory 与 OJ submission 的 FC2 BK128。
- FC1 最优候选仍是 E0，因此计划中的 E6（FC1 最优 + FC2 最优）与 E3 完全等价，
  不重复执行同一 schedule。

阶段 1 在此停止扩大 tile 搜索。下一阶段应转向 profiler 决策表与手工异步流水，
而不是将 E3 设为默认。
