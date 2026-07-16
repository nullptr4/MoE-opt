# MoE 优化阶段总结与仓库状态

- **更新日期**：2026-07-16
- **当前基线标签**：`v1-fullrow-single-buffer-row8-row16`
- **默认实现**：`FullRow + FC1 row8 + FC2 row16 + 单 weight shared buffer`
- **当前结论**：Stage 0–2 已完成，保留 E0；Stage 3 异步流水未满足进入条件，下一执行阶段为 Stage 4。

## 1. 最终性能快照

以下数据均来自 C500-32G 上的官方 Large / Small workload；端到端 latency 与
mcProfiler counter 分开解释，不把 `Total Cycles` 换算成毫秒。

| 版本 | Large (ms) | Small (ms) | 相对官方基线 | 状态 |
|---|---:|---:|---:|---|
| 官方 Square + 双 weight buffer | 425.8014 | 62.3782 | 1.0000× | 历史基线 |
| FullRow row16/row16 + 双 weight buffer | 337.7548 | 50.1424 | — | 已被单 buffer 替代 |
| FullRow row16/row16 + 单 weight buffer | 244.9299 | 38.6086 | — | 验证 shared-memory 降压有效 |
| 当前 E0：FullRow row8/row16 + 单 weight buffer | 243.6845 | 38.6199 | Large 1.7473×；Small 1.6152× | 保留默认 |

相对官方原始基线，当前 E0 的 Large / Small latency 分别下降约
**42.77% / 38.09%**。这些数字是当前实验环境的本地结果，不替代目标 OJ 环境的最终成绩。

## 2. 阶段完成状态

| 阶段 | 状态 | 主要结果 | 后续动作 |
|---|---|---|---|
| Stage 0：固化基线 | 完成 | 创建基线标签；统一三进程、median/MAD/P95、functional/OJ/fuzz 与 profiler 证据规范 | 继续作为所有候选的晋级门禁 |
| Stage 1：FC1/FC2 解耦 | 完成 | BN/BK/stages 解耦；E0–E5 共 12/12 functional、36/36 性能样本完整 | 停止扩大 tile 搜索 |
| Stage 2：Profiler 决策表 | 完成 | 9 个版本、32 个哈希来源；确定 reduction/CTA/流量/pipeline/spill 等瓶颈 | 使用机器可读规则驱动下一阶段 |
| Stage 3：FC1 async pipeline | 暂缓 | E1 BK64 端到端退化 2.8833%，且没有同口径 FC1 wall-clock 收益 | 仅在新 profiler 证明可覆盖加载等待后重开 |
| Stage 4：epilogue 与无效工作 | 下一步 | 合并 FC1 epilogue；区分 full/tail/empty；提前退出无效 CTA | 先补回归测试，再做单变量实现和测量 |
| Stage 5–7 | 条件性待办 | 缓存局部性、高风险数值实验、完整 OJ 治理 | 由 Stage 4 数据决定是否进入 |

## 3. Stage 1 结论

E3（仅 FC2 BK128→64、stage1）是唯一进入交错确认的候选。最终权威确认结果为：

| 配置 | Large median (ms) | Small median (ms) | 综合变化 | 判定 |
|---|---:|---:|---:|---|
| E0 | 243.3780 | 38.5905 | 对照 | 保留 |
| E3 | 241.4574 | 38.0251 | +0.8816% | 低于 1% 晋级线，拒绝 |

E3 的 OJ public ABI 2/2 与 fuzz 4/4 均通过，说明它在功能上可用；拒绝原因仅是收益
没有达到预先固定的统计门槛。默认 submission 和 autotune 数据已恢复并通过哈希守卫。

其他候选的主要失败原因：

- E1 FC1 BK64：K 循环翻倍，综合退化 2.8833%。
- E2 FC1 BN64：CTA 与显存读取放大，综合退化 20.4588%。
- E4 FC2 BK64/stage2：指令、流量和 pipeline 开销增加，综合退化 1.0662%。
- E5 FC2 BN256：出现 340 B stack frame 和大量 private read/write，综合退化 9.1046%。

## 4. Stage 2 证据边界与决策

决策表只使用目标 `kernel_kernel` / `kernel_kernel_1` single-pass 报告，并校验 metadata、
源文件、生成代码和阶段函数哈希。早期误采集的 PyTorch distribution kernel 已排除。

当前可稳定比较的指标包括 Kcycles、指令数、MT/ST register、stack、dynamic shared、
static max warps/PEU、private read/write、缓存命中、内存字节数、shared conflict 和 AP duty。
mcProfiler 没有可靠给出的 active blocks/SM、动态 active warps、VGPR/SGPR 严格映射、
barrier stall、独立 DRAM 读写带宽和 tail/empty CTA 数均显式保持 unavailable。

最终机器决策为：

```text
retained_default = E0
tile_search_status = stopped
enter_fc1_async_pipeline = false
recommended_next_stage = Stage 4
```

## 5. 复现与验证入口

```bash
# 重建并验证 Stage 2 决策表
python scripts/build_moe_stage2_profiler_table.py
python scripts/validate_moe_stage2_profiler.py \
  --table data/profiler/c500-32g/stage2-decision-table.json

# 运行仓库回归测试
python -m unittest discover -s tests -v

# 复现 Stage 1 正式矩阵与边界候选确认
python scripts/run_moe_stage1_experiments.py --host-id c500-32g
python scripts/confirm_moe_stage1_winner.py --host-id c500-32g
```

关键证据：

- [Stage 1 解耦报告](2026-07-15-moe-stage1-decoupling-results.md)
- [Stage 2 Profiler 决策表报告](2026-07-16-moe-stage2-profiler-decision-table.md)
- [机器可读 Stage 2 决策表](../data/profiler/c500-32g/stage2-decision-table.json)
- [MoE 基线实验规范](../docs/MOE_BASELINE_EXPERIMENTS.md)
- [完整调优日志](../logs/moe-tuning.md)

## 6. 下一次实施边界

Stage 4 应保持 E0 为对照，每次只引入一个可回滚变量：

1. 用 127/128/129、exact、skewed routing、empty expert 和极端 route weight 测试锁定行为；
2. 单独合并 FC1 SiLU/乘法/store epilogue，测量 fragment 生命周期与端到端变化；
3. 再实现 full/tail/empty 三路径，确认 full tile 去谓词、empty tile 不进入 GEMM；
4. 通过相同的三进程与 OJ 门禁后，才允许修改默认实现。

晋级规则保持不变：综合中位数至少提升 1%，任一公开 workload 退化不超过 0.5%，
且 functional、OJ ABI、fuzz 全部通过并无新增 spill。
