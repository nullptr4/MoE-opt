# MoE Stage 2：Profiler 决策表

## 结论

Stage 1 的 targeted counter 不支持继续扩大 tile 搜索，也不支持立即进入 FC1 异步流水。
E1 BK64 端到端退化且新增大量 VLS pipeline stall；E2 因 CTA 翻倍显著退化；
E4 的两级流水增加指令；E5 明确 spill。下一步转入 FC1 epilogue 合并以及
full/tail/empty 无效工作消除，默认实现继续保持 E0。

## 统一对照

| 版本 | Large / Small ms | FC1 cycles K | FC2 cycles K | FC1 shared | FC2 shared | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Official Square + double weight buffer | 425.8014 / 62.3782 | — | — | 65536 B | 32768 B | warp_policy_and_fc1_shared_capacity |
| FullRow row16/row16 + double weight buffer | 337.7548 / 50.1424 | — | — | 65536 B | 32768 B | fc1_shared_capacity |
| FullRow row16/row16 + single weight buffer | 244.9299 / 38.6086 | — | — | 32768 B | 32768 B | shared_memory_relief_validated |
| Current FullRow row8/row16 single-buffer control | 243.6845 / 38.6199 | 128635.81 | 92604.21 | 32768 B | 32768 B | control_reference |
| FC1 BK64 | 251.3341 / 39.1100 | 137458.68 | 92604.21 | 16384 B | 32768 B | reduction_loop_overhead |
| FC1 BN64 | 295.4857 / 44.5748 | 187449.16 | 92604.21 | 16384 B | 32768 B | cta_amplification |
| FC2 BK64 stage1 | 241.4574 / 38.0251 | 128635.81 | 90411.86 | 32768 B | 16384 B | memory_traffic_amplification |
| FC2 BK64 stage2 | 246.6621 / 38.6520 | 128635.81 | 96729.56 | 32768 B | 32768 B | pipeline_overhead |
| FC2 BN256 | 266.4236 / 41.5835 | 128635.81 | 119048.79 | 32768 B | 65536 B | register_spill |

## 决策依据

- E0 FC1/FC2 targeted counter 为 128635.81 / 92604.21 Kcycles；这些不是毫秒。
- E1 虽把 FC1 shared 从 32 KiB 降到 16 KiB，但 K 循环翻倍，端到端综合退化。
- E2 的 workgroups 翻倍，指令与显存读取同步增加，occupancy 静态上限改善没有转化成收益。
- E3 FC2 局部 cycles 降低，但全量确认只有 0.8816%，且显存读取增加、L2 命中下降。
- E4 stage2 增加指令和流量；E5 出现 private read/write 与 340 B stack frame。

## 数据边界

官方 Square baseline 与 FullRow 双 buffer 仅有历史端到端日志，没有 target-kernel profiler。
早期 `output20260712094550/94821` 实际采集的是 PyTorch distribution kernel，未纳入本表。
mcProfiler 未直接提供 active blocks/SM、VGPR/SGPR 映射、VALU utilization、barrier stall
以及 empty/tail CTA 数；这些字段保持 unavailable，不使用相似计数代替。

## 下一步

执行 Stage 4：先用回归测试锁定行为，再合并 FC1 epilogue，并实现 full/tail/empty 三路径。
只有新的同口径 profiler 证明 FC1 存在可被覆盖的权重加载等待时，才重新打开异步流水分支。

机器可读数据：[`stage2-decision-table.json`](../data/profiler/c500-32g/stage2-decision-table.json)。
