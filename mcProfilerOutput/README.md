# mcProfiler MoE 数据包

本目录保存 MetaX C500 上的 mcProfiler 原始采集产物。2026-07-20 的 E0–E7 数据以
Large / Small 成对归档：

| 方案 | Large 目录 | Small 目录 |
| --- | --- | --- |
| E0 | `e0-large-20260720T002113Z-targeted` | `e0-small-20260720T014808Z-targeted` |
| E1 | `e1-large-20260720T004536Z-targeted` | `e1-small-20260720T015015Z-targeted` |
| E2 | `e2-large-20260720T004718Z-targeted` | `e2-small-20260720T015139Z-targeted` |
| E3 | `e3-large-20260720T002257Z-targeted` | `e3-small-20260720T015303Z-targeted` |
| E4 | `e4-large-20260720T004853Z-targeted` | `e4-small-20260720T015427Z-targeted` |
| E5 | `e5-large-20260720T005042Z-targeted` | `e5-small-20260720T015551Z-targeted` |
| E6 | `e6-empty-tile-large-20260720T023737Z-targeted` | `e6-empty-tile-small-20260720T023901Z-targeted` |
| E7 | `e7-exact-metadata-large-20260720T030900Z-targeted` | `e7-exact-metadata-small-20260720T031100Z-targeted` |
| E3+E7 | `e3e7-combined-large-20260720T034000Z-targeted` | `e3e7-combined-small-20260720T034100Z-targeted` |
| Aggressive A1（logical BN64） | `aggressive-a1-wide-bn64-large-20260720T052056Z-targeted` | 未采集；Large 已出现决定性退化 |
| Aggressive A1b（logical BN128） | `aggressive-a1b-wide-bn128-large-20260720T051739Z-targeted` | `aggressive-a1b-wide-bn128-small-20260720T051921Z-targeted` |

每个定向目录中的 `1_kernel_kernel.txt.json` 对应 FC1，
`2_kernel_kernel_1.txt.json` 对应 FC2。目录还包含 mcProfiler 原始 JSON、文本、
CSV、PDF、HTML 和生成图；分析时不要把 Kcycles 直接换算成端到端毫秒。

E0–E7 的定义、测量口径、性能结论与基准数据入口见
`reports/2026-07-20-moe-e0-e7-summary.md`。
Aggressive A1/A1b、INT8 准入门和停止条件见
`reports/2026-07-20-moe-aggressive-r1-r2.md`。
