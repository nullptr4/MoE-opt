# H4F1 pipeline/pair-layout 后续：真实 MetaX C500 结果

日期：2026-07-20（UTC）

## 结论

H8F1 是可重复的确定性胜者，已经带回 `custom_fusedmoe.py`。它只把 combined Gate/Up FC1
的 K loop 从 serial 改为同一 `s1_stages=1` 的 `T.Pipelined`；RoutedMoEKernel API、submission
ABI、数值路径、tile、memory scope 和 FC2 均不变。新源码 SHA-256 是
`6543398edf4e48c95c896b5a703c1a58f3f49a8a7dec543cb892cb0467aea7a3`。

H8F1 两次独立 correctness+10/100+profiler 均超过 mean/median 0.5% 门槛；相邻
H4F1/H8F1/H8F1/H4F1 ABBA 的 mean/median 提升为 `1.9755%/1.9047%`，且两个独立 B
位置全部过门。H8F2、H8F4 明确回退；H8F3 被安装 TileLang layout checker 硬拒绝。

## 协议与设备

正式 workload 是 official `performance-large + performance-small` combined endpoint；每批 warmup
10、timing 100、outlier `none`，全部 100 个有序样本保存在 agent artifact。正确性使用 official
functional Large/Small、seed `81394`、FP16 input、FP32 accumulate、`atol=rtol=0.01`、
`equal_nan=false`。profile workload 明确是 `performance-large` 连续两次 custom kernel：launch
0/2 是 FC1，1/3 是 FC2；不把一个 profile 冒充另一个 workload。

设备为 MetaX C500 64 GiB，driver `3.8.30`，MXMACA `3.7.1.5`，mxcc
`1.0.0@d9102a1572`，PyTorch `2.8.0+metax3.7.1.3`，TileLang-MACA
`0.1.12+maca.gitec48829b@ec48829b`，mcProfiler `3.8.1.4+575f5a9f6d`。统一 environment
fingerprint 是 `sha256:1c750ce4e2ddb41d6d35fd68aaf1a15a714f289cd5cacfcc14983aafa5138d2a`；
每个 artifact 保存温度、功耗、P-state，时钟未由操作员锁定。

## 新采 H4F1 基线

artifact `baseline-6cc629eb15ef9354d8cdc553.json`：Large/Small 均 0 mismatch，mean / median /
stddev / p90 为 `238.554263 / 238.695934 / 1.109980 / 239.684737 ms`。四个匹配 launch 的
FC1 cycles 是 `126724.58/126527.41 Kcycles`，FC2 是 `52749.00/52135.31 Kcycles`，private
read/write 全 0。本次 baseline 的 Global Read event 为异常小量级，已原样保留但不跨 capture
比较。

## 全部真实候选

正数表示相对同一 H4F1 基线更快。

| 候选 | mean / median / stddev / p90 (ms) | mean / median 提升 | mcProfiler 归因 | 决定 |
|---|---:|---:|---|---|
| H8F1 run 1 | 233.946306 / 234.062716 / 1.116737 / 235.157502 | +1.9316% / +1.9411% | FC1 `121760.41/120283.65`；FC2 `53026.54/52625.17 Kcycles`；FC1 private reads `263424/263424` | 初胜，进入复验 |
| H8F1 run 2 | 233.797839 / 233.762686 / 1.154054 / 235.181927 | +1.9939% / +2.0668% | FC1 `123268.57/121803.84`；FC2 `51523.71/52583.13 Kcycles`；同样 FC1 private reads `263424/263424` | 复验通过 |
| H8F2 route prefetch | 240.442935 / 240.472063 / 0.539256 / 241.128934 | -0.7917% / -0.7441% | private 0；FC2 升至 `55743.82/54180.56 Kcycles`，提前 load 未隐藏延迟 | 拒绝 |
| H8F3 pair-last | 无 timing | 无 | build 失败：`gate_up_logits_view` 的 `(i,j,1)` 与 `(i,j,0)` 不满足 StructuralEqual；correctness/profiler 按顺序门禁不运行 | 硬拒绝 |
| H8F4 FC2 copy order | 241.001615 / 241.238016 / 1.077951 / 242.187420 | -1.0259% / -1.0650% | private 0；FC2 `53626.37/55092.68 Kcycles`，endpoint 与 cycles 均回退 | 拒绝 |

H8F1、H8F2、H8F4 的 Large/Small mismatch 均为 0，最大绝对误差均为
`0.005859375/0.0078125`。完整 provenance：

- H8F1 run 1：experiment `experiment-02efcb62ab729fd3719584c5`，candidate commit
  `45735af1370c1c295b62106b5674ac0a75fddb10`，artifact
  `experiment-23cc182ce9af42079dd4eb55.json`；
- H8F1 run 2：experiment `experiment-6eaf4667d55c59aed550fc5c`，candidate commit
  `482705a05a64c07ad072f6552eb82b6a5e13ba6d`，artifact
  `experiment-99780c4ca0c103e9c6ce5c0d.json`；
- H8F2：experiment `experiment-c319b011f85fcf71c81ca139`，candidate commit
  `7e21502d93ca089a783c8e4e8216fbb7d248f4d0`，artifact
  `experiment-6957d7150e34741315454b60.json`；
- H8F3：experiment `experiment-cdfa2e684cbf643ca7f3e665`，candidate commit
  `69260e088da1838378a93948b380e98dbaa3840e`，artifact
  `experiment-1c772636e6cfa01ce588bb5f.json`；
- H8F4：experiment `experiment-35adb0d0c242192eaad9c10e`，candidate commit
  `9ca9ade0ec47d53d9959b444660d30c00a7142ad`，artifact
  `experiment-f32962d0fb5e61de0253c8ae.json`。

四个源码 diff SHA 分别为 `30e9f459...`、`8f345d07...`、`7b13a01e...`、
`8d091bc3...`；机器可读文件保存完整值和 patch/source hash。全部 agent artifact 同时引用隔离
worktree diff、命令 stdout/stderr、100 ordered samples、环境 fingerprint 与 profiler raw
DB/native/per-kernel JSON。

## ABBA

artifact `abba-eed48f61533f3dcbfd7798eb.json`，顺序为 H4F1/H8F1/H8F1/H4F1：

| 位置 | mean / median / stddev / p90 (ms) |
|---|---:|
| H4F1-r1 | 238.573057 / 238.615431 / 1.093340 / 239.851668 |
| H8F1-r1 | 233.893708 / 234.114820 / 1.024390 / 235.120875 |
| H8F1-r2 | 233.997585 / 234.054787 / 1.196870 / 235.672929 |
| H4F1-r2 | 238.747871 / 238.644609 / 1.024204 / 240.153479 |

H8F1 两个位置的 mean 提升 `1.9973%/1.9538%`，median 提升
`1.8921%/1.9173%`。聚合 H4F1/H8F1 mean 为 `238.660464/233.945647 ms`，median 为
`238.630020/234.084804 ms`，稳定选择 H8F1。

## Profiler 解释与剩余边界

H8F1 不是“无代价”变更：stage-1 pipeline lowering 在两次 capture 中都引入每 FC1 launch
`263424` private reads 和 1 private write。但 dominant FC1 cycles 在两次 capture 中均从 baseline
约 `126.5–126.7M` 降至约 `120.3–123.3M`，两轮 endpoint 与 ABBA 都重复约 2% 收益，因而
机制一致且净收益确定。不能把此结论推广为 stage2 或其他 tile；那些历史方向仍关闭。

pair-last 被安装后端 layout invariant 硬关闭；route prefetch 和 copy order 由 endpoint + FC2
cycles 双重关闭；真正 async copy、resident-grid 和离线 weight repack 没有新的 MACA/ABI 能力
证据。未解决方向只包括未来 TileLang-MACA 新增可验证的 async/layout lowering、可靠 resident
grid/occupancy counters，或新 profiler 因果信号；在能力变化前不得重命名重跑。

## 目标验证

- `CAMP_TEST_BACKEND=metax python -m pytest -q`：`86 passed`；
- `python scripts/check_repository.py`：passed（3359 tracked files）；
- `scripts/verify-maca.sh`：MetaX C500 passed；
- `scripts/run-moe.sh`：official functional 2/2 passed，performance diagnostic 完成；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：6/6 passed；
- Stage-2 decision table 由确定性生成器按新源码 SHA 刷新；`git diff --check` passed。
