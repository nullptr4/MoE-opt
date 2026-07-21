# H10F5 direct-fusion/lowering follow-up：真实 MetaX C500 结果

日期：2026-07-21（UTC）

## 结论

本轮没有候选同时达到 mean 和 median 至少 0.5% 的接受门。因此不允许第二次复验、
ABBA 或内核落地；H10F5 保持唯一不可变胜者，源码 SHA-256 仍为
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。

## 协议与环境

每个候选先运行 official functional Large/Small，seed `81394`、FP16 input、FP32
accumulate、`atol=rtol=0.01`；之后是 official performance Large+Small，warmup 10、timing
100、outlier `none`，全部有序样本保存在 agent artifact。profiler workload 是
performance-large 连续两次，对应 FC1/FC2/FC1/FC2 四个 launch。

设备为 MetaX C500 64 GiB；driver `3.8.30`、MXMACA `3.7.1.5`、mxcc
`1.0.0@d9102a1572`、PyTorch `2.8.0+metax3.7.1.3`、TileLang-MACA
`0.1.12+maca.gitec48829b@ec48829b`、mcProfiler `3.8.1.4+575f5a9f6d`。本轮正式数据共享
fingerprint `sha256:3e74c2c09ca6ffe4b25c0d4303cbfece510253ec9a0fd7c0b529acf7cec2729e`。

## 正式实测

正数表示相对同 commit/fingerprint H10F5 基线更快；cycles 单位 Kcycles。

| 项目 | mean / median / stddev / p90 (ms) | mean / median 提升 | FC1 / FC2 cycles | 结论 |
|---|---:|---:|---|---|
| H10F5 | 229.157745 / 229.197317 / 1.053318 / 230.647648 | 对照 | FC1 118139.33/117943.22；FC2 51529.68/53556.64 | 唯一正式基线 |
| H12F1 direct Gate expression | 229.187323 / 229.307004 / 1.113380 / 230.443975 | -0.0129% / -0.0479% | FC1 118541.56/118972.01 | 拒绝 |
| H12F2 one logical output traversal | 235.047411 / 235.114239 / 1.266994 / 236.573333 | -2.5701% / -2.5816% | FC1 122438.99/123198.00；FC2 50893.12/51647.13 | 拒绝 |
| H12F3 native `T.sigmoid` | 229.516892 / 229.653122 / 1.013415 / 230.803453 | -0.1567% / -0.1989% | FC1 119487.45/117875.92；FC2 53605.11/53254.42 | 拒绝 |

H12F1 的 compatible FC1 private read/write 仍为 `921984/921984`，生成代码仍有 `exp2f`
和两个 `uint4` output traversal，说明源码 state-removal 没有转化为资源机制。H12F2 的一个
logical traversal 被 MACA 后端降成两个 conditional `uint4` store 并改变 thread mapping，
FC1 cycles 明显上升。H12F3 将 manual `exp2f` 路径改为 `expf`，没有产生正向 lowering。
H12F2/H12F3 profiler 自动选择的部分 memory event scale 与基线不兼容；原始值保留，但不跨
capture 做因果比较，归因只用 workload-matched Total Cycles 和生成源码。

三条候选全部 official Large/Small `0/0` mismatch，并各自保留完整 10/100 与独立四 launch
mcProfiler。H12S1–H12S3 的静态硬边界见调研报告。H12F1+H12F3 不运行：前者没有实现资源
变化，后者是明确负向的 `expf` lowering，组合两个 rejected form 没有独立正向机制。

## Provenance

完整 experiment ID、candidate commit、diff SHA-256、environment fingerprint、100 个有序样本、
raw profiler 路径和生成源码 SHA-256 在
[`h10f5-direct-fusion-lowering-followups-20260721.json`](../data/benchmarks/c500-64g/h10f5-direct-fusion-lowering-followups-20260721.json)。
agent control commit `225ff7e513c50fcfeeaf2b0a56dda33a9c9fb249` 早于基线和所有 H12 硬件数据。

ABBA 未执行：没有任何候选通过首次 mean+median 0.5% 双门。本轮只落地收敛证据，未修改
`custom_fusedmoe.py`。

## 目标验证

- `CAMP_TEST_BACKEND=metax CAMP_METAX_BUILD_DIR=/data/MoE-opt/materials/Intro-ops/build-metax
  python -m pytest -q`：clean detached `129bb27` worktree 中 `86 passed in 25.06s`；此前在
  `a6fab680` 未给 build-dir 的诊断运行是 `60 passed, 26 failed`，所有失败均为 detached
  worktree 缺少忽略的 `libcamp_ops.so`，未冒充代码回归；
- `python scripts/check_repository.py`：通过，`3287 tracked files inspected`；
- `scripts/verify-maca.sh`：通过，识别 MetaX C500、driver `3.8.30`、MACA `3.7.1.5`；
- `scripts/run-moe.sh`：official functional Large/Small `2/2` 通过；diagnostic timing
  `197.56380859/31.35567383 ms`，不作正式性能对照；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：submission policy 通过，functional
  public+fuzz `6/6` 通过；
- `git diff --check`：clean detached `129bb27` worktree 通过。

验证使用 agent 仓库内 detached target worktree。H12 测量完成后，独立提交 `129bb27` 将
standalone submission 同步到相同 H10F5 schedule 并保持 `custom_fusedmoe.py` SHA 不变；上述
最终回归覆盖该提交。本轮报告提交仍只暂存三份 H12 证据，不改内核或 submission。

未解决方向仅保留未来出现的新 MetaX resident-grid/bank/occupancy 第一手事实、安装版
TileLang-MACA 新增已验证 async/layout lowering，或 fixed ABI 增加 workspace/problem descriptor。
已收敛 H11/H12 方向不得改名重跑。
