# H10F5 compiler-pass follow-up：真实 MetaX C500 结果

日期：2026-07-21（UTC）

## 结论

本轮没有候选同时达到 mean 和 median 至少 0.5% 的接受门。H13F1 在生成源码层面是 no-op，
H13F2 明确回退约 2.05%，H13F3 明确回退约 8.46%。因此不允许第二次复验、ABBA 或内核落地；
H10F5 保持唯一不可变胜者，源码 SHA-256 仍为
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。

## 协议与环境

每个实际候选先运行 official functional Large/Small，seed `81394`、FP16 input、FP32
accumulate、`atol=rtol=0.01`；随后运行 official performance Large+Small，warmup 10、timing
100、outlier `none`。全部有序样本在 agent artifact 中。profiler workload 是
performance-large 连续两次，即 FC1/FC2/FC1/FC2 四个 launch。

设备为 MetaX C500 64 GiB；driver `3.8.30`、MXMACA `3.7.1.5`、mxcc
`1.0.0@d9102a1572`、PyTorch `2.8.0+metax3.7.1.3`、TileLang-MACA
`0.1.12+maca.gitec48829b@ec48829b`、mcProfiler `3.8.1.4+575f5a9f6d`。本轮正式数据共享
fingerprint `sha256:51fa4ca1fb5a34b38a0bf9bda7278bf9fc76692609cb0576c7561a31c6733be0`。

## 正式实测

正数表示相对同 commit/fingerprint H10F5 基线更快；cycles 单位 Kcycles。

| 项目 | mean / median / stddev / p90 (ms) | mean / median 提升 | FC1 / FC2 cycles | 结论 |
|---|---:|---:|---|---|
| H10F5 | 228.943995 / 228.959486 / 1.077249 / 230.349930 | 对照 | FC1 118762.11/117957.30；FC2 54446.69/51788.42 | 唯一正式基线 |
| H13F1a StorageRewrite in-place | 229.157896 / 229.259903 / 1.233255 / 230.635620 | -0.0934% / -0.1312% | FC1 118300.95/120868.86；FC2 53683.85/52182.97 | 拒绝 no-op |
| H13F2 non-predicated LDG/STG | 233.638953 / 233.556866 / 1.251732 / 235.438715 | -2.0507% / -2.0079% | FC1 121031.33/122332.51；FC2 51422.97/54413.95 | 拒绝回退 |
| H13F3 eager LetInline | 248.309292 / 248.361471 / 0.577325 / 248.966807 | -8.4585% / -8.4740% | FC1 119081.79/118692.80；FC2 70023.95/70206.82 | 拒绝回退 |

三个候选均 official Large/Small `0/0` mismatch，Large/Small max abs error 分别
`0.005859375/0.0078125`，并各自保留完整 10/100 与独立四 launch mcProfiler。

H13F1a performance-large 生成文件是 33405 bytes、SHA-256
`6f697d6f6d676a89a4bd23ec1dda9d4c842c876828ef15877838cf2bf4469575`，与未开开关的 H10F5
缓存文件逐字节相同，所以开关未作用于该内核。其 profiler 自动选择的 private-read 事件值
`5` 与基线 `921984` 不可比较，不能声称读流量下降。第一次 H13F1 候选生成还因 patch 旧上下文在
build 前失败；修正 patch 后的 H13F1a 才是上表正式实验，失败 worktree 保留在 agent quarantine。

H13F2 生成文件增至 34855 bytes、SHA-256
`3bbdd31fdfd5b13c6a7163467fc81817b99d4cfa3e949918158df5c931ccf6cf`，确实出现 10 个
32-bit、2 个 64-bit、8 个 128-bit global-load helper 和 1 个 64-bit、2 个 128-bit
global-store helper，但没有 256-bit helper；因此既证明机制实现、也以回退关闭“只禁 256-bit”
后续。该 capture 的 memory event scale 与基线不兼容，原始值保留但不跨 capture 归因。

H13F3 生成文件增至 59708 bytes、SHA-256
`5f4e14e9993dca565eb09c1dbf428ac67ca0ccde35c2234bbe3792d49465f176`；FC2 cycles 升至约
70M，且首次编译约 189 秒，说明 eager replay 造成代码膨胀和确定性回退。编译时间不混入 GPU
endpoint timing。H13S1-S3 的硬边界见调研报告。

## 复验、组合与落地

没有候选通过首次 mean+median 0.5% 双门，故第二次完整复验和 ABBA 均未执行。H13F1 是代码生成
no-op，H13F2/H13F3 是独立负向 lowering；不存在外部来源或 profiler 支持的正向分量可组合，
所以没有运行非因果组合。本轮只落地收敛证据，未修改 `custom_fusedmoe.py`。

完整 experiment ID、candidate commit、diff/patch SHA-256、100 个有序样本、raw profiler、
环境状态和生成源码信息在机器记录及 agent artifact 中。agent control commit
`c91fecdbcbb7180ece1855e932197fa0df9db456` 早于基线与所有 H13 硬件数据。

## 目标验证

- `source scripts/activate-maca.sh` 后，以 MetaX C ABI backend 运行 `python -m pytest -q`：
  `86 passed in 25.06s`；
- `python scripts/check_repository.py`：通过，检查 `3382 tracked files`；
- `scripts/verify-maca.sh`：通过，识别 MetaX C500、driver `3.8.30`、MACA `3.7.1.5`；
- `scripts/run-moe.sh`：official functional Large/Small `2/2` 通过；diagnostic timing
  `198.93847656/31.74616455 ms`，不作正式性能对照；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：submission policy、uneven smoke、
  official public shape 和五个 fuzz case 全部通过；
- `git diff --check`：通过。

未解决方向仅保留未来 TileLang-MACA 新版本真正改变 generated code 的 storage/lowering pass，
MetaX 官方 resident-grid/bank/occupancy 资料，或 fixed ABI 新增 workspace/descriptor。已收敛的
H11-H13 与 A1d-A1g 方向不得改名重跑。
