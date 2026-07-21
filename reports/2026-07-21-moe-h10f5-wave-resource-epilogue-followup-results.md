# H10F5 wave/resource/epilogue follow-up：真实 MetaX C500 结果

日期：2026-07-21（UTC）

## 结论

本轮没有候选同时达到 mean 和 median 至少 0.5% 的接受门。H14F1 未改变 FC1 private
traffic；H14F2 和 H14F4 虽消除 private traffic，但 wave/grid/read/cycle 代价造成明确回退；
H14F3 的小幅正向结果低于两项门槛且 profiler cycle 不一致。因此没有第二次复验、ABBA 或
内核落地。H10F5 保持唯一不可变胜者，源码 SHA-256 仍为
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。

## 协议、环境和基线

每个实际候选先运行 official functional Large/Small，seed `81394`、FP16 input、FP32
accumulate、`atol=rtol=0.01`；随后运行 official performance Large+Small，warmup 10、timing
100、outlier `none`。全部有序样本保存在 agent experiment artifact。profiler workload 是
performance-large 连续两次，即 FC1/FC2/FC1/FC2 四个 launch。

设备为 MetaX C500 64 GiB；driver `3.8.30`、MXMACA `3.7.1.5`、mxcc
`1.0.0@d9102a1572`、PyTorch `2.8.0+metax3.7.1.3`、TileLang-MACA
`0.1.12+maca.gitec48829b@ec48829b`、mcProfiler `3.8.1.4+575f5a9f6d`。

H14F1–H14F3 使用 fingerprint
`sha256:943165667ab1e8188e1aa37de9e4984f217602356b156fd8f1766d9fb9fa9d04`，匹配 H10F5
基线 `228.634757/228.455039 ms`。为 H14F4 提交受控 patch 后，evaluator 在 hardware 前拒绝
旧 integration commit 的 artifact；重新 prepare 并采集 source-identical H10F5 基线，fingerprint
`sha256:fdfe0b3675a71d6f4dd417e87961a12f6434b6077d4b1f468730af8d4f67f42e`，测得
`229.063196/229.141253 ms`。两种 fingerprint 绝不跨组比较。

## 正式实测

正数表示相对同 commit/fingerprint H10F5 更快；cycles 单位 Kcycles。

| 项目 | mean / median / stddev / p90 (ms) | mean / median 提升 | profiler 归因 | 结论 |
|---|---:|---:|---|---|
| H10F5 B1 | 228.634757 / 228.455039 / 1.115210 / 230.115301 | 对照 | FC1 116604.17/118165.70；FC2 51834.93/54500.53；FC1 private 921984/921984 | H14F1–F3 基线 |
| H14F1 FullCol | 228.904558 / 228.835450 / 1.160574 / 230.351437 | -0.1180% / -0.1665% | FC1 117547.29/119022.20；workgroups/waves 16464/65856，private 仍 921984/921984 | 拒绝 |
| H14F2 FC1 512 threads | 292.718917 / 292.665726 / 1.187241 / 293.946580 | -28.0291% / -28.1065% | private 归零，但 waves 131712，FC1 180649.78/180709.81 | 拒绝 |
| H14F3 FC2 shared epilogue | 228.051084 / 228.122491 / 0.734096 / 228.978553 | +0.2553% / +0.1456% | FC2 52062.97/52036.11；private 0；write bytes 约 1.879 GB，cycle 非一致改善 | 低于门槛 |
| H10F5 B2 | 229.063196 / 229.141253 / 1.184275 / 230.571261 | 对照 | FC1 118382.56/118581.37；FC2 51075.28/50850.19 | H14F4 基线 |
| H14F4 FC1 BM64 | 264.334495 / 264.509058 / 1.408422 / 265.937026 | -15.3981% / -15.4349% | private 归零；workgroups/waves 32928/131712，global read 177020928，FC1 151185.17/150458.95 | 拒绝 |

所有候选 official Large/Small 均为 `0/0` mismatch，Large/Small max abs error 为
`0.005859375/0.0078125`。每个候选都保留完整 10/100 和独立 profiler。mcProfiler 自动选择的
部分 instruction-event scale 在 capture 间不兼容；原始值保留，但只用相容的 cycles、
workgroup/wave、private 和 byte counters 归因，未声称虚假的 read/write instruction 改善。

## follow-up、复验和落地

Measured knowledge 复查表明 route cache 已是 H10F5 组成部分；H5F5/H7F4 的正向仅
0.06%–0.11%，且历史 predicate/vectorization 组合已闭合。H14F3 加这些噪声级机制也没有
可信的 0.5% 双门预期，故没有执行非因果组合。H14S1 由安装版自动 layout 与 H6F2 padding
回退硬排除。

没有候选通过首次双门，故第二次 correctness+10/100+profiler 和 ABBA 均未执行。本轮只落地
收敛证据，未修改 `benchmarks/tilelang-moe/custom_fusedmoe.py`。完整 experiment ID、candidate
commit、diff/patch SHA-256、全部样本和 raw profiler 见机器记录与
`/data/metax-c500-autotune-agent/artifacts/`。

## 未解决方向

只保留需要新能力证据的方向：TileLang-MACA 新版本真正提供新的 async/pipeline/layout lowering，
MetaX 官方 resident-grid/bank-conflict/occupancy counter，或 fixed ABI 新增 workspace/problem
descriptor。已闭合的 FullCol、512-thread、BM64、dead-shared epilogue、manual swizzle 以及
H11–H13 方向不得改名重跑。

## 目标验证

- `source scripts/activate-maca.sh`，`CAMP_TEST_BACKEND=metax python -m pytest -q`：
  `89 passed in 25.07s`；直接使用仓库默认 NVIDIA backend 的诊断运行是 `63 passed, 26
  failed`，失败均因 NVIDIA `libcamp_ops.so` 不存在，不是正式 MetaX 结果；
- `python scripts/check_repository.py`：通过，暂存三份 H14 证据后检查 `3386 tracked files`；
- `scripts/verify-maca.sh`：通过，识别 MetaX C500、driver `3.8.30`、MACA `3.7.1.5`；
- `scripts/run-moe.sh`：official functional Large/Small `2/2` 通过；diagnostic timing
  `198.16566406/31.58330322 ms`，不作正式性能对照；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：submission policy、uneven smoke、
  official public shape、六个 fuzz/terminal-offset case 全部通过；
- JSON 解析、`git diff --check`：通过。
