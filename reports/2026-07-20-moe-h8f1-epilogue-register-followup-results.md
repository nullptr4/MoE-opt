# H8F1 epilogue/register follow-up：真实 MetaX C500 结果

日期：2026-07-20（UTC）

## 结论

H9F1–H9F4 均通过 official Large/Small correctness，但没有一个同时达到 mean/median 0.5%
接受门。H9F1 endpoint 仅改善 `0.3287%/0.2622%`；H9F2、H9F4 回退；H9F3 的 mean 几乎
不变、median 回退，而且编译器 private allocation 从 20 增至 68 B/thread。因此没有候选进入第二次
完整复验或 ABBA，也没有修改目标内核。H8F1 仍是唯一正式胜者，SHA-256 保持
`6543398edf4e48c95c896b5a703c1a58f3f49a8a7dec543cb892cb0467aea7a3`。

## 协议与环境

两次控制面提交导致 environment fingerprint 按规则变化，因此 H9F4 前重新采集 H8F1 B2，未跨
fingerprint 复用 baseline。所有 timing 是 official performance Large+Small combined endpoint，
warmup 10、timing 100、outlier `none`；agent artifact 嵌入全部 100 个有序样本。正确性使用 official
functional Large/Small、seed `81394`、FP16 input、FP32 accumulate、`atol=rtol=0.01`。

设备为 MetaX C500 64 GiB；driver `3.8.30`、MXMACA `3.7.1.5`、mxcc
`1.0.0@d9102a1572`、PyTorch `2.8.0+metax3.7.1.3`、TileLang-MACA
`0.1.12+maca.gitec48829b@ec48829b`、mcProfiler `3.8.1.4+575f5a9f6d`。profile workload 是
performance-large 连续两次 custom kernel，launch 0/2 为 FC1、1/3 为 FC2；每个 artifact 保留
raw database、native report 和 per-kernel JSON。

## 正式基线与候选

正数表示相对同 fingerprint 的 H8F1 更快；cycles 单位是 Kcycles。

| 项目 | mean / median / stddev / p90 (ms) | mean / median 提升 | FC1 / FC2 cycles | 编译资源与决定 |
|---|---:|---:|---|---|
| H8F1 B1 | 233.718072 / 233.574272 / 1.140546 / 235.273042 | 对照 | FC1 122591.38/121649.26；FC2 51715.82/51357.51 | FC1 256 vreg、38 sreg、20 B/thread private |
| H9F1 | 232.949890 / 232.961924 / 1.047240 / 234.211507 | +0.3287% / +0.2622% | FC1 120330.12/121854.63；FC2 54254.83/53326.40 | 资源仍 256/38/20；低于双门，拒绝 |
| H9F2 | 233.806788 / 233.653631 / 1.180230 / 235.454366 | -0.0380% / -0.0340% | FC1 122221.20/121341.83；FC2 51835.71/53507.91 | 256/38/20；endpoint 回退，拒绝 |
| H9F3 | 233.705623 / 233.730046 / 1.055907 / 235.049111 | +0.0053% / -0.0667% | FC1 122737.03/123239.04；FC2 51878.45/52553.64 | private 20→68 B/thread，拒绝 |
| H8F1 B2 | 233.624686 / 233.699329 / 1.074056 / 234.992107 | 对照 | FC1 122248.57/120330.18；FC2 54552.65/51425.29 | 新 agent commit/fingerprint 后重采 |
| H9F4 | 233.875193 / 234.014721 / 1.182450 / 235.256907 | -0.1072% / -0.1350% | FC1 121971.12/120934.91；FC2 54570.94/52793.08 | 256/38/20；endpoint 回退，拒绝 |

全部四个实际候选 Large/Small mismatch 均为 0。provenance 如下：

- H9F1：experiment `experiment-df76d411b304a11694f67d44`，candidate commit
  `87daa4f93602ee2cccf4586d58bdc1d8ea1f8c8f`，diff SHA-256
  `52df5ab18b6d1e6afaccae9ea24543898daf7ef2c78069793a5af95d544d13d1`，artifact
  `experiment-1acc11170d15601bfa30f91f.json`；
- H9F2：`experiment-ba9dfdce26fb6b69dc985888`，commit
  `dadc36d211e24029ea856ed4d4aeef730ff56f43`，diff
  `247bb4227b93c40da8ff2b56252feaa3d5b65afb6cab07e6698231a078e1aeb7`，artifact
  `experiment-fcd6e8dcbebabffe3df8238e.json`；
- H9F3：`experiment-3712c7a8e727fff489a85024`，commit
  `e36c582334b8aed888adb036287a74f7c5e289da`，diff
  `7c0093e29928681c635d0ea2173ecf8586fd5b31321c880780bbe5fe636af1c2`，artifact
  `experiment-66c420616b31c02d35f6a9cf.json`；
- H9F4：`experiment-9e14ddab140d68a0077d2e6b`，commit
  `ff5c73c34794d4a749b959607399c0360f7f8037`，diff
  `126ce2de15e7ddc72b2da830b617ff77c15a894fb00c43c29bf7154bc37771e3`，artifact
  `experiment-5094659f17acd7b37ef5a18d.json`。

B1/B2 artifacts 分别是 `baseline-d8e89cf7922ec29397198f98.json` 与
`baseline-80fd2bc835977ddb8ca6906d.json`，environment fingerprints 分别为
`sha256:6cae8ad22d3618c65605e8399ab52351d85df9c425e679b9dc1fe6afe18da314` 和
`sha256:7b5a071acfffdfb6666a632676deef436500a38309121f76e13ade9dcb423419`。

## Profiler 归因与边界

H9F1 的 FC1 cycles 方向略正，但 FC2 波动使 endpoint 双指标都低于门槛；不能把它当胜者。H9F3
把 FC1 private allocation 增至 68 B/thread，四个同 capture private event 也显著增加，是完整 tile
copy 未能获益的编译资源硬证据。H9F2/H9F4 没改变静态 private allocation，也没有 endpoint 收益。

mcProfiler 的 private/global event 在不同 single-pass batch 间出现 label/scale 互换；报告原样保存
每次 raw event，但只在同一 capture 内解释，绝不跨不兼容 capture 比较其绝对值。Total Cycles 和
编译 metadata 是稳定可比较信号。H9F4 回退后，条件式 H9F5 不执行；H9S1–H9S3 继续由原语、
文档和资源边界排除。因为没有初筛双门胜者，按协议不存在第二次复验或 ABBA。

本轮收敛后仍未解决的方向仅限：未来 TileLang-MACA 提供可验证 async/copy lowering，MetaX 发布
resident-grid/bank/occupancy 资料，或新 profiler 信号证明新的资源因果关系。能力未变化前不得将本轮
epilogue/copy/predicate 机制改名重跑。

## 目标仓库验证

- `CAMP_TEST_BACKEND=metax python -m pytest -q`：`86 passed`；
- `python scripts/check_repository.py`：passed（3365 tracked files inspected）；
- `scripts/verify-maca.sh`：MetaX C500 passed；
- `scripts/run-moe.sh`：official functional 2/2 passed；performance diagnostic
  `202.54564453/32.92630859 ms`，不作正式对照；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：6/6 passed；
- `git diff --check`：passed。

验证后再次计算的内核 SHA-256 仍为
`6543398edf4e48c95c896b5a703c1a58f3f49a8a7dec543cb892cb0467aea7a3`。
