# A1b 后续：外部机制 C500 实测与 H4 落地

日期：2026-07-20（UTC）

本报告只把真实 MetaX C500 运行称为“实测”。外部平台机制和性能不外推为 C500
结论；外部来源、许可证边界、逐字查询与 hypothesis-to-source 映射见
[`2026-07-20-moe-external-mechanism-research.md`](2026-07-20-moe-external-mechanism-research.md)
和机器可读的
[`a1b-external-research-20260720.json`](../data/benchmarks/c500-64g/a1b-external-research-20260720.json)。

## 结论

唯一胜者是 H4/P4：把 FC2 的 `up_logits` A tile 从 TileLang fragment/register scope
改为 shared memory。它保持 `RoutedMoEKernel`、submission ABI、官方 shape/seed 和
`atol=rtol=0.01` 不变，只改一行 memory scope。两次独立完整 10/100 均超过 0.5%
mean+median 门槛，mcProfiler 显示 FC2 cycle 和 global-read instruction 同方向下降且无
private spill，ABBA 的两个相邻 challenger run 均稳定领先 A1b。因此将其带回目标分支。

接受后的源码 SHA-256：
`c70e328a8d69ca5278fac4ec2ebf85fac1d099696a7f98c392c66b1ff7be1c78`。

## 不可变基线与环境

- A1b 源码 SHA-256：
  `581a00710cfe04b75a7da2bda8aa9fd273f9abfad9030fcf8c00191bc6fe8a27`；本轮所有
  comparison 都引用该受控副本，没有以 A1d–A1g 或 E3+E7 为对照。
- 设备：MetaX C500 64GB，PCI `0000:35:00.0`；driver `3.8.30`；BIOS `1.33.5.0`。
- 软件：MXMACA `3.7.1.5`；mxcc `1.0.0 (d9102a1572)`；PyTorch
  `2.8.0+metax3.7.1.3`；TileLang-MACA
  `0.1.12+maca.gitec48829b@ec48829`；mcProfiler `3.8.1.4+575f5a9f6d`。
- 环境指纹：
  `sha256:134605998c671ae89eb52a1288ce871da02ec21fc2da634e314af218e09a9fb5`。
- 热/钟状态：每次 timing 后由 `mx-smi` 记录；设备处于 P9，时钟未人为锁定。
- 正式 workload：官方 `performance-large` + `performance-small`；warmup 10；有序
  timing 100；outlier policy `none`。所有有序样本嵌在对应 agent artifact 中。
- correctness：官方 `functional-large` 与 `functional-small`，reference
  `official-ref-fusedmoe-seed-81394-v1`；所有实际候选两 shape 均为 0 mismatch。

新采集的 A1b 基线为 mean `275.0647430419922 ms`、median
`275.03167724609375 ms`、stddev `0.4462178865343444 ms`、p90
`275.5010814666748 ms`。证据：
`/data/metax-c500-autotune-agent/artifacts/p9/baseline-710b813ed767e6760ae81412.json`。

## 正式候选结果

以下提升均为相对同一新采集 A1b；正数表示更快。首轮中的 2/7 quick timing 仅作诊断，
未用于任何接受/拒绝决定。

| 候选 | 来源机制 | mean / median / stddev (ms) | mean / median 提升 | mcProfiler Large 关键证据 | 决定 |
|---|---|---:|---:|---|---|
| H1/P1 FC2 persistent CTA | TileLang/Triton/CUTLASS/CK grouped-persistent scheduler | 335.686980 / 335.681162 / 0.424268 | -22.0393% / -22.0518% | FC2 workgroups `57624→104`、waves `230496→416`，但 cycles `89072.59→146126.07 Kcycles`；private R/W 均 0 | 拒绝：机制生效，但单 CTA/SM 丧失 latency hiding |
| H2/P2 route-weight fragment cache | fused combine/scale 与 device metadata reuse | 273.814953 / 273.754499 / 0.439914 | +0.4544% / +0.4644% | FC2 global-read instructions `92170989→88954532`，cycles `89072.59→88041.41 Kcycles`；无 spill | 拒绝：机制成立但 mean、median 均未到 0.5% |
| H3/P3 route scaling 前移 FC1 | activation/scale/epilogue fusion | 276.627703 / 276.736767 / 0.708730 | -0.5682% / -0.6200% | FC2 cycles 降至 `87840.49 Kcycles`，但 FC1 cycles 升至 `129786.51 Kcycles`，新增 private read/write 各 `197568` | 拒绝：0 mismatch 但 FP16 中间前移造成 private traffic 与端到端回退 |
| H4/P4 FC2 A tile shared（run 1） | cooperative shared-memory movement 与 resource/occupancy tradeoff | 239.856085 / 240.271107 / 1.348266 | +12.8001% / +12.6388% | FC2 cycles `52170.90 Kcycles`、global-read instructions `62679987`，private R/W 均 0 | 进入复验 |
| H4/P4 FC2 A tile shared（run 2） | 同上，独立完整复验 | 239.989835 / 240.115322 / 1.239417 | +12.7515% / +12.6954% | FC2 cycles `54728.38 Kcycles`、global-read instructions `62664123`，private R/W 均 0 | 进入 ABBA |

候选 source commit/diff 与完整证据：

- H1：source commit `db8f8b46eb3dcbe7895b152e0afa9e5444f78171`，diff
  `sha256:782a1234215087280d943280e06ea5e57c3f169eaef2a3d5b0170a0b873e81a3`，
  `artifacts/p9/experiment-1a4561cd9ad1cb23e599c4dd.json`。
- H2：source commit `239ee31b5b6afb34190947f99823eececf33cc54`，diff
  `sha256:cae90a60ca8a92adb99be51fa8524c1cc911314ec857dacf3d87ad639c1bbfb8`，
  `artifacts/p9/experiment-651372a539eaaa5938a3ae85.json`。
- H3：source commit `4a0c4aae588d60971f5d3acf996e0ab5a18c5382`，diff
  `sha256:3cbea6ed9888fb522d8b2064fba36d2159ddd63184ffb84cdd94452d48f89bda`，
  `artifacts/p9/experiment-acc457a6763636a0b7e77605.json`。
- H4 run 1：source commit `2ea3ee05c34364cf8f3f9941c96f22cc209beaf9`；H4 run 2：
  source commit `bb6090af906160f4783e20a7806af8b3b3a34125`；两者 exact diff
  `sha256:fae1e53adacc2a2f4ba91ba8382e0d8e6c0a487f0fcc418e4ccb7c72f7b09776`；
  artifacts 分别为 `artifacts/p9/experiment-68a5259b5a15dd0356e1a28e.json` 与
  `artifacts/p9/experiment-989798e2a12068dad99e7840.json`。

上述 `artifacts/...` 均相对于 `/data/metax-c500-autotune-agent`；其中含 build、Large/Small
correctness、100 个有序样本、命令/stdout/stderr、源码 patch/diff、设备状态，以及每个
候选独立的原始 mcProfiler DB、native output、解析 JSON 和 invocation manifest。四个
profiler 均成功解析四个 workload kernel，没有用一个 profile 代替另一候选。

## H4 ABBA 复核

证据：
`/data/metax-c500-autotune-agent/artifacts/p9/abba-42fb02517c2d75307b9acaec.json`。
严格顺序为 A1b-r1、H4-r1、H4-r2、A1b-r2；每段均 warmup 10、timing 100、无剔除：

| 段 | mean (ms) | median (ms) | stddev (ms) |
|---|---:|---:|---:|
| A1b-r1 | 275.531244 | 275.409147 | 0.740311 |
| H4-r1 | 239.339161 | 239.383038 | 1.348093 |
| H4-r2 | 240.673206 | 241.010693 | 1.177612 |
| A1b-r2 | 275.273288 | 275.154181 | 0.640744 |

A1b aggregate mean/median 为 `275.402266 / 275.281664 ms`，H4 为
`240.006183 / 240.196865 ms`，对应 `+12.8525% / +12.7451%`。两个 H4 相邻段各自
mean 提升 `13.0947%`、`12.6103%`，median 提升 `13.0407%`、`12.4494%`；stable gate
为 true。

## 外部主要方向的收敛边界

- generic persistent CTA 已由 H1 实测排除当前“一 CTA/SM、FC2 logical tile loop”实现；
  它不是 C500 上一概无效，只是当前 fixed ABI/grid 的该具体机制有硬回退证据。
- route metadata cache 已由 H2 证明 compiler/codegen 可减少指令，但端到端收益低于门槛；
  不继续做相邻 cache 参数网格。
- scale/epilogue 前移已由 H3 的 private traffic 和回退排除；不通过放宽 tolerance 挽救。
- shared-memory scope 已由 H4 两次正式 run、两个 profiler 和 ABBA 收敛为胜者。
- Split-K 需要 fixed ABI 不提供的 reduction workspace，atomic 路径还改变 reduction order；
  ptr-table grouped GEMM 需要新增 descriptor ABI；block-sparse/drop/capacity、dense FP4 会改变
  workload/精度；TMA/WGMMA/warp specialization 不存在于当前 TileLang-MACA。这些均有
  外部能力/ABI 静态证据排除，未伪装成 C500 性能实验。

## 验证记录边界

目标仓库第一次直接运行 pytest 时因默认 `TILELANG_HOME` 不存在而 collection failed；指定
本机 `/opt/tilelang-metax` 后，第二次因 Intro-ops conftest 默认选择 NVIDIA backend 而出现
26 个缺少 NVIDIA `libcamp_ops.so` 的失败，另有一个源码哈希表正确检测到 H4 变更。这些是
诊断运行，不是正式通过记录。随后用仓库支持的 `CAMP_TEST_BACKEND=metax`、现有
`build-metax` 并由自带生成器刷新 stage2 source provenance，完整 pytest 为
`86 passed in 24.91s`。最终门禁结果以提交前后的成功命令清单为准。

成功门禁（均从 `/data/MoE-opt`、继承真实 `HOME`，并使用
`TILELANG_HOME=/opt/tilelang-metax` 后执行 `source scripts/activate-maca.sh`）：

- `python -m pytest -q`（`CAMP_TEST_BACKEND=metax`、`CAMP_METAX_BUILD_DIR` 指向仓库内
  `materials/Intro-ops/build-metax`）：`86 passed in 24.91s`；
- `python scripts/check_repository.py`：`3347 tracked files inspected`；
- `scripts/verify-maca.sh`：C500/MXMACA/TileLang target 核验通过；
- `scripts/run-moe.sh`：官方 Large、Small functional 均通过；脚本自带诊断 timing 为
  Large `207.95806641 ms`、Small `33.72411865 ms`，不用于正式接受判断；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：uneven smoke、public small 与
  五个 fuzz/boundary case 全部 PASS；
- `git diff --check`：通过。
