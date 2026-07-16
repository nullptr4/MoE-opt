# 贡献指南

感谢参与 MoE-opt。仓库同时包含可维护代码、实测报告和上游资料归档，因此每类改动的证据要求不同。

## 开始之前

1. 从最新 `main` 创建功能分支，建议命名为 `feature/<主题>`、`fix/<主题>` 或 `docs/<主题>`。
2. 不要修改 `materials/` 下的归档内容，除非改动目标就是修正归档。
3. 不要提交 Conda/venv、build、TileLang cache、编译产物、凭据或未脱敏的 profiler 数据。
4. 性能优化应先通过正确性测试，再比较性能。

## 改动类型与证据

### Kernel 或 schedule

Pull Request 至少说明：

- workload shape 和测试模式；
- C500 型号/显存配置、MACA、PyTorch、TileLang 版本；
- source commit 和完整 schedule；
- functional/ABI 结果；
- warm-up、iteration、缓存策略和性能结果；
- 相对基线的变化以及已知风险。

不得只提交最快数字而省略失败候选、正确性状态或测试条件。

### autotune、autoheuristic 或 profiler 数据

- 每台机器使用稳定且唯一的 `MOE_HOST_ID`。
- 每条记录必须包含 workload、hardware profile、config、latency 和 correctness。
- 原始大文件使用 Git LFS；元数据和 SHA256 清单使用普通 Git。
- 自动生成的 index 必须能从已提交记录重建。

### 文档和资料

- 标明资料来源、版本或 commit。
- 外部规则与仓库说明冲突时，以赛事官方最新规则为准。
- 避免复制不必要的大段第三方内容。

## 本地检查

所有改动至少运行：

```bash
python scripts/check_repository.py
git diff --check
```

涉及 MoE 实现时，在可用的 MACA C500 上继续运行：

```bash
source scripts/activate-maca.sh
scripts/verify-maca.sh
scripts/run-moe.sh
scripts/test-moe-submission.sh --public-shape
```

若没有 GPU，请在 PR 中明确列出未运行的检查，不要将静态检查描述为性能验证。

## Commit 与 Pull Request

- 一个 commit 解决一个清晰问题，使用简短的祈使句或结果描述。
- PR 标题概括完整改动；正文使用模板列出动机、范围、验证和性能证据。
- 不相关的代码、数据和文档不要混入同一个 PR。
- 合并前应处理 review 意见，并确保自动检查通过。
