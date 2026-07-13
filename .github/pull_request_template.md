## 目标与范围

<!-- 说明为什么需要此改动，以及明确不在本 PR 范围内的内容。 -->

## 主要改动

- <!-- 列出对代码、数据、文档或工作流的具体改动。 -->

## 验证

- [ ] `python scripts/check_repository.py`
- [ ] `git diff --check`
- [ ] MACA 环境检查（涉及 GPU 代码时）
- [ ] functional / ABI 测试（涉及算子或提交代码时）
- [ ] performance benchmark（声称性能变化时）
- [ ] 文档与数据格式已同步更新

未运行的检查及原因：

## 性能证据

<!-- 没有性能变化请填写“不适用”。有性能变化时填写如下信息。 -->

- Host ID / GPU 显存：
- MACA / PyTorch / TileLang：
- Workload：
- Baseline：
- Candidate：
- Warm-up / iterations / cache policy：
- Correctness：
- Profiler 或记录路径：

## 风险与回滚

<!-- 说明兼容性、隐藏 shape、资源占用、数据迁移风险以及回滚方式。 -->
