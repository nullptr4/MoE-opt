# 发布内容说明

本仓库在原有比赛资料归档基础上，补充了服务器上实际验证过的 MoE 优化版本、OJ 提交文件、调优日志和报告。

- `benchmarks/tilelang-moe/custom_fusedmoe.py`：本地 benchmark 优化 kernel；
- `benchmarks/tilelang-moe/submission.py`：固定 OJ ABI 的独立提交文件；
- `benchmarks/tilelang-moe/tune_moe.py`：可复现的候选参数筛选工具；
- `benchmarks/tilelang-moe/test_moe_submission.py`：padded token ABI 对拍工具；
- `reports/`：包含成功与失败故事的完整调优报告；
- `logs/`：官方 functional/performance、OJ 对拍和每个候选的原始日志；
- `MATERIALS.lock`：官方资料、TileLang race 分支和环境版本锁定信息；
- `materials/Intro-ops/`、`benchmarks/tilelang-race-tests/`：补充的训练营、MLA 和 NSA 资料。

没有提交服务器 Conda 环境、TileLang build/cache、递归子模块工作树或 Python 字节码。完整 TileLang 源码按 `docs/REPRODUCIBILITY.md` 放在 `external/tilelang-metax` 后即可复现 benchmark。

