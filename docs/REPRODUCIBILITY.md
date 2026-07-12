# 复现说明

## 固定版本

| 组件 | 来源 | 版本/提交 |
| --- | --- | --- |
| 比赛资料 | `metax-maca/op_optimization` | `c06e7fa12b2f52bf2bc4f600ff701bbbf98c1318` |
| TileLang | `tile-ai/tilelang-metax` | `race` / `ee6db4376484f2f7270183c01fd0d90f794965cb` |
| PyTorch | 赛事镜像预装 | 2.8.0 |
| Python | 赛事镜像预装 | 3.12 |
| MACA | 赛事镜像预装 | 3.7.2.1 |

## 准备外部 TileLang 源码

完整 TileLang 源码和子模块没有复制进本仓库。可在仓库根目录准备固定版本：

```bash
git clone --branch race --recurse-submodules \
  https://github.com/tile-ai/tilelang-metax.git external/tilelang-metax
git -C external/tilelang-metax checkout ee6db4376484f2f7270183c01fd0d90f794965cb
git -C external/tilelang-metax submodule update --init --recursive
```

如果源码位于其他位置：

```bash
export TILELANG_HOME=/path/to/tilelang-metax
```

## 验证顺序

```bash
source scripts/activate-maca.sh
scripts/verify-maca.sh
scripts/run-moe.sh
```

`rebuild-tilelang.sh` 会删除 `${TILELANG_HOME}/build` 并重装 editable package，只应在明确需要重编译时执行。
