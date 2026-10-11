# 开发约定

在 `code/` 内开发；用 uv 管理 Python 依赖，所有实验输出放入 outputs。
不要将虚拟环境、安装包、缓存、密钥或大模型文件加入 Git。

提交前运行：

```bash
bash scripts/run.sh ruff check .
bash scripts/run.sh pytest
```

涉及仿真或依赖变更时，还需运行 `fair-signal doctor` 和 `fair-signal smoke`。
正式实验注明配置、Git 提交、种子、依赖版本与原始指标文件。
新控制器使用统一的信号规则和评价口径；奖励改动需要消融。
引入第三方代码或路网时，保留来源、版本与许可证，更新 THIRD_PARTY_NOTICES。
README 必须区分已实现功能、计划功能和实测结果。

Git 仓库仅覆盖 `code/`，远程 `origin` 为 https://github.com/Highsun/Fair-Signal.git。
完成本地提交后，将相应分支推送到 GitHub；首次推送新分支时设置 upstream。
推送前检查暂存文件及差异，不提交环境、缓存、安装包、密钥或实验大文件。
如远程分支已有更新，先获取并检查差异，保留双方提交，不默认强制推送。
远程 CI 尚未启用；本地检查仍需执行。
