# 环境与安装记录

日期：2026-10-11（Asia/Shanghai）。以下为本机实际安装与检查结果。

## 安装前已存在

- Apple Silicon Mac；系统版本检查为 macOS 27.0.1。
- 用户级独立 uv 0.12.15：`~/.local/bin/uv`，以及 uvx。
- Homebrew 与 Python 3.14.5；uv 共享目录内有 Python 3.13.15。
- Miniconda、Git、Command Line Tools。
- 当前 PATH 中没有 SUMO；XQuartz 常见目录不存在。

这些已有工具不属于本项目卸载范围。不更新已有课程的 Python 环境。

## 本次项目内新增

- Python 3.12.14：`.runtime/python/`。
- 独立虚拟环境：`.venv/`；精确 Python 包版本以 `uv.lock` 为准。
- 安装包、uv/Matplotlib/字节码缓存：`.cache/`。

## 系统安装状态

- SUMO 1.28.0：已下载官方安装包，SHA-256 与官网一致。
- 系统信任服务核验：Eclipse Foundation 签名、Apple 公证通过。
- SUMO 1.28.0：系统安装成功，安装收据 `org.eclipse.sumo.framework`。
  框架位置 `/Library/Frameworks/EclipseSUMO.framework`；运行程序位于
  `Versions/1.28.0/EclipseSUMO/bin/`。
- SUMO_HOME：`/Library/Frameworks/EclipseSUMO.framework/EclipseSUMO/share/sumo`。
- 启动器：`/Applications/SUMO sumo-gui.app`、`SUMO netedit.app`、`SUMO Scenario Wizard.app`。
  对应收据 `org.eclipse.sumo.apps.sumo-gui`、`org.eclipse.sumo.apps.netedit`、
  `org.eclipse.sumo.apps.scenario-wizard`。
- XQuartz 2.8.6：签名与 Apple 公证通过，系统安装成功；收据 `org.xquartz.X11`。
  应用位于 `/Applications/Utilities/XQuartz.app`，运行库位于 `/opt/X11`。
  安装包 SHA-256：`9ac35a505095bfbd3009c3b4772f0c6421e2f79c4210ab908459270d1c447909`。
- 未通过 Homebrew 安装额外公式，也未修改 `.zshrc`。项目脚本仅设置当前进程环境。
- XQuartz 安装器要求注销重新登录。没有自动注销用户；GUI 窗口渲染待会话刷新后检查。

初次在沙箱内验证 SUMO 签名时报告 invalid signature，允许系统信任服务后验证通过。
未跳过签名检查或关闭系统安全机制。
安装器无法从受保护的 Desktop 路径读取安装包，因此使用任务专用系统临时目录中转；
安装完成后删除中转副本，原始安装包仍保存在本项目 `.cache/downloads/`。

## 已锁定的主要 Python 包

| 包 | 实际版本 |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.14.1 |
| Stable-Baselines3 | 2.9.0 |
| SUMO-RL | 1.4.5 |
| Gymnasium | 1.2.3 |
| TraCI / sumolib | 1.28.0 |
| NumPy | 2.5.4 |
| Pandas | 2.3.3 |
| Matplotlib | 3.11.2 |
| TensorBoard | 2.21.0 |
| pytest / Ruff | 9.1.1 / 0.17.0 |

完整依赖与下载校验信息以 uv.lock 为准。

## 验证状态

- `fair-signal doctor`：通过；确认原生 arm64、独立 Python、SUMO 及客户端版本匹配。
- `sumo-gui --version`：通过；仅检查程序可加载，不代表已验证窗口渲染。
- Ruff 静态检查：通过。
- pytest：3 项通过（输出目录边界、符号链接逃逸、误用正式训练步数的防护）。
- `fair-signal smoke --steps 128`：通过，CPU，seed=42；使用 SUMO-RL 的现成单路口。
  21 维状态、4 个动作、19844 个 Q 网络参数；128 决策步对应 640 秒仿真。
  28 次梯度更新、参数变化、观测/奖励有限、保存与重载动作一致；最多同时在网 76 辆车。
  本次 `learn()` 耗时 2.28 秒，约 56.14 决策步/秒。
  首次导入、环境建立、模型保存等不计入该计时；不能将这个短跑当作正式训练吞吐或收敛保证。

原始联通报告为 `outputs/smoke/report.json`，模型为 `outputs/smoke/dqn-smoke.zip`，均不入 Git。
沙箱禁止本机端口绑定时，TraCI 的空闲端口探测可能返回 None；本次检查通过允许 localhost
通信后完成。在正常终端运行无需管理员身份，只需正常的本机网络权限。

已知上游限制：SUMO-RL 1.4.5 的 max_green 参数不执行最大绿灯上限，正式实验需实现公共约束。
未执行正式训练、权重扫描、多种子比较或算法效果评估。

## 磁盘占用快照

准备阶段 `.venv` 约 801 MiB、项目 Python 约 66 MiB、`.cache` 约 1.3 GiB。
缓存包含下载包、构建缓存与字节码，可在保留环境后按清理说明回收。
联通输出约 344 KiB；此快照不包含全局 SUMO 和 XQuartz 的占用。
