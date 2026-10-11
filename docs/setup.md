# 安装与运行

## 安装分层

复用用户已有 uv。SUMO 和 XQuartz 全局安装；Python 3.12、训练库与开发工具放在本仓库。
安装包下载到 `.cache/downloads/`，不入 Git。

SUMO 官方 macOS 包：
https://sumo.dlr.de/releases/1.28.0/sumo-1.28.0.pkg

官网 SHA-256：
`ff81b37ad302652c32ab286d8308c98aa83466c27a1f25bbaeecbb0ead2d0f8c`。

XQuartz 稳定版：
https://github.com/XQuartz/XQuartz/releases/download/XQuartz-2.8.6/XQuartz-2.8.6.pkg

下载后使用 `pkgutil --check-signature` 检查签名，再使用 macOS Installer 安装。
系统组件安装需要操作系统管理员身份认证；这与是否已授权项目操作是两回事。
不关闭 Gatekeeper，不使用跳过签名检查的参数。

SUMO 1.28.0 安装包的框架安装位置为
`/Library/Frameworks/EclipseSUMO.framework`。运行目录由安装后的文件结构检查确定，
以 `docs/environment.md` 记录及 `fair-signal doctor` 输出为准。
本机 SUMO_HOME 为 `/Library/Frameworks/EclipseSUMO.framework/EclipseSUMO/share/sumo`，
可执行文件位于该目录相对的 `../../bin/`；项目启动脚本已处理此布局。
XQuartz 首次安装后可能需要注销重新登录以刷新 DISPLAY；训练不应启动 GUI。

## Python 环境

```bash
bash scripts/setup.sh
bash scripts/run.sh fair-signal doctor
bash scripts/run.sh fair-signal smoke --steps 128
```

项目缓存和 Python 安装目录由 `scripts/env.sh` 设置。
`uv.lock` 应提交 Git；`.venv/`、`.runtime/`、`.cache/` 不提交。
后续依赖变更使用 uv 更新锁定文件并记录原因，不直接修改全局 Python 包。

## VS Code

打开 `code/`，选择 `.venv/bin/python`。终端使用 `bash scripts/run.sh ...`。
仓库提供的编辑器配置只作用于本工作区；不要求安装额外扩展。

## 其他机器

在目标机器安装 SUMO 1.28.0，设置 SUMO_HOME，运行同样的项目安装命令。
Windows 可用官方便携 ZIP；当前 Bash 启动脚本可在 Git Bash 中使用，或手动设置相同环境变量。
跨平台依赖由 uv.lock 描述，但本轮只验证本机 macOS，不宣称已测试 Windows。
本项目 CPU 默认即可，台式机 CUDA 加速需要另作 PyTorch 安装源及实际速度验证。

## 验证层次

1. `doctor`：Python、CPU 架构、包版本、SUMO 可执行文件及客户端版本。
2. `smoke`：SUMO-RL 初始化、真实交通仿真、少量 DQN 梯度更新、保存及重载。
3. GUI：安装并启动 XQuartz 后手动检查 sumo-gui 的窗口与渲染。
4. 真实环境集成：`bash scripts/run.sh pytest -m integration`，验证公共约束、删失和回放。
5. 短跑/正式实验：使用 `configs/pilot.yaml` / `configs/formal.yaml`，见 [运行指南](running.md)。

环境与集成检查不代表控制策略已收敛或有实际性能收益。

## 常见问题

- macOS Installer 无法从受保护的 Desktop 目录读取包时，可将已验证的包临时复制到
  任务专用 `/private/tmp/` 子目录，再执行安装；完成后删除中转副本。
- 受限执行沙箱可能禁止 localhost 端口绑定，使 TraCI 启动时报 remote-port=None。
  应允许本机进程通信，或在普通终端运行；不要把这一现象误判为 SUMO/TraCI 版本不兼容。
- XQuartz 首次安装后提示注销时，保存其他工作再自行注销登录；不能仅凭 GUI 程序能返回
  版本就声称已验证交通动画渲染。
