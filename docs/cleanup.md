# 清理与卸载

先停止训练、TensorBoard、SUMO 及 GUI 进程，再操作文件。

## 保留成果、删除运行环境

保留源码、配置、uv.lock、文档，以及另行归档的模型/结果。
项目内 `.venv/`、`.runtime/`、`.cache/` 和不需要的 `outputs/` 内容可以删除。
虚拟环境不能脱离其项目 Python 运行，因此这两部分需要一起重建。
重建方式为 `bash scripts/setup.sh`。

共享 uv 及其其他课程环境不在清理范围内，不运行清空全局缓存/运行时目录的命令。
源码包、下载包、缓存和生成数据未加入 Git；删除它们不会改变仓库历史。

## 全局 SUMO

按安装记录检查 `org.eclipse.sumo.*` 安装收据和对应文件。
先列出实际安装文件，确认仅移除本次安装的版本及启动器。
本次官方包的框架目录为 `/Library/Frameworks/EclipseSUMO.framework`，可能还有应用启动器。
`pkgutil --forget` 只删除收据，不能代替删除程序文件。
如果其他项目开始使用 SUMO，则保留共享安装。

## XQuartz

XQuartz 涉及 `/opt/X11`、应用目录、LaunchAgents/LaunchDaemons、路径配置和安装记录。
使用与已安装版本对应的[官方卸载说明](https://www.xquartz.org/FAQs.html#uninstall-xquartz-28-and-later)。
清理时逐项确认归属，不直接复制宽泛的通配符删除命令。
用户偏好、GUI 历史和浏览器记录如有需要另行清理；不能承诺操作系统零痕迹。

## Git 与完整移除

Git 根目录只在 `code/.git/`。删除本仓库不会影响父目录其他课程资料。
GitHub 远程为 https://github.com/Highsun/Fair-Signal；删除本地项目不会删除远程仓库或提交历史。
本机运行环境、缓存和实验大文件被 Git 忽略，不随源码推送到 GitHub。
