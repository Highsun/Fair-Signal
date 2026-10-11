# 场景资料

正式实验使用 `src/fair_signal/demand.py` 中的原创四进口路网生成器。
每次实验的节点、路段、连接、信号 XML、netconvert 结果与 SHA-256 保存于
`outputs/<run>/network/`；不下载真实道路数据，也不复制第三方网络。

显式随机车辆列表保存于 `outputs/<run>/cases/{validation,test}/`，由所有方法共用。
训练车流在对应模型的 `episodes/` 下保存；每份清单记录需求强度、种子、车辆数与校验值。
场景、车辆参数与划分见 [实现说明](../docs/implementation.md)。
生成文件随实验隔离并被 Git 忽略，删除某个实验目录不会影响其他实验。

环境联通检查直接读取已安装 SUMO-RL 自带的单路口网络，并把临时文件写入
`outputs/smoke/`；不复制第三方路网到仓库，不将其许可证替换为本项目 MIT。
