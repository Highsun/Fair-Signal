# 第三方依赖与资料

本项目 MIT 许可证只适用于原创代码，不重新许可第三方依赖或资料。

| 项目 | 来源 | 许可证/用途 |
| --- | --- | --- |
| SUMO | https://github.com/eclipse-sumo/sumo | EPL-2.0 / GPL-2.0-or-later 选项；仿真工具 |
| SUMO-RL | https://github.com/LucasAlegre/sumo-rl | MIT；环境接口，联通检查读取其已安装路网 |
| Stable-Baselines3 | https://github.com/DLR-RM/stable-baselines3 | MIT；DQN 实现 |
| PyTorch | https://github.com/pytorch/pytorch | BSD-style；神经网络库及其第三方声明 |
| Gymnasium | https://github.com/Farama-Foundation/Gymnasium | MIT；环境接口 |

其他间接依赖以各发行包自带的许可证为准。当前没有将第三方仓库整体复制或内嵌进本仓库。
若后续分发路网、应用程序或依赖二进制，需要核对对应文件的独立许可及通知要求。

研究背景：Raeis and Leon-Garcia, *A Deep Reinforcement Learning Approach for Fair Traffic Signal Control*,
2021, https://arxiv.org/abs/2107.10146 。当前仅引用研究思想，不复制论文内容或作者代码。
