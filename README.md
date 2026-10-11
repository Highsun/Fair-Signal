# Fair Signal

**兼顾通行效率与等待公平性的深度强化学习信号控制。**

面向深度强化学习课程的可复现实验项目。使用 SUMO 仿真一个十字路口，研究在
方向不均衡、车流随时间变化时，DQN 能否减少整体排队，同时改善少数车辆的长时间等待。

本仓库独立位于 `code/`，同步至 [Highsun/Fair-Signal](https://github.com/Highsun/Fair-Signal)。
后续完成本地提交后，将相应分支同步到 GitHub。
本项目原创代码采用 [MIT License](LICENSE)，依赖与第三方资料保留各自许可证。

## 当前状态

已实现**首版完整实验管线**：原生路网与可复现车流生成、统一信号约束、三种传统控制器、
三个 DQN 消融、验证选参、固定测试、模型保存/回放、分层统计与绘图。
已完成 1 万步短跑、每模型 5 万步的加长验证，以及奖励×状态的三种子开发对照。
后者共 18 个学习模型，每模型 5 万步，统一使用 1 小时验证车流。
随后完成六条 15 万步轨迹的 [长预算与服务护栏对照](docs/long-budget-results.md)：
发现纯 DQN 在清空尾段可能相位饥饿，加入共享服务护栏后 105/105 回合完成。
混合控制与纯 DQN 分开评价，保留未取得优势及删失的结果。
30 万步的正式实验尚未执行。
现新增独立的[两路口协同控制开发实验](docs/arterial-plan.md)，保留历史单路口成果。
两路口联通、逐秒物理约束、累计停车记账和三种 DQN 的实际更新/重载检查已通过；
每模型 5 万步、三个训练种子的开发验证正在进行，不预设学习方法胜出。
短跑和环境联通检查都不构成算法收敛或优越性的证据。
设计细节见 [实现说明](docs/implementation.md)，实际安装见 [环境记录](docs/environment.md)。
首轮实测与失败分析见 [短跑结果](docs/pilot-results.md)：传统基线暂时领先，不能宣称 DQN 更优。
[加长验证](docs/stability-results.md) 保留首个训练种子的历史结果。
最新 [算法检查与改进实验](docs/improvement-results.md) 检查奖励和观测逻辑，
增加连续公平成本、完整信号状态及四格消融，并保留未取得优势的结果。
不能用单种子排名判断算法优劣；预留最终测试集尚未评估。

## 研究问题

在东西方向车流持续较大、南北方向车流较少时，单纯优化总排队可能使少量车辆等待过久。
本项目考察：增加等待状态和超时惩罚，是否能在较小效率代价下减少极端等待？

这是对已有公平交通控制思想的课程实现与实验研究，不宣称提出全新算法。
不会预先保证 DQN 优于调节合理的传统控制，也不会把奖励阈值解释为车辆通行时间保证。

## 项目范围

| 内容 | 首版设定 |
| --- | --- |
| 环境 | 一个四进口路口、普通机动车、单智能体 |
| 输入 | 数值交通状态，不加入视觉识别 |
| 动作 | 选择预定义合法绿灯相位 |
| 控制周期 | 每 5 秒决策；最小绿灯 10 秒，最大 60 秒，黄灯 3 秒，全红 1 秒 |
| 算法 | CPU 上训练的小型 DQN，经验回放和目标网络 |
| 改进点 | 等待信息增强、超时等待惩罚、效率与公平性的权衡 |
| 首版边界 | 不加入多路口、行人、公交优先或真实道路部署 |

## 预期实验

| 控制方法 | 比较目的 |
| --- | --- |
| 调节后的固定配时 | 静态基线 |
| 感应式控制 | 根据当前需求调整的传统基线 |
| 加入等待老化规则的传统控制 | 检查简单防饥饿规则是否足够 |
| 普通 DQN | 学习方法基线 |
| 增加等待状态的 DQN | 分离状态信息的作用 |
| 增加等待状态和公平奖励的 DQN | 分离公平目标的作用 |
| 连续公平成本 × 原/完整状态 | 检查信号密度与状态表示的作用 |
| 原公平成本＋完整状态 | 完成奖励×状态四格对照 |

三类车流：均衡、方向不均衡、时变高峰。训练/验证/测试种子分离，同一测试车流供所有方法使用。
正式配置为每种学习配置训练 3 个随机种子，每次 30 万决策步，先做短跑再调整预算。
公平性权重用验证集选择，测试集只作最终评估。

主要指标：车辆平均等待、P95 等待、平均行程时间、平均队列、通行数、未完成/未插入车辆数、
相位切换数与各方向等待。用原始交通指标评价，不能跨奖励定义比较累计奖励。
完整定义、风险和消融设计见 [实验方案](docs/experiment-plan.md)。

## 流程

```mermaid
flowchart TD
    A[检查安装与仿真联通] --> B[建立路网与可复现车流]
    B --> C[实现传统控制与普通 DQN]
    C --> D[验证等待状态和指标口径]
    D --> E[实现公平性奖励]
    E --> F[验证集选择参数]
    F --> G[固定测试集与多个训练种子评估]
    G --> H[消融与效率公平性权衡分析]
    H --> I[报告 模型 演示与复现说明]
```

## 环境要求

- macOS Apple Silicon（本机为 M3 Pro）或具有 SUMO 的其他平台。
- [uv](https://docs.astral.sh/uv/)；已有 uv 可直接复用。
- Python 3.12；由 uv 放入项目 `.runtime/python/`，无需更改系统 Python。
- 全局 SUMO 1.28.0；Python 端的 TraCI/sumolib 固定为对应版本。
- macOS 原生 GUI 使用 XQuartz；无界面训练无需启动 XQuartz 会话。

系统安装与清理见 [安装说明](docs/setup.md) 和 [清理说明](docs/cleanup.md)。

## 快速开始

以下命令在本仓库根目录（`code/`）执行。首次安装需要网络。

```bash
# 全局 SUMO 安装完成后，建立项目环境并检查
bash scripts/setup.sh

# 查看 Python 包版本与 SUMO 路径
bash scripts/run.sh fair-signal doctor

# 执行有限步数的仿真、DQN 更新与模型保存/加载检查
bash scripts/run.sh fair-signal smoke --steps 128

# 首轮端到端短跑：5 个学习配置各 1 万步，随后验证与固定测试
bash scripts/run.sh fair-signal run --config configs/pilot.yaml --output outputs/pilot

# 重新生成结果表、质量检查与图表，不重新训练
bash scripts/run.sh fair-signal analyze --run outputs/pilot

# 加载验证集选出的公平 DQN，回放一条测试车流
bash scripts/run.sh fair-signal replay --run outputs/pilot --method dqn_fair \
  --seed 11 --scenario imbalanced --traffic-seed 2001 --output outputs/replay

# 代码检查
bash scripts/run.sh ruff check .
bash scripts/run.sh pytest

# 真实 SUMO 集成检查（需要允许本机 TraCI 端口）
bash scripts/run.sh pytest -m integration
```

`scripts/run.sh` 会设置项目缓存目录和 SUMO 路径，不修改全局 shell 配置。
如果 SUMO 装在其他位置，用 `SUMO_HOME=/实际路径 bash scripts/run.sh ...` 覆盖。
直接使用 `uv run` 时，应先在 Bash 中 `source scripts/env.sh`。

联通测试会读取 SUMO-RL 自带路网，并在 `outputs/smoke/` 生成临时车流、模型与检查报告。
这些文件不进入 Git。正式实验使用项目原创路网生成器，不依赖该示例路网。

运行结果位于 `outputs/pilot/report.md`；原始测试表为 `test.csv`，模型级均值为
`per_model_summary.csv`，跨训练种子的统计为 `summary.csv`，图表在 `figures/`。
模型位于 `models/<方法>/candidate_<编号>/seed_<种子>/model.zip`。
`selection.json` 记录验证选择；`quality.json` 记录未完成、传送与碰撞检查。
有不完整回合的方法/场景不输出汇总性能，原始删失结果仍保留。

可分阶段运行 `--stage prepare/train/tune/test`。相同代码和配置下，已完成的训练和评估
会校验并复用；中断的单个训练从头重跑，不声称支持优化器/回放池断点续训。
修改代码或配置后必须使用新的输出目录，防止混入旧结果。

长预算实验显式运行以下命令；会产生 15 个 30 万步训练，**不建议直接作为首次运行**：

```bash
bash scripts/run.sh fair-signal run --config configs/formal.yaml --output outputs/formal
```

GUI 推荐执行 `bash scripts/watch.sh`：1 小时仿真车流、默认初始暂停、200 ms 慢速，
点击 Play 后约可观看 12 分钟及清空阶段。无需重新训练，需 XQuartz 会话可用。
`bash scripts/watch.sh --gui-delay 1000` 接近实时观看；`--autoplay` 可改为自动开始。
已有首轮模型经精确代码兼容映射与全部 36 个测试回放核对，可以继续使用。
完整执行、产物和部署说明见 [运行指南](docs/running.md)。

## 仓库结构

```text
code/
├── src/fair_signal/       # 环境、信号状态机、控制器、训练、评估、分析与回放
├── scripts/              # 环境设置和运行脚本
├── configs/              # 短跑、加长验证、改进消融与正式配置
├── data/                 # 场景说明；实际生成数据随各实验保存
├── tests/                # 关键行为检查
├── docs/                 # 实验、环境、安装与清理文档
├── outputs/              # 日志、模型和图表（Git 忽略）
├── .venv/                # 独立 Python 环境（Git 忽略）
├── .runtime/             # 项目 Python（Git 忽略）
├── .cache/               # 安装包和工具缓存（Git 忽略）
├── pyproject.toml
├── uv.lock               # 精确依赖版本
├── README.md
└── LICENSE
```

## 最终成果与里程碑

1. **环境准备**：本地仓库、依赖锁定、SUMO 联通、安装与清理记录。
2. **基础实验**：可复现车流、传统基线、普通 DQN 和统一指标。
3. **公平性实验**：增强状态、奖励、消融和公平性权重扫描。
4. **最终交付**：问题与算法设计、流程图、源码、执行入口、已训练模型、部署说明、
   多种子结果表、效率公平性图、失败案例与 GUI 演示。

本项目的“执行程序”为 `fair-signal` 命令行入口与模型加载程序，通过 `scripts/run.sh`
执行；首版不分发独立桌面应用。训练配置、路线/模型校验值和环境来源随每次实验保存。
后续正式阶段需要更充分的训练、多种子结果和最终课程报告。

## 开发与贡献

参见 [CONTRIBUTING.md](CONTRIBUTING.md)。使用 `src/` 布局、Ruff 和 pytest。
在 VS Code 中打开本仓库，选择 `.venv/bin/python` 即可。当前没有远程 CI 或 GitHub Actions。

变更记录见 [CHANGELOG.md](CHANGELOG.md)，引用与依赖许可证见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 参考资料

- [SUMO](https://eclipse.dev/sumo/)：交通微观仿真。
- [SUMO-RL](https://github.com/LucasAlegre/sumo-rl)：Gymnasium 接口与单路口示例。
- [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)：标准 DQN 实现。
- [A Deep Reinforcement Learning Approach for Fair Traffic Signal Control](https://arxiv.org/abs/2107.10146)：公平性研究背景。
