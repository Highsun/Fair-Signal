# 运行与部署

以下均在 `code/` 执行。执行入口是安装于项目虚拟环境的 `fair-signal`；
`scripts/run.sh` 为它配置项目 Python、缓存和 SUMO 路径。

## 环境与测试

```bash
bash scripts/setup.sh
bash scripts/run.sh fair-signal doctor
bash scripts/run.sh ruff check .
bash scripts/run.sh pytest
bash scripts/run.sh pytest -m integration
```

默认 pytest 执行不启动 SUMO 的单元测试；`-m integration` 启动本机仿真端口。
集成测试包含故意让部分方向饥饿的失败策略、逐秒约束、重复观测一致性、
DQN 参数实际更新、模型保存重载、验证/测试分离、删失统计与可重复回放。

## 短跑与正式运行

```bash
# 完整短跑；5 个模型，每个 1 万决策步，1 个训练种子
bash scripts/run.sh fair-signal run --config configs/pilot.yaml --output outputs/pilot

# 分阶段执行（顺序：prepare -> train -> tune -> test）
bash scripts/run.sh fair-signal run --config configs/pilot.yaml --output outputs/pilot --stage train
bash scripts/run.sh fair-signal run --config configs/pilot.yaml --output outputs/pilot --stage tune
bash scripts/run.sh fair-signal run --config configs/pilot.yaml --output outputs/pilot --stage test

# 从原始测试表重新汇总/绘图
bash scripts/run.sh fair-signal analyze --run outputs/pilot

# 长预算，确认短跑结果和时间预算后显式运行
bash scripts/run.sh fair-signal run --config configs/formal.yaml --output outputs/formal
```

pilot 需求注入 900 秒，额外清空上限 900 秒；正式需求注入 3600 秒、清空上限 1800 秒。
正式配置包含 3 个普通 DQN、3 个等待状态 DQN、3 个公平权重×3 种子，共 15 次训练。
30 万步为初定预算，并不保证收敛。根据短跑吞吐与学习曲线决定是否值得投入此预算。
短跑测试种子 2001/2002 已用于开发诊断，正式配置改用保留的 3001–3010；
不要用正式测试来反复调整参数。原始 experiment-plan.yaml 仅为历史规划存档。

相同源码、uv.lock、pyproject 和配置可复用已完成阶段；运行目录会校验代码与配置身份。
未完成的某个模型下次会从头训练；不支持无损恢复整个回放池/优化器状态。
不要手改已有输出再继续实验。修改代码或参数时选择新目录，例如 `outputs/pilot-v2`。
中止可按 Ctrl+C，finally 会关闭当前 SUMO 连接；不要删除其他项目的进程。

## 加长训练的验证实验

介于短跑与完整多种子实验之间，`configs/stability.yaml` 固定 5 个模型、每个 5 万步、
1 个训练种子、3600 秒需求和最多 1800 秒清空；14 个验证候选共 210 回合。
这一阶段只训练和验证，不执行最终测试：

```bash
bash scripts/run.sh fair-signal run --config configs/stability.yaml \
  --output outputs/stability-50k --stage train
bash scripts/run.sh fair-signal run --config configs/stability.yaml \
  --output outputs/stability-50k --stage tune
bash scripts/run.sh python scripts/summarize_validation.py --run outputs/stability-50k
bash scripts/run.sh tensorboard --logdir outputs/stability-50k/models --host 127.0.0.1
```

汇总脚本校验完整候选矩阵、同车流匹配与模型校验值，拒绝混入测试记录，
将小体积指标和图表导出至 `docs/results/stability-50k/`。
`reward_activation.csv` 只统计固定需求窗口：任一车道停车队首超过 90 秒的逐秒比例，
等价于公平惩罚项非零的比例；它不等于惩罚幅度，也不是车辆超时比例。
超过 180 秒的比例用于检查队首等待观测可能发生的裁剪，不能反映所有状态分量。
完整方案与实际结果见 [加长验证记录](stability-results.md)。

## 改进算法的三种子验证

```bash
# 首批 15 个模型对应源码版本 b90018e；复现/重导出时使用该版本
# 固定 15 个模型、每个 5 万步；三个独立 CPU 工作进程
bash scripts/run.sh python scripts/improvement_study.py run --workers 3
# 可改 --workers 1 串行；完成模型与验证回合会校验后复用
bash scripts/run.sh python scripts/summarize_validation.py --run outputs/improvement-50k \
  --destination docs/results/improvement-50k
bash scripts/run.sh python scripts/improvement_study.py summarize
bash scripts/run.sh tensorboard --logdir outputs/improvement-50k/models --host 127.0.0.1

# 当前版本补充：原公平奖励＋44 维状态，三个种子
bash scripts/run.sh python scripts/improvement_study.py run \
  --config configs/fair-state.yaml --output outputs/fair-state-50k --workers 3
bash scripts/run.sh python scripts/summarize_validation.py --run outputs/fair-state-50k \
  --destination docs/results/fair-state-50k

# 复核历史行为：246 个早期回合＋270 个首批改进回合
bash scripts/run.sh python scripts/check_legacy.py --include-improvement \
  --output outputs/legacy-fair-state-check --workers 2
# 校验两个已导出的冻结实验，生成四格对照表与图
bash scripts/run.sh python scripts/summarize_factorial.py

# 观看新状态版本，复用已见过的开发种子 2001，不作为最终测试
bash scripts/watch.sh --run outputs/improvement-50k --method dqn_dense_state --seed 11
```

首批源码摘要与补充试验不同，原输出目录不能在新源码下续跑或重导出。
需要复现首批时，在相应版本的独立检出中运行，勿重置有修改的工作区。
四格汇总明确保留两个来源摘要，并要求历史行为回归通过，不绕过原身份检查。

三个种子 11/22/33 共享固定配置，分别写入自身模型目录；父进程评估传统基线。
该脚本只创建验证车流，不生成预留最终测试车流。程序只在所有工作进程成功后统一选参；
短预算 smoke 的失败性能不能当作正式结果。实验设计与分析见
[改进记录](improvement-results.md)。
新轨迹的 fair_cost 为当前方法的实际公平成本定义；`reward_activation.csv` 优先使用此列。
历史轨迹缺少该列时，才以队首超阈值判断原成本是否非零。

## 固定探索日程的长预算验证

```bash
# 两种 DQN，各三个种子；六条不中断的 15 万步轨迹
bash scripts/run.sh python scripts/long_budget_study.py --workers 3
# 核查三组检查点、前缀一致性并导出结果
bash scripts/run.sh python scripts/summarize_long_budget.py
bash scripts/run.sh tensorboard --logdir outputs/long-budget-150k/trajectories --host 127.0.0.1
```

`configs/long-budget.yaml` 固定前 25000 步探索衰减、原公平权重 1.0，
保存 50000/100000/150000 步梯度更新后的模型，完成整条轨迹后才评估。
模型在 `outputs/long-budget-150k/step_<步数>/models/`，
完整训练日志和 TensorBoard 在 `trajectories/`；18 个检查点来自六条独立轨迹。
不按最好检查点选模型，主要报告 150000 步末端。
实验脚本本身也有 SHA-256 身份校验；不要改已有运行目录后继续训练。
新验证共 300 回合，预留最终测试不生成、不运行。
实验设计与实际结果见 [长预算记录](long-budget-results.md)。

## 共享服务护栏对照

```bash
# 先完成上面的 15 万步实验；复用六个固定末端模型，无新增训练
bash scripts/run.sh python scripts/guard_study.py --workers 3
bash scripts/run.sh python scripts/summarize_guard_study.py
```

固定 180 秒请求年龄阈值，同一护栏用于普通 DQN、Fair DQN 与感应式。
原始病例、实际相位轨迹与每个决策的 proposed/executed/overridden 记录在
`outputs/service-guard-150k/evaluation/`；小体积导出在 `docs/results/service-guard-150k/`。
混合控制器分别命名为 `dqn_queue_guard`、`dqn_fair_guard`、`actuated_guard`，
不会修改原策略模型或替换原实验结果。程序会核查脚本、护栏、核心与模型摘要。
这轮没有新增模型训练，因此护栏输出目录没有新的 TensorBoard；
其基础模型的训练曲线仍在上面的 `long-budget-150k/trajectories/`。

## 产物

```text
outputs/pilot/
├── config.json                    # 完整展开的参数 + 源码内容摘要
├── provenance.json                # Git、依赖、平台、CPU、验证目标
├── network/                       # 原创输入 XML、netconvert 结果及校验值
├── cases/{validation,test}/       # 所有方法共用的显式车辆列表与清单
├── models/<method>/candidate_N/seed_N/
│   ├── model.zip                  # SB3 可直接加载的策略
│   ├── training.json              # 实际步数、梯度更新、耗时、模型 SHA-256
│   ├── learning.csv / monitor.csv # 完成回合的原始交通指标/训练回报
│   ├── tensorboard/               # 训练诊断
│   └── episodes/                  # 训练需求、TripInfo 和回合指标
├── evaluation/{validation,test}/  # 每模型/每条车流的评估身份及逐秒轨迹
├── validation.csv / validation_scores.json
├── selection.json                 # 仅由验证集冻结的选择，含诊断失败标记
├── test.csv / per_model_summary.csv / summary.csv
├── quality.json / report.md
└── figures/                       # 学习、效率公平性、方向等待、车流和相位图
```

单次评估下的 `episode_00000/` 包含 `tripinfo.xml`、`trips.csv`、`trace.csv`、`metrics.json`
与 `sumo.log`。测试发生删失时，报告留空的性能列不能视作零等待。
`valid_complete=false` 的具体原因可查看原始 metrics，不应只看图片排名。

查看 TensorBoard：

```bash
bash scripts/run.sh tensorboard --logdir outputs/pilot/models --host 127.0.0.1
```

各奖励配置的 TensorBoard 回报不能跨配置比大小；优劣用相同测试车流的原始交通指标判断。

## 加载与演示

推荐直接运行一小时车流的慢速演示：

```bash
bash scripts/watch.sh

# 更慢：每个仿真秒至少显示约 1 秒；也可换为传统控制
bash scripts/watch.sh --gui-delay 1000 --method aging
```

默认复用已有公平 DQN，不重新训练；显示 changing_peak 的 3600 秒需求及清空阶段。
启动后停在初始状态，点击 SUMO 工具栏 **▶ Play** 才开始。默认 Delay=200 ms，
对应约 5 倍速，需求阶段约能观看 12 分钟；计算/渲染开销和清空阶段会增加实际耗时。
可随时用 **■ Stop** 暂停，暂停后 **Single Step / Ctrl+D** 每次前进 1 秒；
Delay 可在 GUI 中继续修改。加 `--autoplay` 才会自动开始，加 `--gui-delay 0` 才以最快速度播放。
结束后窗口关闭；等待初始播放时终端没有继续输出属于正常现象。
Ctrl+C 会关闭该次演示自己的 GUI 子进程，包括尚未开始的暂停窗口。
观看脚本在 macOS 中优先使用 DISPLAY 或 launchctl 的显示地址，必要时连接已经存在的
XQuartz `:0` 本地套接字；不修改全局环境变量或 shell 配置。
每次输出保存到新的 `outputs/gui-1h/<时间戳>/`，可用 `--output` 指定其他项目输出目录。

完整命令及原测试回放入口：

```bash
bash scripts/run.sh fair-signal replay --run outputs/pilot --method dqn_fair \
  --seed 11 --scenario changing_peak --traffic-seed 2001 --output outputs/replay

# 一小时车流 + 慢速 + 等待手动开始
bash scripts/run.sh fair-signal replay --run outputs/pilot --method dqn_fair \
  --scenario changing_peak --traffic-seed 2001 --demand-seconds 3600 \
  --output outputs/gui-demo --gui --gui-delay 200

# 原测试车流也默认暂停、慢速
bash scripts/run.sh fair-signal replay --run outputs/pilot --method aging \
  --scenario imbalanced --traffic-seed 2001 --output outputs/gui-demo --gui
```

replay 从 selection.json 读取相位控制参数或模型候选，校验代码、网络与模型身份。
未指定需求时长时，使用原测试路线与种子。指定 `--demand-seconds` 时，在新输出目录
生成演示车流，并标记 `purpose=demonstration`；原始实验配置和统计不变。
这是延长推理/观看时间，不是增加训练步数，也不自动成为正式性能实验。
GUI 是可视化仿真，首版不连接任何真实路口或硬件信号机。

此次观看功能更新与原始首轮源码之间记录了明确的 SHA-256 兼容映射，
并逐一回放原始 36 个测试实例，所有指标一致（包括有删失的失败回合）。
旧模型因此可以继续观看；模型、路线和网络校验仍执行，训练/评估续跑仍要求精确源码身份。
未来任意源码修改会使这条映射失效，不提供通用“跳过校验”选项。

已在本机实际运行带延迟的 SUMO GUI 自动播放通路，并与同车流无界面指标核对。
一小时车流的独立无界面检查中，1192 辆车全部完成，清空用时 81 秒；此结果仅用于演示检查。
按钮位置和快捷键可参考 [SUMO GUI 官方说明](https://sumo.dlr.de/docs/sumo-gui.html)。

## 跨机器复现与清理

保留本仓库的对应版本及 uv.lock，另行传输整个 `outputs/<run>/` 到同样的仓库位置；
安装 SUMO 1.28.0，配置 SUMO_HOME，执行 setup 和 doctor，然后 replay。
来源 JSON 中的绝对路径仅作诊断，运行时重新解析本机仓库与 SUMO 路径。
Windows 可用 Git Bash 或手动设置 scripts/env.sh 对应变量；首轮仅实际验证 macOS CPU。
小 MLP 的负载主要包括仿真与进程通信，不能假定 RTX GPU 会按算力倍数加速。

安装阶段之外没有下载外部资料、数据集或代码仓库，没有新增 Python 依赖。
训练文件只在本仓库输出目录与缓存中；清理不需要操作 CS336 环境或共享 uv。
GitHub 保存源码、配置、文档与小型结果记录；大模型/原始实验目录默认不推送。
项目结束后的删除范围见 [清理说明](cleanup.md)。
