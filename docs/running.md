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

```bash
bash scripts/run.sh fair-signal replay --run outputs/pilot --method dqn_fair \
  --seed 11 --scenario changing_peak --traffic-seed 2001 --output outputs/replay

# XQuartz 可用后添加 --gui
bash scripts/run.sh fair-signal replay --run outputs/pilot --method aging \
  --scenario imbalanced --traffic-seed 2001 --output outputs/gui-demo --gui
```

replay 从 selection.json 读取相位控制参数或模型候选，校验代码与模型身份。
它使用与原测试相同的网络、路线和 SUMO 种子，不向原实验目录覆盖结果。
GUI 是可视化仿真，首版不连接任何真实路口或硬件信号机。

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
