# improvement-50k-3seed 验证结果（非最终测试）

训练种子 [11, 22, 33]，每模型 50000 步，需求 3600 秒。
共 15 个模型，纯训练合计 45.13 分钟。
验证完整回合 270/270，滞留车辆计数 0（跨方法/候选求和）。

以下均为验证选参后的描述性结果，不能作为最终泛化性能。P95 为回合 P95 的平均。
任一场景有不完整回合时该候选/场景性能留空；完整性仍显示。

| 场景 | 方法 | 候选 | 完整回合 | 平均等待 s | 平均回合 P95 s | 超 90 秒比例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced | actuated | 0 | 5/5 | 14.50 | 48.89 | 0.0109 |
| changing_peak | actuated | 0 | 5/5 | 13.51 | 45.83 | 0.0113 |
| imbalanced | actuated | 0 | 5/5 | 13.24 | 46.24 | 0.0136 |
| balanced | aging | 0 | 5/5 | 16.11 | 42.36 | 0.0000 |
| changing_peak | aging | 0 | 5/5 | 15.13 | 40.80 | 0.0000 |
| imbalanced | aging | 0 | 5/5 | 14.98 | 40.57 | 0.0000 |
| balanced | dqn_dense | 0 | 15/15 | 16.59 | 59.46 | 0.0172 |
| changing_peak | dqn_dense | 0 | 15/15 | 15.37 | 58.55 | 0.0171 |
| imbalanced | dqn_dense | 0 | 15/15 | 14.14 | 56.69 | 0.0138 |
| balanced | dqn_dense_state | 0 | 15/15 | 16.08 | 52.46 | 0.0105 |
| changing_peak | dqn_dense_state | 0 | 15/15 | 15.11 | 54.44 | 0.0136 |
| imbalanced | dqn_dense_state | 0 | 15/15 | 14.15 | 57.50 | 0.0133 |
| balanced | dqn_fair | 0 | 15/15 | 15.89 | 52.79 | 0.0069 |
| changing_peak | dqn_fair | 0 | 15/15 | 14.84 | 55.23 | 0.0095 |
| imbalanced | dqn_fair | 0 | 15/15 | 14.38 | 56.83 | 0.0125 |
| balanced | dqn_queue | 0 | 15/15 | 16.09 | 58.80 | 0.0171 |
| changing_peak | dqn_queue | 0 | 15/15 | 14.86 | 56.39 | 0.0199 |
| imbalanced | dqn_queue | 0 | 15/15 | 13.76 | 53.75 | 0.0183 |
| balanced | dqn_wait | 0 | 15/15 | 16.02 | 53.90 | 0.0132 |
| changing_peak | dqn_wait | 0 | 15/15 | 15.30 | 57.16 | 0.0184 |
| imbalanced | dqn_wait | 0 | 15/15 | 14.30 | 58.75 | 0.0187 |
| balanced | fixed | 0 | 5/5 | 24.79 | 59.40 | 0.0000 |
| changing_peak | fixed | 0 | 5/5 | 25.96 | 59.85 | 0.0005 |
| imbalanced | fixed | 0 | 5/5 | 27.84 | 63.93 | 0.0015 |

完整候选、失败回合及成对差异见同目录 CSV；所有数值来自真实运行。
单训练种子不能估计训练方差；随机训练回合曲线不能证明收敛。
预留最终测试种子尚未进行性能评估。
