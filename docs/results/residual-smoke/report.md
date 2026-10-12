# Baseline-assisted DQN: audited development results

| Method | Complete | Mean wait | P95 | >90s | System time | Correction rate |
| --- | --- | --- | --- | --- | --- | --- |
| actuated | 3/3 | 30.97 | 97.82 | 5.22% | 112.89 | 0.00% |
| max_pressure | 3/3 | 31.15 | 86.57 | 6.17% | 110.71 | 0.00% |
| aging | 3/3 | 34.49 | 82.32 | 4.41% | 120.50 | 0.00% |
| always_extend | 3/3 | 33.87 | 102.38 | 7.30% | 112.96 | 30.99% |
| always_urgent | 3/3 | 30.13 | 82.00 | 6.84% | 113.06 | 2.76% |
| residual_dqn | 3/3 | 31.89 | 97.88 | 6.67% | 112.58 | 15.67% |
| fair_residual_dqn | 3/3 | 29.54 | 85.12 | 5.21% | 109.22 | 15.60% |
