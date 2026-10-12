# Baseline-assisted DQN: audited development results

Purpose: `development_validation`; 50000 steps/model; 3 model seeds. No reserved test evaluated.

Supplementary diagnostic added after inspecting results: mean_excess_wait = mean(max(lifetime_stop_wait - 90, 0)); primary metrics unchanged.

| Method | Complete | Mean wait | P95 | >90s | System time | Correction rate |
| --- | --- | --- | --- | --- | --- | --- |
| actuated | 15/15 | 32.08 | 96.24 | 6.37% | 114.42 | 0.00% |
| max_pressure | 15/15 | 33.21 | 102.48 | 7.42% | 113.09 | 0.00% |
| aging | 15/15 | 38.64 | 95.62 | 5.94% | 128.05 | 0.00% |
| always_extend | 15/15 | 35.54 | 108.54 | 8.12% | 114.70 | 30.99% |
| always_urgent | 15/15 | 33.96 | 92.53 | 8.31% | 119.05 | 3.10% |
| residual_dqn | 45/45 | 30.35 | 92.54 | 5.94% | 110.52 | 8.50% |
| fair_residual_dqn | 45/45 | 30.87 | 91.51 | 6.47% | 111.47 | 8.30% |
