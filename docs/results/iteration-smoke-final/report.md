# Audited iteration results

Connectivity check only; not performance evidence.

Groups stay separate. SD is across model-seed means, not a confidence interval.

## development: rate_1

| Method | Complete | Wait(s) | System(s) | >90s | P95(s) | Excess(s) |
| --- | --- | --- | --- | --- | --- | --- |
| actuated | 3/3 | 36.61 | 122.71 | 7.78% | 110.42 | 6.25 |
| max_pressure | 3/3 | 37.75 | 118.77 | 10.48% | 122.65 | 6.60 |
| aging | 3/3 | 39.40 | 129.85 | 7.90% | 102.40 | 3.62 |
| residual_dqn | 3/3 | 35.11 | 119.07 | 9.19% | 102.13 | 4.54 |
| fair_residual_dqn | 3/3 | 36.17 | 120.44 | 9.93% | 100.50 | 4.59 |
| excess_025 | 3/3 | 35.41 | 116.14 | 8.64% | 103.33 | 3.77 |
| crossing_30 | 3/3 | 36.56 | 117.73 | 8.90% | 103.67 | 3.75 |
| crossing_90 | 3/3 | 36.54 | 117.17 | 8.37% | 104.07 | 4.29 |
| matched_full | 3/3 | 40.53 | 125.64 | 10.72% | 120.00 | 5.76 |
| matched_local | 3/3 | 42.79 | 130.61 | 10.92% | 127.90 | 6.67 |

