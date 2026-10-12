"""Failure-aware aggregation is part of the experimental evidence contract."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location(
    "iteration_report_test_module", SCRIPT_DIR / "summarize_iteration.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_failed_case_withholds_whole_scope_ranking():
    rows = []
    for seed, value in ((11, 10.0), (22, 30.0), (33, 20.0)):
        for _case in (1, 2):
            rows.append(
                dict(
                    split="development",
                    group="rate_1",
                    scenario="steady",
                    method="residual_dqn",
                    training_seed=seed,
                    valid_complete=True,
                    **{m: value for m in report.METRICS},
                )
            )
    frame = pd.DataFrame(rows)
    summary = report.summaries(frame)
    row = summary[(summary.method == "residual_dqn") & (summary.scenario == "all")].iloc[0]
    assert row.mean_wait == 20.0 and row.mean_wait_seed_sd == 10.0
    frame.loc[0, "valid_complete"] = False
    failed = report.summaries(frame)
    row = failed[(failed.method == "residual_dqn") & (failed.scenario == "all")].iloc[0]
    assert not row.eligible and row.complete_cases == 5 and pd.isna(row.mean_wait)
