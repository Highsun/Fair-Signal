import runpy

import numpy as np
import pandas as pd
import pytest

from fair_signal.config import ExperimentConfig
from fair_signal.runtime import project_root

summarize = runpy.run_path(str(project_root() / "scripts/summarize_validation.py"))["summarize"]


def example():
    config = ExperimentConfig(
        methods=["fixed"], scenarios=["balanced"], fixed_candidates=[[25, 10, 25, 10]]
    )
    rows = [
        {
            "method": "fixed",
            "candidate": 0,
            "training_seed": None,
            "scenario": "balanced",
            "traffic_seed": seed,
            "split": "validation",
            "route_sha256": "same-demand-for-this-case",
            "valid_complete": True,
            "running": 0,
            "pending_insertion": 0,
            "collision_events": 0,
            "teleport_events": 0,
            "mean_wait": wait,
            "p95_wait": 30.0,
            "over_threshold_fraction": 0.0,
            "mean_queue": 2.0,
            "mean_depart_delay": 0.1,
        }
        for seed, wait in zip(config.validation_seeds, [10.0, 20.0], strict=True)
    ]
    return pd.DataFrame(rows), config


def test_incomplete_validation_never_exposes_finished_only_performance():
    frame, config = example()
    assert summarize(frame, config).iloc[0].mean_wait == 15.0
    frame.loc[0, "valid_complete"] = False
    frame.loc[0, "running"] = 3
    row = summarize(frame, config).iloc[0]
    assert not row.all_cases_complete and np.isnan(row.mean_wait)
    assert row.valid_cases == 1 and row.unfinished_vehicles == 3


@pytest.mark.parametrize("change", ["missing", "duplicate", "test"])
def test_wrong_validation_matrix_or_split_rejected(change):
    frame, config = example()
    if change == "missing":
        frame = frame.iloc[:1]
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError):
        summarize(frame, config)


def test_unmatched_demand_rejected():
    frame, config = example()
    config.fixed_candidates.append([20, 10, 40, 10])
    other = frame.copy()
    other["candidate"] = 1
    other["route_sha256"] = "different-demand"
    with pytest.raises(ValueError, match="identical demand"):
        summarize(pd.concat([frame, other]), config)
