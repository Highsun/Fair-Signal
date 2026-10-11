import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from fair_signal.analysis import METRICS, hierarchical_summary
from fair_signal.config import ExperimentConfig
from fair_signal.demand import generate_routes, sha256
from fair_signal.environment import trip_metrics


def test_route_reproducibility_and_demand_window(tmp_path):
    first = generate_routes(tmp_path / "a", "imbalanced", "test", 2001, 900)
    same = generate_routes(tmp_path / "b", "imbalanced", "test", 2001, 900)
    other = generate_routes(tmp_path / "c", "imbalanced", "test", 2002, 900)
    assert sha256(first) == sha256(same) != sha256(other)
    vehicles = ET.parse(first).getroot().findall("vehicle")
    departures = [float(v.get("depart")) for v in vehicles]
    assert departures == sorted(departures)
    assert min(departures) >= 0 and max(departures) < 900
    assert len({v.get("id") for v in vehicles}) == len(vehicles)
    assert {v.get("departLane") for v in vehicles} == {"0", "1"}


def test_seed_leakage_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        ExperimentConfig(test_seeds=[11]).validate()


def test_trip_totals_and_unfinished_are_separate(tmp_path):
    path = tmp_path / "trips.xml"
    path.write_text("""<tripinfos>
      <tripinfo id="n_1" arrival="1000" waitingTime="400" duration="1000" departDelay="20"/>
      <tripinfo id="e_2" arrival="50" waitingTime="10" duration="50" departDelay="0"/>
      <tripinfo id="s_3" arrival="-1" waitingTime="900" duration="1000"/>
      <tripinfo id="w_4" arrival="20" waitingTime="3" vaporized="collision"/>
    </tripinfos>""")
    summary, rows = trip_metrics(path, 90)
    assert summary["completed"] == 2 and len(rows) == 4
    assert summary["mean_wait"] == 205
    assert summary["p95_wait"] == pytest.approx(380.5)
    assert summary["over_threshold_fraction"] == 0.5
    assert summary["unfinished_wait_max"] == 900
    assert summary["mean_system_time"] == 535


def test_summary_averages_within_model_before_across_models():
    rows = []
    for seed, waits in ((11, [10, 20]), (22, [30, 40])):
        for wait in waits:
            rows.append(
                {
                    "method": "dqn_fair",
                    "scenario": "balanced",
                    "training_seed": seed,
                    "valid_complete": True,
                    **dict.fromkeys(METRICS, wait),
                }
            )
    per_model, summary = hierarchical_summary(pd.DataFrame(rows))
    assert per_model.mean_wait.tolist() == [15, 35]
    assert summary.iloc[0].mean_wait == 25
    assert summary.iloc[0].mean_wait_training_std == pytest.approx(14.1421356)


def test_summary_does_not_hide_failed_model_behind_successful_one():
    frame = pd.DataFrame(
        [
            {
                "method": "dqn_queue",
                "scenario": "balanced",
                "training_seed": seed,
                "valid_complete": valid,
                **dict.fromkeys(METRICS, 1.0),
            }
            for seed, valid in ((11, True), (22, False))
        ]
    )
    _, summary = hierarchical_summary(frame)
    assert not summary.iloc[0].all_cases_complete
    assert pd.isna(summary.iloc[0].mean_wait)
