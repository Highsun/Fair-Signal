import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from fair_signal.arterial.demand import build_network, generate_routes
from fair_signal.config import EnvConfig
from fair_signal.iteration.environment import IterationEnv
from fair_signal.residual.environment import ResidualEnv

pytestmark = pytest.mark.integration
PARAMETERS = {"minimum": 10, "hysteresis": 1.0}


def test_scheduled_environment_matches_frozen_residual_and_counts_crossings(tmp_path):
    config = EnvConfig(demand_seconds=300, drain_seconds=600, queue_scale=160, wait_threshold=10)
    network = build_network(tmp_path / "network")
    route = generate_routes(tmp_path / "case", "downstream_slowdown", "validation", 2003, 300)
    results, rewards = [], []
    for cls in (ResidualEnv, IterationEnv):
        extra = (
            {"slowdown_window": (100, 200), "crossing_weight": 30} if cls == IterationEnv else {}
        )
        env = cls(
            config,
            network,
            tmp_path / cls.__name__,
            route_file=route,
            scenario="downstream_slowdown",
            sumo_seed=2003,
            fair_weight=0.25,
            trace=True,
            baseline_parameters=PARAMETERS,
            **extra,
        )
        seen = 0
        try:
            env.reset(seed=2003)
            episode_rewards = []
            while True:
                start = env.time
                _, reward, terminated, truncated, info = env.step(0)
                episode_rewards.append(reward)
                if cls == IterationEnv:
                    seen += info["first_crossings"]
                    expected = (
                        -info["queue_cost"] - 0.25 * info["wait_cost"] - 30 * info["crossing_cost"]
                    )
                    assert reward == pytest.approx(expected)
                    assert info["crossing_cost"] * 160 * (env.time - start) == pytest.approx(
                        info["first_crossings"]
                    )
                if terminated or truncated:
                    break
            results.append(info["metrics"])
            rewards.append(episode_rewards)
            assert info["metrics"]["valid_complete"]
            if cls == IterationEnv:
                waits = [float(t.get("waitingTime")) for t in ET.parse(env.trip_file).getroot()]
                assert seen == sum(w > 10 for w in waits) > 0
                assert env.summary["mean_excess_wait"] == pytest.approx(
                    np.mean(np.maximum(np.array(waits) - 10, 0))
                )
        finally:
            env.close()
    assert all(results[0][k] == results[1][k] for k in results[0])
    assert sum(rewards[1]) < sum(rewards[0])


def test_shifted_slowdown_changes_only_requested_window(tmp_path):
    config = EnvConfig(demand_seconds=120, drain_seconds=300, queue_scale=160)
    network = build_network(tmp_path / "network")
    route = generate_routes(tmp_path / "case", "downstream_slowdown", "validation", 3001, 120)
    env = IterationEnv(
        config,
        network,
        tmp_path / "run",
        route_file=route,
        scenario="downstream_slowdown",
        sumo_seed=3001,
        trace=True,
        slowdown_window=(20, 60),
        baseline_parameters=PARAMETERS,
    )
    try:
        env.reset(seed=3001)
        while True:
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        trace = pd.read_csv(env.episode_dir / "trace.csv")
        assert np.array_equal(trace.slowdown, (trace.second > 20) & (trace.second <= 60))
        assert info["metrics"]["valid_complete"]
    finally:
        env.close()
