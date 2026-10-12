import csv
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from fair_signal.arterial.controllers import ArterialController
from fair_signal.arterial.demand import build_network, generate_routes
from fair_signal.arterial.environment import ArterialEnv
from fair_signal.arterial.validation import audit_signal_rows
from fair_signal.config import EnvConfig
from fair_signal.residual.environment import ResidualEnv

pytestmark = pytest.mark.integration
PARAMETERS = {"minimum": 10, "hysteresis": 1.0}


def test_accept_only_reproduces_original_environment_and_reward(tmp_path):
    config = EnvConfig(demand_seconds=120, drain_seconds=300, queue_scale=160)
    network = build_network(tmp_path / "network")
    route = generate_routes(tmp_path / "case", "tidal", "validation", 2001, 120)
    metrics = []
    for index, cls in enumerate((ArterialEnv, ResidualEnv)):
        extra = {"baseline_parameters": PARAMETERS} if cls == ResidualEnv else {}
        env = cls(
            config,
            network,
            tmp_path / str(index),
            route_file=route,
            scenario="tidal",
            sumo_seed=2001,
            fair_weight=1,
            trace=True,
            **extra,
        )
        controller = ArterialController("actuated", PARAMETERS)
        try:
            obs, _ = env.reset(seed=2001)
            if cls == ResidualEnv:
                assert obs.shape == (162,)
            while True:
                action = 0 if cls == ResidualEnv else controller.predict(env)
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)
                assert reward == pytest.approx(-info["queue_cost"] - info["wait_cost"])
                if terminated or truncated:
                    break
            metrics.append(info["metrics"])
        finally:
            env.close()
    assert metrics[0]["valid_complete"]
    assert all(metrics[0][key] == metrics[1][key] for key in metrics[0])
    assert metrics[1]["correction_effective"] == 0


def test_bounded_correction_state_and_physical_accounting(tmp_path):
    config = EnvConfig(demand_seconds=300, drain_seconds=600, queue_scale=160)
    network = build_network(tmp_path / "network")
    route = generate_routes(tmp_path / "case", "downstream_slowdown", "validation", 2003, 300)
    env = ResidualEnv(
        config,
        network,
        tmp_path / "run",
        route_file=route,
        scenario="downstream_slowdown",
        sumo_seed=2003,
        trace=True,
        baseline_parameters=PARAMETERS,
    )
    try:
        check_env(env, warn=True)
        obs, _ = env.reset(seed=2003)
        while True:
            obs, _, terminated, truncated, info = env.step(4)
            assert env.action_space.n == 9 and np.isfinite(obs).all()
            if terminated or truncated:
                break
        metrics = info["metrics"]
        assert metrics["valid_complete"]
        assert metrics["extension_seconds_charged"] > 0
        trace = list(csv.DictReader((env.episode_dir / "trace.csv").open()))
        assert audit_signal_rows(trace, config) > 0
        trips = ET.parse(env.trip_file).getroot().findall("tripinfo")
        assert sum(float(t.get("waitingTime")) for t in trips) == sum(
            int(row["queue"]) for row in trace
        )
        charged = {}
        for row in env.decisions:
            key = row["junction"], row["switches_before"]
            charged[key] = charged.get(key, 0) + row["extension_charged"]
            assert row["extension_used"] <= 10
        assert max(charged.values()) <= 10
    finally:
        env.close()
