import xml.etree.ElementTree as ET

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from fair_signal.arterial.controllers import ArterialController
from fair_signal.arterial.demand import build_network, generate_routes
from fair_signal.arterial.environment import ArterialEnv
from fair_signal.arterial.validation import audit_signal_rows
from fair_signal.config import EnvConfig

pytestmark = pytest.mark.integration


def test_two_junction_physics_rewards_and_lifetime_accounting(tmp_path):
    config = EnvConfig(demand_seconds=120, drain_seconds=900, queue_scale=160)
    net = build_network(tmp_path / "network")
    route = generate_routes(tmp_path / "case", "downstream_slowdown", "validation", 2001, 120)
    env = ArterialEnv(
        config,
        net,
        tmp_path / "run",
        route_file=route,
        scenario="downstream_slowdown",
        sumo_seed=2001,
        trace=True,
        fair_weight=1,
    )
    controller = ArterialController("actuated", {"minimum": 10, "hysteresis": 1.0})
    try:
        check_env(env, warn=True)
        results = []
        for _ in range(2):
            obs, _ = env.reset(seed=11)
            queue_integral = fair_integral = 0.0
            while True:
                before = env.time
                obs, reward, terminal, truncated, info = env.step(controller.predict(env))
                dt = env.time - before
                queue_integral += info["queue_cost"] * dt
                fair_integral += info["wait_cost"] * dt
                assert reward == pytest.approx(-info["queue_cost"] - info["wait_cost"])
                assert env.observation_space.contains(obs)
                if terminal or truncated:
                    break
            assert audit_signal_rows(env.trace_rows, config) > 0
            assert info["metrics"]["valid_complete"]
            assert env.planned == env.departed == env.arrived
            assert fair_integral == pytest.approx(
                sum(max(w - 90, 0) for w in env.total_waits.values()) / 160
            )
            assert queue_integral == pytest.approx(
                (sum(env.total_waits.values()) + sum(r["pending"] for r in env.trace_rows)) / 160
            )
            for trip in ET.parse(env.trip_file).getroot().findall("tripinfo"):
                assert env.total_waits.get(trip.get("id"), 0) == float(trip.get("waitingTime"))
            assert any(row["slowdown"] for row in env.trace_rows)
            assert not env.slowdown_active
            results.append((np.array([r["queue"] for r in env.trace_rows]), info["metrics"]))
        assert np.array_equal(results[0][0], results[1][0])
        assert results[0][1] == results[1][1]
    finally:
        env.close()


def test_full_load_low_speed_departure_regression(tmp_path):
    config = EnvConfig(demand_seconds=1800, drain_seconds=900, queue_scale=160)
    network = build_network(tmp_path / "network")
    route = generate_routes(tmp_path / "case", "tidal", "tuning", 902, 1800)
    env = ArterialEnv(
        config, network, tmp_path / "run", route_file=route, scenario="tidal", sumo_seed=902
    )
    controller = ArterialController(
        "coordinated_fixed", {"durations": [20, 10, 35, 10], "offset": 0}
    )
    try:
        env.reset(seed=902)
        while True:
            _, _, terminated, truncated, info = env.step(controller.predict(env))
            if terminated or truncated:
                break
        assert info["metrics"]["valid_complete"]
        # This car was inserted at 0.04 m/s; its insertion state is not one spent waiting second.
        assert env.total_waits["e_000583"] == 128
    finally:
        env.close()
