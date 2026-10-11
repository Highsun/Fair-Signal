import numpy as np
import pytest

from fair_signal.config import EnvConfig, ExperimentConfig
from fair_signal.environment import TrafficEnv
from fair_signal.objectives import policy_options, quadratic_wait_increment
from fair_signal.signals import Signal


def test_quadratic_reward_telescopes_and_covers_hidden_long_wait():
    assert sum(quadratic_wait_increment([w], 90, 80) for w in range(1, 121)) == pytest.approx(
        120**2 / (2 * 90 * 80)
    )
    assert quadratic_wait_increment([], 90, 80) == 0
    # A head with 10s and a vehicle behind with 120s: both must contribute.
    assert quadratic_wait_increment([10, 120], 90, 80) == pytest.approx(
        quadratic_wait_increment([10], 90, 80) + quadratic_wait_increment([120], 90, 80)
    )
    assert quadratic_wait_increment([1], 90, 80) > 0


def observed(mode, phase=0, wait=180, revised=True):
    env = TrafficEnv.__new__(TrafficEnv)
    env.config = EnvConfig()
    env.enhanced = True
    env.observation_mode = "service" if revised else "legacy"
    env.signal = Signal(env.config, phase=phase, target=2, mode=mode)
    env.counts = env.queues = np.zeros(8)
    env.capacity = np.ones(8)
    env.head_waits = env.max_waits = np.full(8, wait, dtype=float)
    env.request_ages = np.zeros(8)
    return env._observation()


def test_revised_observation_removes_signal_aliases_and_wait_clipping():
    assert np.array_equal(observed("yellow", revised=False), observed("all_red", revised=False))
    assert np.array_equal(observed("yellow", revised=False), observed("yellow", 1, revised=False))
    assert not np.array_equal(observed("yellow"), observed("all_red"))
    assert not np.array_equal(observed("yellow"), observed("yellow", 1))
    assert np.array_equal(
        observed("green", wait=180, revised=False), observed("green", wait=900, revised=False)
    )
    assert not np.array_equal(observed("green", wait=180), observed("green", wait=900))
    values = observed("green", wait=900)
    assert values.shape == (44,) and np.isfinite(values).all()
    assert (values >= 0).all() and (values <= 1).all()


def test_variant_mapping_keeps_legacy_algorithms_and_default_matrix():
    assert policy_options("dqn_fair", {"fair_weight": 1}) == {
        "enhanced": True,
        "fair_weight": 1,
        "reward_mode": "head_threshold",
        "observation_mode": "legacy",
    }
    assert policy_options("dqn_dense", {})["observation_mode"] == "legacy"
    assert policy_options("dqn_dense_state", {})["reward_mode"] == "vehicle_quadratic"
    options = policy_options("dqn_fair_state", {"fair_weight": 1.0})
    assert options["observation_mode"] == "service"
    assert options["reward_mode"] == "head_threshold"
    assert options["fair_weight"] == 1.0
    from fair_signal.experiment import candidates

    assert candidates(ExperimentConfig(fair_weights=[1.0]), "dqn_fair_state") == [
        {"fair_weight": 1.0}
    ]
    assert len(ExperimentConfig().methods) == 6
