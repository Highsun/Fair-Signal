import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml
from stable_baselines3.common.env_checker import check_env

from fair_signal.config import EnvConfig
from fair_signal.demand import build_network, generate_routes
from fair_signal.environment import TrafficEnv
from fair_signal.experiment import run_experiment
from fair_signal.replay import replay
from fair_signal.runtime import output_directory

pytestmark = pytest.mark.integration


def test_environment_reproducibility_and_accounting(tmp_path):
    network = build_network(tmp_path / "network")
    routes = generate_routes(tmp_path / "case", "imbalanced", "test", 2001, 120)
    env = TrafficEnv(
        EnvConfig(demand_seconds=120, drain_seconds=300),
        network,
        tmp_path / "env",
        enhanced=True,
        route_file=routes,
        sumo_seed=2001,
        trace=True,
    )
    try:
        check_env(env, warn=True)
        observations, metrics = [], []
        for _ in range(2):
            observation, _ = env.reset(seed=1)
            values = [observation]
            while True:
                observation, reward, terminal, truncated, info = env.step(0)
                assert np.isfinite(reward) and env.observation_space.contains(observation)
                values.append(observation)
                if terminal or truncated:
                    metrics.append(info["metrics"])
                    break
            observations.append(np.array(values))
        assert np.array_equal(*observations)
        assert metrics[0] == metrics[1]
        result = metrics[0]
        assert result["forced_switches"] > 0
        assert result["planned"] == result["departed"] + result["pending_insertion"]
        assert result["departed"] == result["arrived"] + result["running"]
        assert result["teleport_events"] == result["collision_events"] == 0
        assert result["truncated"] and not result["valid_complete"]
        assert result["unfinished_wait_sum"] > 0
        # Validate the online reward/state counter against SUMO's independent lifetime output.
        for trip in ET.parse(env.trip_file).getroot().findall("tripinfo"):
            assert env.total_waits.get(trip.get("id"), 0) == float(trip.get("waitingTime"))
    finally:
        env.close()


def test_training_selection_test_and_reload(tmp_path):
    config = {
        "label": "integration",
        "env": {"demand_seconds": 120, "drain_seconds": 300},
        "training_steps": 128,
        "learning_starts": 16,
        "buffer_size": 512,
        "batch_size": 16,
        "training_seeds": [11],
        "validation_seeds": [1001],
        "test_seeds": [2001],
        "scenarios": ["imbalanced"],
        "methods": ["fixed", "dqn_queue"],
        "fixed_candidates": [[20, 10, 30, 10]],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    # All experiment artifacts, including test artifacts, stay under code/outputs.
    output = output_directory("outputs/integration-tests")
    import shutil

    if output.exists():
        shutil.rmtree(output)
    run_experiment(path, str(output))
    quality = json.loads((output / "quality.json").read_text())
    assert quality["test_cases"] == 2
    assert not quality["all_cases_complete"] and quality["incomplete_cases"] == 1
    selection = json.loads((output / "selection.json").read_text())
    assert selection["fixed"]["validation_eligible"]
    assert selection["dqn_queue"]["diagnostic_only"]
    saved = json.loads((output / "models/dqn_queue/candidate_0/seed_11/training.json").read_text())
    assert saved["gradient_updates"] > 0 and saved["save_reload_action_matches"]
    destination = output_directory("outputs/integration-replay")
    metrics = replay(output, "dqn_queue", 11, "imbalanced", 2001, destination, False)
    original = output / "evaluation/test/dqn_queue/candidate_0/model_11/imbalanced/2001"
    assert metrics == json.loads((original / "episode_00000/metrics.json").read_text())
    # Re-running completed stages must reuse verified artifacts without training again.
    before = Path(output / "models/dqn_queue/candidate_0/seed_11/model.zip").stat().st_mtime_ns
    run_experiment(path, str(output))
    assert (output / "models/dqn_queue/candidate_0/seed_11/model.zip").stat().st_mtime_ns == before
