"""Load a frozen policy and reproduce an evaluation with optional SUMO GUI."""

import json
from pathlib import Path

from fair_signal.config import EnvConfig, ExperimentConfig
from fair_signal.controllers import TraditionalController
from fair_signal.demand import save_json, sha256
from fair_signal.environment import TrafficEnv
from fair_signal.experiment import case_path, model_path, source_digest


def replay(
    run: Path,
    method: str,
    seed: int,
    scenario: str,
    traffic_seed: int,
    destination: Path,
    gui: bool,
) -> dict:
    saved = json.loads((run / "config.json").read_text())
    if saved["source_sha256"] != source_digest():
        raise ValueError("Run code differs from current code; use its recorded source revision")
    values = saved["config"]
    values["env"] = EnvConfig(**values["env"])
    config = ExperimentConfig(**values)
    config.validate()
    selection = json.loads((run / "selection.json").read_text())[method]
    params = selection["parameters"]
    route = case_path(run, config, "test", scenario, traffic_seed)
    policy = None
    if method.startswith("dqn_"):
        import torch
        from stable_baselines3 import DQN

        torch.set_num_threads(1)
        model = model_path(run, method, selection["candidate"], seed)
        expected = json.loads((model.parent / "training.json").read_text())["model_sha256"]
        if sha256(model) != expected:
            raise ValueError("Model checksum mismatch")
        policy = DQN.load(model, device="cpu")
    controller = TraditionalController(method, params)
    env = TrafficEnv(
        config.env,
        run / "network/intersection.net.xml",
        destination,
        enhanced=method != "dqn_queue",
        fair_weight=params.get("fair_weight", 0.0),
        route_file=route,
        sumo_seed=traffic_seed,
        gui=gui,
        trace=True,
    )
    try:
        observation, _ = env.reset(seed=traffic_seed)
        while True:
            action = (
                int(policy.predict(observation, deterministic=True)[0])
                if policy
                else (controller.predict(env))
            )
            observation, _, terminal, truncated, info = env.step(action)
            if terminal or truncated:
                save_json(
                    destination / "replay.json",
                    {
                        "method": method,
                        "training_seed": seed,
                        "scenario": scenario,
                        "traffic_seed": traffic_seed,
                        "metrics": info["metrics"],
                    },
                )
                return info["metrics"]
    finally:
        env.close()
