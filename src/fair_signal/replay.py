"""Load a frozen policy and reproduce an evaluation with optional SUMO GUI."""

import json
from dataclasses import replace
from pathlib import Path

from fair_signal.config import EnvConfig, ExperimentConfig
from fair_signal.controllers import TraditionalController
from fair_signal.demand import generate_routes, save_json, sha256
from fair_signal.environment import TrafficEnv
from fair_signal.experiment import case_path, model_path, source_digest
from fair_signal.objectives import policy_options
from fair_signal.runtime import project_root


def compatible_source(saved_digest: str) -> bool:
    """Allow only current code or an explicitly verified pair of exact code hashes.

    Whole-source checks otherwise reject old models even for GUI-only changes.
    A later source edit invalidates the pair automatically; no general bypass flag.
    Training/evaluation resumption continues to require an exact whole-source match.
    """
    current = source_digest()
    if current == saved_digest:
        return True
    path = project_root() / "configs/replay-compatibility.json"
    if not path.is_file():
        return False
    pairs = json.loads(path.read_text(encoding="utf-8"))["verified_pairs"]
    return any(
        pair["viewer_sha256"] == current and pair["source_sha256"] == saved_digest for pair in pairs
    )


def replay(
    run: Path,
    method: str,
    seed: int,
    scenario: str,
    traffic_seed: int,
    destination: Path,
    gui: bool,
    gui_delay: int = 200,
    start_paused: bool = True,
    demand_seconds: int | None = None,
) -> dict:
    saved = json.loads((run / "config.json").read_text())
    if not compatible_source(saved["source_sha256"]):
        raise ValueError("Run code differs from current code; use its recorded source revision")
    values = saved["config"]
    values["env"] = EnvConfig(**values["env"])
    config = ExperimentConfig(**values)
    config.validate()
    selection = json.loads((run / "selection.json").read_text())[method]
    params = selection["parameters"]
    if demand_seconds is not None:
        if type(demand_seconds) is not int or not 60 <= demand_seconds <= 86400:
            raise ValueError("Demonstration demand must be between 60 and 86400 seconds")
        config.env = replace(config.env, demand_seconds=demand_seconds)
        config.env.validate()
        route = generate_routes(
            destination / "demo-demand", scenario, "test", traffic_seed, demand_seconds
        )
    else:
        route = case_path(run, config, "test", scenario, traffic_seed)
    network = run / "network/intersection.net.xml"
    network_manifest = json.loads((run / "network/manifest.json").read_text())
    if sha256(network) != network_manifest["sha256"]:
        raise ValueError("Network checksum mismatch")
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
        network,
        destination,
        **policy_options(method, params),
        route_file=route,
        sumo_seed=traffic_seed,
        gui=gui,
        trace=True,
        gui_delay=gui_delay,
        start_paused=start_paused,
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
                        "purpose": "demonstration"
                        if demand_seconds is not None
                        else "reproduction",
                        "demand_seconds": config.env.demand_seconds,
                        "gui_delay_ms": gui_delay if gui else None,
                        "source_sha256": saved["source_sha256"],
                        "viewer_sha256": source_digest(),
                        "route_sha256": sha256(route),
                        "metrics": info["metrics"],
                    },
                )
                return info["metrics"]
    finally:
        env.close()
