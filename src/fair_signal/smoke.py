"""Bounded integration check; no fairness claims or benchmark results."""

from __future__ import annotations

import importlib.metadata
import json
import time
import xml.etree.ElementTree as ET

from fair_signal.runtime import configure_sumo, output_directory


def run_smoke(steps: int, seed: int, output: str) -> dict:
    """Exercise simulation, gradients and model serialization with actual vehicles."""
    configure_sumo()
    output_dir = output_directory(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imports are intentionally after SUMO_HOME configuration (SUMO-RL requires it).
    import numpy as np
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from sumo_rl import SumoEnvironment

    torch.set_num_threads(1)
    package = importlib.metadata.distribution("sumo-rl")
    network_dir = package.locate_file("sumo_rl/nets/2way-single-intersection")
    network_file = network_dir / "single-intersection.net.xml"
    reference_routes = network_dir / "single-intersection-vhvh.rou.xml"
    if not network_file.is_file() or not reference_routes.is_file():
        raise FileNotFoundError("The installed SUMO-RL wheel is missing its example network.")

    # Read route definitions only; do not run the upstream 100000-second demand.
    routes = ET.Element("routes")
    for route in ET.parse(reference_routes).getroot().findall("route"):
        routes.append(ET.Element("route", route.attrib))
    for index, route_id in enumerate(("route_ns", "route_sn", "route_we", "route_ew")):
        ET.SubElement(
            routes,
            "flow",
            {
                "id": f"smoke_{index}",
                "route": route_id,
                "begin": str(index),
                "end": str((steps + 20) * 5),
                "period": "10",
                "departLane": "best",
                "departSpeed": "max",
            },
        )
    route_file = output_dir / "smoke.rou.xml"
    ET.ElementTree(routes).write(route_file, encoding="utf-8", xml_declaration=True)

    class ValidateTraffic(BaseCallback):
        def __init__(self):
            super().__init__()
            self.max_running_vehicles = 0

        def _on_step(self) -> bool:
            rewards = self.locals["rewards"]
            observations = self.locals["new_obs"]
            if not np.isfinite(rewards).all() or not np.isfinite(observations).all():
                raise RuntimeError("Non-finite reward or observation during smoke check")
            self.max_running_vehicles = max(
                self.max_running_vehicles, environment.unwrapped.sumo.vehicle.getIDCount()
            )
            return True

    environment = SumoEnvironment(
        net_file=str(network_file),
        route_file=str(route_file),
        single_agent=True,
        use_gui=False,
        num_seconds=(steps + 20) * 5,
        delta_time=5,
        yellow_time=2,
        min_green=5,
        sumo_seed=seed,
        reward_fn="queue",
        time_to_teleport=-1,
        max_depart_delay=-1,
        additional_sumo_cmd="--no-step-log true --duration-log.disable true",
    )
    environment = Monitor(environment)
    try:
        model = DQN(
            "MlpPolicy",
            environment,
            device="cpu",
            seed=seed,
            policy_kwargs={"net_arch": [128, 128]},
            buffer_size=2048,
            learning_starts=16,
            batch_size=32,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=64,
            verbose=0,
        )
        initial_parameters = [parameter.detach().clone() for parameter in model.q_net.parameters()]
        callback = ValidateTraffic()
        started = time.perf_counter()
        model.learn(total_timesteps=steps, callback=callback)
        elapsed = time.perf_counter() - started
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(initial_parameters, model.q_net.parameters(), strict=True)
        )
        if model._n_updates == 0 or not changed:
            raise RuntimeError("DQN did not update its parameters")
        if callback.max_running_vehicles == 0:
            raise RuntimeError("No running vehicles observed; simulation check is inconclusive")
        model_file = output_dir / "dqn-smoke.zip"
        model.save(model_file)
        reloaded = DQN.load(model_file, device="cpu")
        probe = model._last_obs
        if not np.array_equal(
            model.predict(probe, deterministic=True)[0],
            reloaded.predict(probe, deterministic=True)[0],
        ):
            raise RuntimeError("Reloaded model does not reproduce its deterministic action")
        report = {
            "purpose": "environment integration only; not policy performance evaluation",
            "seed": seed,
            "device": "cpu",
            "decision_steps": model.num_timesteps,
            "simulated_seconds": model.num_timesteps * 5,
            "learn_seconds": round(elapsed, 3),
            "decision_steps_per_second": round(model.num_timesteps / elapsed, 2),
            "gradient_updates": model._n_updates,
            "network_parameters": sum(parameter.numel() for parameter in model.q_net.parameters()),
            "observation_shape": list(environment.observation_space.shape),
            "action_count": int(environment.action_space.n),
            "max_running_vehicles": callback.max_running_vehicles,
            "finite_observations_and_rewards": True,
            "parameters_changed": changed,
            "save_reload_action_matches": True,
            "network_source": "installed sumo-rl 1.4.5 example network (MIT)",
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        environment.close()
