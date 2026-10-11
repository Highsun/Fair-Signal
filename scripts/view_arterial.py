"""Replay a frozen two-junction validation case in SUMO GUI, initially paused."""

from __future__ import annotations

import argparse
import json

from arterial_study import LEARNED, TRADITIONAL, source_digest

from fair_signal.arterial.controllers import ArterialController
from fair_signal.arterial.environment import ArterialEnv
from fair_signal.config import EnvConfig
from fair_signal.demand import sha256
from fair_signal.runtime import output_directory


def replay(output, method, seed, scenario, traffic, delay, headless=False):
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    saved = json.loads((output / "study.json").read_text())
    config = saved["config"]
    if saved["source_sha256"] != source_digest():
        raise ValueError("Frozen study source changed")
    if scenario not in config["scenarios"] or traffic not in config["validation_seeds"]:
        raise ValueError("Replay only an existing development validation case")
    selected = json.loads((output / "selection.json").read_text())
    index = selected[method]["candidate"] if method in TRADITIONAL else 0
    model_seed = seed if method in LEARNED else None
    case = output / f"evaluation/validation/{method}/candidate_{index}"
    case = case / f"seed_{model_seed}/{scenario}/{traffic}"
    original = json.loads((case / "result.json").read_text())
    route = output / f"cases/validation/{scenario}/{traffic}/vehicles.rou.xml"
    if sha256(route) != original["route_sha256"]:
        raise ValueError("Replay route changed")
    model = controller = None
    if method in LEARNED:
        path = output / f"models/{method}/seed_{seed}/model.zip"
        if sha256(path) != original["model_sha256"]:
            raise ValueError("Replay model changed")
        model = DQN.load(path, device="cpu")
    else:
        controller = ArterialController(method, selected[method]["parameters"])
    env = ArterialEnv(
        EnvConfig(**config["env"]),
        output / "network/arterial.net.xml",
        output / f"viewer/{method}/{seed}/{scenario}/{traffic}",
        route_file=route,
        scenario=scenario,
        sumo_seed=traffic,
        trace=True,
        fair_weight=config["fair_weight"] if method == "fair_joint_dqn" else 0,
        guard_threshold=config["guard_threshold"],
        gui=not headless,
        gui_delay=delay,
        start_paused=True,
    )
    print(
        "SUMO starts paused: use Play / Stop / Single Step. Delay controls display speed.",
        flush=True,
    )
    try:
        obs, _ = env.reset(seed=traffic)
        while True:
            action = (
                int(model.predict(obs, deterministic=True)[0]) if model else controller.predict(env)
            )
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                metrics = info["metrics"]
                if any(original[k] != value for k, value in metrics.items()):
                    raise RuntimeError("Replay metrics differ from frozen validation")
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
                return metrics
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/arterial-50k")
    parser.add_argument("--method", choices=TRADITIONAL + LEARNED, default="joint_dqn")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--scenario", choices=("steady", "tidal", "downstream_slowdown"), default="tidal"
    )
    parser.add_argument("--traffic", type=int, default=2001)
    parser.add_argument("--delay", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    replay(
        output_directory(args.output),
        args.method,
        args.seed,
        args.scenario,
        args.traffic,
        args.delay,
        args.headless,
    )
