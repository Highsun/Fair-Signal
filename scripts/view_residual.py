"""Replay a frozen baseline-assisted validation case in SUMO GUI or headlessly."""

from __future__ import annotations

import argparse
import json

from residual_study import LEARNED, METHODS, case_directory, make_env, source_digest

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
    if method in LEARNED and seed not in config["training_seeds"]:
        raise ValueError("Unknown training seed")
    model_seed = seed if method in LEARNED else None
    directory = case_directory(output, method, model_seed, scenario, traffic)
    original = json.loads((directory / "result.json").read_text())
    identity = json.loads((directory / "identity.json").read_text())
    route = output / f"cases/validation/{scenario}/{traffic}/vehicles.rou.xml"
    model_file = output / f"models/{method}/seed_{seed}/model.zip"
    if identity != {
        "source_sha256": source_digest(),
        "config": config,
        "model_sha256": sha256(model_file) if method in LEARNED else None,
        "route_sha256": sha256(route),
        "network_sha256": sha256(output / "network/arterial.net.xml"),
        "selection_sha256": sha256(output / "selection.json"),
    }:
        raise ValueError("Frozen replay inputs changed")
    model = DQN.load(model_file, device="cpu") if method in LEARNED else None
    env = make_env(
        output,
        method,
        output / f"viewer/{method}/{seed}/{scenario}/{traffic}",
        route_file=route,
        scenario=scenario,
        sumo_seed=traffic,
        trace=True,
        gui=not headless,
        gui_delay=delay,
        start_paused=True,
    )
    if not headless:
        print("SUMO starts paused: Play / Stop / Single Step; Delay controls speed.", flush=True)
    try:
        obs, _ = env.reset(seed=traffic)
        while True:
            action = (
                int(model.predict(obs, deterministic=True)[0])
                if model
                else {
                    "always_extend": 4,
                    "always_urgent": 8,
                }.get(method, 0)
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
    parser.add_argument("--output", default="outputs/residual-50k")
    parser.add_argument("--method", choices=METHODS, default="residual_dqn")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--scenario", choices=("steady", "tidal", "downstream_slowdown"), default="tidal"
    )
    parser.add_argument("--traffic", type=int, default=2001)
    parser.add_argument("--delay", type=int, default=200)
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
