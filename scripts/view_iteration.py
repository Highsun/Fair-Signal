"""Replay an already evaluated iteration case with frozen inputs and exact metrics."""

import argparse
import json

from iteration_study import BASELINES, METHODS, cases, make_env, model_path, source_digest

from fair_signal.demand import sha256
from fair_signal.runtime import output_directory


def replay(output, method, seed, split, case_id, delay, headless=False):
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    study = json.loads((output / "study.json").read_text())
    config = study["config"]
    if study["source_sha256"] != source_digest():
        raise ValueError("Study source changed")
    case = next((c for c in cases(config, split) if c["id"] == case_id), None)
    if case is None or (method not in BASELINES and seed not in config["training_seeds"]):
        raise ValueError("Unknown evaluated case/model seed")
    model_seed = None if method in BASELINES else seed
    directory = output / f"evaluation/{split}/{method}/seed_{model_seed}/{case_id}"
    original = json.loads((directory / "result.json").read_text())
    identity = json.loads((directory / "identity.json").read_text())
    route = output / f"cases/{split}/{case_id}/vehicles.rou.xml"
    path = model_path(output, method, seed) if method not in BASELINES else None
    if identity != dict(
        source_sha256=source_digest(),
        config=config,
        case=case,
        model_sha256=sha256(path) if path else None,
        route_sha256=sha256(route),
        network_sha256=sha256(output / "network/arterial.net.xml"),
        selection_sha256=sha256(output / "selection.json"),
    ):
        raise ValueError("Frozen replay inputs changed")
    model = DQN.load(path, device="cpu") if path else None
    env = make_env(
        output,
        method,
        output / f"viewer/{split}/{method}/{model_seed}/{case_id}",
        route_file=route,
        scenario=case["scenario"],
        sumo_seed=case["traffic"],
        slowdown_window=case["window"],
        trace=True,
        gui=not headless,
        gui_delay=delay,
        start_paused=True,
    )
    if not headless:
        print("SUMO starts paused: Play / Stop / Single Step; Delay controls speed.", flush=True)
    try:
        obs, _ = env.reset(seed=case["traffic"])
        while True:
            action = int(model.predict(obs, deterministic=True)[0]) if model else 0
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                metrics = info["metrics"]
                if any(original[k] != value for k, value in metrics.items()):
                    raise RuntimeError("Replay differs from frozen evaluation")
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
                return metrics
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/iteration-50k")
    parser.add_argument("--method", choices=METHODS, default="residual_dqn")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--case", default="downstream_slowdown_rate1_3001")
    parser.add_argument("--delay", type=int, default=200)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    replay(
        output_directory(args.output),
        args.method,
        args.seed,
        args.split,
        args.case,
        args.delay,
        args.headless,
    )
