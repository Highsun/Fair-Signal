"""Fixed 180s service guard on frozen 150k policies; no additional learning."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from service_guard import guard_action

from fair_signal.config import EnvConfig, ExperimentConfig
from fair_signal.controllers import TraditionalController
from fair_signal.demand import save_json, sha256
from fair_signal.environment import TrafficEnv, write_rows
from fair_signal.experiment import candidates, model_path, source_digest
from fair_signal.objectives import policy_options
from fair_signal.runtime import output_directory, project_root


def case(job: dict) -> dict:
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    root = project_root()
    source = root / "outputs/long-budget-150k/step_150000"
    config_values = json.loads((source / "config.json").read_text())["config"]
    config_values["env"] = EnvConfig(**config_values["env"])
    config = ExperimentConfig(**config_values)
    base, seed = job["base_method"], job["training_seed"]
    scenario, traffic = job["scenario"], job["traffic_seed"]
    parameters = candidates(config, base)[0]
    network = source / "network/intersection.net.xml"
    route = source / f"cases/validation/{scenario}/{traffic}/vehicles.rou.xml"
    model_file = model_path(source, base, 0, seed) if seed is not None else None
    destination = Path(job["output"]) / f"evaluation/{base}_guard/model_{seed}/{scenario}/{traffic}"
    identity = {
        "base_method": base,
        "training_seed": seed,
        "route_sha256": sha256(route),
        "network_sha256": sha256(network),
        "model_sha256": sha256(model_file) if model_file else None,
        "source_sha256": source_digest(),
        "runner_sha256": sha256(Path(__file__)),
        "guard_sha256": sha256(root / "scripts/service_guard.py"),
        "threshold": 180,
        "parameters": parameters,
        "environment": config.env.__dict__,
    }
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / "result.json"
    if report.exists():
        if json.loads((destination / "identity.json").read_text()) != identity:
            raise ValueError("Guard evaluation identity changed")
        return json.loads(report.read_text())
    save_json(destination / "identity.json", identity)
    policy = DQN.load(model_file, device="cpu") if model_file else None
    controller = TraditionalController(base, parameters) if policy is None else None
    env = TrafficEnv(
        config.env,
        network,
        destination,
        **policy_options(base, parameters),
        route_file=route,
        sumo_seed=traffic,
        trace=True,
    )
    actions = []
    try:
        observation, _ = env.reset(seed=traffic)
        while True:
            proposed = (
                int(policy.predict(observation, deterministic=True)[0])
                if policy
                else controller.predict(env)
            )
            decision = guard_action(env.signal, env.counts, env.request_ages, proposed)
            actions.append(
                {
                    "second": env.time,
                    "proposed": proposed,
                    "executed": decision.action,
                    "overridden": decision.overridden,
                    "oldest_service_age": decision.oldest_age,
                }
            )
            observation, _, terminal, truncated, info = env.step(decision.action)
            if terminal or truncated:
                metrics = info["metrics"]
                break
    finally:
        env.close()
    action_frame = pd.DataFrame(actions)
    demand = action_frame[action_frame.second < config.env.demand_seconds]
    result = {
        "method": f"{base}_guard",
        "base_method": base,
        "training_seed": seed,
        "candidate": 0,
        "scenario": scenario,
        "traffic_seed": traffic,
        "split": "validation",
        "route_sha256": identity["route_sha256"],
        "model_sha256": identity["model_sha256"],
        "guard_overrides": int(action_frame.overridden.sum()),
        "guard_override_fraction": float(action_frame.overridden.mean()),
        "guard_overrides_demand": int(demand.overridden.sum()),
        "guard_override_fraction_demand": float(demand.overridden.mean()),
        **metrics,
    }
    write_rows(destination / "guard_actions.csv", actions)
    save_json(report, result)
    return result


def run(output: Path, workers: int) -> None:
    root = project_root()
    source = root / "outputs/long-budget-150k/step_150000"
    saved = json.loads((source / "config.json").read_text())
    if saved["source_sha256"] != source_digest():
        raise ValueError("Frozen base source changed")
    config = saved["config"]
    model_hashes = {}
    for method in ("dqn_queue", "dqn_fair"):
        for seed in config["training_seeds"]:
            path = model_path(source, method, 0, seed)
            record = json.loads(path.with_name("training.json").read_text())
            if record["model_sha256"] != sha256(path) or record["decision_steps"] != 150000:
                raise ValueError("Frozen base model changed")
            model_hashes[f"{method}/{seed}"] = sha256(path)
    identity = {
        "purpose": "posthoc_development_hybrid_validation",
        "threshold": 180,
        "runner_sha256": sha256(Path(__file__)),
        "guard_sha256": sha256(root / "scripts/service_guard.py"),
        "source_sha256": source_digest(),
        "base_config": saved,
        "models": model_hashes,
        "base_validation_sha256": sha256(source / "validation.csv"),
        "training_performed": False,
        "test_evaluated": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "study.json").exists() and json.loads(
        (output / "study.json").read_text()
    ) != identity:
        raise ValueError("Guard study configuration changed")
    save_json(output / "study.json", identity)
    jobs = [
        {
            "base_method": method,
            "training_seed": seed,
            "scenario": scenario,
            "traffic_seed": traffic,
            "output": str(output),
        }
        for method in ("dqn_queue", "dqn_fair", "actuated")
        for seed in (config["training_seeds"] if method.startswith("dqn_") else [None])
        for scenario in config["scenarios"]
        for traffic in config["validation_seeds"]
    ]
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for index, result in enumerate(pool.map(case, jobs)):
            rows.append(result)
            if (index + 1) % 15 == 0:
                print(f"Guard validation: {index + 1}/{len(jobs)}", flush=True)
    pd.DataFrame(rows).to_csv(output / "validation.csv", index=False)
    save_json(
        output / "execution.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "validation_cases": len(rows),
            "training_performed": False,
            "test_evaluated": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/service-guard-150k")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    run(output_directory(args.output), args.workers)
