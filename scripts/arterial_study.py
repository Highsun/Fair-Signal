"""Frozen two-junction study: separate baseline tuning, training and development validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fair_signal.arterial.controllers import ArterialController, candidates
from fair_signal.arterial.demand import SCENARIOS, build_network, generate_routes
from fair_signal.arterial.environment import ArterialEnv
from fair_signal.config import EnvConfig
from fair_signal.demand import save_json, sha256
from fair_signal.environment import write_rows
from fair_signal.runtime import output_directory, project_root

TRADITIONAL = ("coordinated_fixed", "actuated", "aging", "max_pressure")
LEARNED = ("local_dqn", "joint_dqn", "fair_joint_dqn")


def source_digest():
    root = project_root()
    paths = sorted((root / "src/fair_signal/arterial").glob("*.py"))
    paths += [
        root / f"src/fair_signal/{f}.py"
        for f in ("config", "demand", "environment", "objectives", "runtime", "signals")
    ]
    paths += [root / "scripts/arterial_study.py", root / "uv.lock", root / "pyproject.toml"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    EnvConfig(**c["env"]).validate()
    groups = [
        c[k] for k in ("training_seeds", "tuning_seeds", "validation_seeds", "reserved_test_seeds")
    ]
    if any(not s or len(s) != len(set(s)) for s in groups):
        raise ValueError("Seed groups must be nonempty and unique")
    if any(set(groups[i]) & set(groups[j]) for i in range(4) for j in range(i)):
        raise ValueError("Seed groups must be disjoint")
    if set(c["scenarios"]) != set(SCENARIOS) or set(c["methods"]) != set(TRADITIONAL + LEARNED):
        raise ValueError("This study requires the complete method/scenario matrix")
    if c["training_steps"] <= c["learning_starts"] or c["learning_starts"] < c["batch_size"]:
        raise ValueError("Insufficient replay warmup/training budget")
    if c["fair_weight"] < 0 or c["guard_threshold"] <= 0:
        raise ValueError("Invalid reward/guard")
    return c


def initialize(config, output):
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": config,
        "source_sha256": source_digest(),
        "test_evaluated": False,
        "selection_rule": "complete tuning matrix; minimize mean(mean_wait + 0.5*p95_wait)",
        "all_methods_guarded_during_training_and_evaluation": True,
    }
    path = output / "study.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Output belongs to different code/config; choose a new output")
    if not path.exists():
        save_json(path, identity)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        save_json(
            output / "provenance.json",
            {
                "git_revision": revision,
                "platform": platform.platform(),
                "device": "cpu",
                "torch_threads": 1,
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ("torch", "stable-baselines3", "gymnasium", "traci", "numpy")
                },
            },
        )
    build_network(output / "network")
    for split, seeds in (
        ("tuning", config["tuning_seeds"]),
        ("validation", config["validation_seeds"]),
    ):
        for scenario in config["scenarios"]:
            for seed in seeds:
                directory = output / f"cases/{split}/{scenario}/{seed}"
                path = directory / "vehicles.rou.xml"
                if not path.exists():
                    generate_routes(
                        directory, scenario, split, seed, config["env"]["demand_seconds"]
                    )
                if sha256(path) != json.loads((directory / "manifest.json").read_text())["sha256"]:
                    raise ValueError("Frozen routes changed")


def train_one(job):
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    from fair_signal.arterial.policies import LocalDQNPolicy

    torch.set_num_threads(1)
    output = Path(job["output"])
    config = json.loads((output / "study.json").read_text())["config"]
    method, seed = job["method"], job["seed"]
    directory = output / f"models/{method}/seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.zip"
    report = directory / "training.json"
    if report.exists():
        saved = json.loads(report.read_text())
        if saved["source_sha256"] != source_digest() or saved["model_sha256"] != sha256(path):
            raise ValueError("Trained model changed")
        return saved
    weight = config["fair_weight"] if method == "fair_joint_dqn" else 0.0

    class Log(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def _on_step(self):
            if (
                not np.isfinite(self.locals["new_obs"]).all()
                or not np.isfinite(self.locals["rewards"]).all()
            ):
                raise RuntimeError("Nonfinite learning observation/reward")
            info = self.locals["infos"][0]
            self.logger.record_mean("traffic/queue_cost", info["queue_cost"])
            self.logger.record_mean("traffic/fair_cost", info["wait_cost"])
            self.logger.record_mean("traffic/guard_overrides", info["guard_overrides"])
            if "metrics" in info:
                metrics = info["metrics"]
                self.rows.append({"step": self.num_timesteps, **metrics})
                write_rows(directory / "learning.csv", self.rows)
                for key in (
                    "mean_wait",
                    "p95_wait",
                    "mean_system_time",
                    "mean_stops",
                    "valid_complete",
                ):
                    self.logger.record(f"traffic/{key}", float(metrics[key]))
            return True

    env = Monitor(
        ArterialEnv(
            EnvConfig(**config["env"]),
            output / "network/arterial.net.xml",
            directory / "episodes",
            fair_weight=weight,
            guard_threshold=config["guard_threshold"],
        ),
        filename=str(directory / "monitor.csv"),
    )
    started = time.perf_counter()
    print(f"Train {method} seed={seed} steps={config['training_steps']}", flush=True)
    try:
        model = DQN(
            LocalDQNPolicy if method == "local_dqn" else "MlpPolicy",
            env,
            seed=seed,
            device="cpu",
            verbose=0,
            policy_kwargs={"net_arch": config["hidden"]},
            **{
                k: config[k]
                for k in (
                    "learning_rate",
                    "buffer_size",
                    "learning_starts",
                    "batch_size",
                    "gamma",
                    "train_freq",
                    "target_update_interval",
                    "exploration_fraction",
                )
            },
            exploration_final_eps=0.05,
            gradient_steps=1,
            tensorboard_log=str(directory / "tensorboard"),
        )
        before = [p.detach().clone() for p in model.q_net.parameters()]
        callback = Log()
        model.learn(config["training_steps"], callback=callback)
        # Short smoke runs may not reach SB3's default four-episode log interval.
        model.logger.dump(model.num_timesteps)
        elapsed = time.perf_counter() - started
        changed = any(
            not torch.equal(a, b) for a, b in zip(before, model.q_net.parameters(), strict=True)
        )
        if model._n_updates <= 0 or not changed:
            raise RuntimeError("DQN did not update weights")
        model.save(path)
        loaded = DQN.load(path, device="cpu")
        if not np.array_equal(
            model.predict(model._last_obs, deterministic=True)[0],
            loaded.predict(model._last_obs, deterministic=True)[0],
        ):
            raise RuntimeError("Reload changed predictions")
        result = {
            "method": method,
            "training_seed": seed,
            "decision_steps": model.num_timesteps,
            "gradient_updates": model._n_updates,
            "weights_changed": changed,
            "save_reload_action_matches": True,
            "parameter_count": sum(p.numel() for p in model.q_net.parameters()),
            "training_seconds": elapsed,
            "episodes": len(callback.rows),
            "complete_episodes": sum(r["valid_complete"] for r in callback.rows),
            "source_sha256": source_digest(),
            "model_sha256": sha256(path),
            "fair_weight": weight,
        }
        save_json(report, result)
    finally:
        env.close()
    print(f"Finished {method} seed={seed}: {elapsed:.1f}s", flush=True)
    return result


def evaluate_one(job):
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    output = Path(job["output"])
    config = json.loads((output / "study.json").read_text())["config"]
    method, split, scenario = job["method"], job["split"], job["scenario"]
    seed, traffic, index = job["seed"], job["traffic"], job["candidate"]
    directory = (
        output / f"evaluation/{split}/{method}/candidate_{index}/seed_{seed}/{scenario}/{traffic}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    route = output / f"cases/{split}/{scenario}/{traffic}/vehicles.rou.xml"
    model_file = output / f"models/{method}/seed_{seed}/model.zip" if method in LEARNED else None
    identity = {
        "source_sha256": source_digest(),
        "parameters": job["parameters"],
        "model_sha256": sha256(model_file) if model_file else None,
        "route_sha256": sha256(route),
        "network_sha256": sha256(output / "network/arterial.net.xml"),
        "config": config,
    }
    report = directory / "result.json"
    if report.exists():
        if json.loads((directory / "identity.json").read_text()) != identity:
            raise ValueError("Evaluation inputs changed")
        return json.loads(report.read_text())
    save_json(directory / "identity.json", identity)
    model = DQN.load(model_file, device="cpu") if model_file else None
    controller = None if model else ArterialController(method, job["parameters"])
    env = ArterialEnv(
        EnvConfig(**config["env"]),
        output / "network/arterial.net.xml",
        directory,
        route_file=route,
        scenario=scenario,
        sumo_seed=traffic,
        trace=True,
        fair_weight=config["fair_weight"] if method == "fair_joint_dqn" else 0,
        guard_threshold=config["guard_threshold"],
    )
    try:
        obs, _ = env.reset(seed=traffic)
        while True:
            action = (
                int(model.predict(obs, deterministic=True)[0]) if model else controller.predict(env)
            )
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        result = {
            "method": method,
            "split": split,
            "candidate": index,
            "training_seed": seed,
            "scenario": scenario,
            "traffic_seed": traffic,
            "route_sha256": sha256(route),
            "model_sha256": identity["model_sha256"],
            **info["metrics"],
        }
        save_json(report, result)
        return result
    finally:
        env.close()


def pool_map(function, jobs, workers):
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        rows = []
        for i, row in enumerate(pool.map(function, jobs)):
            rows.append(row)
            if (i + 1) % 15 == 0:
                print(f"{function.__name__}: {i + 1}/{len(jobs)}", flush=True)
    return rows


def evaluation_jobs(config, output, split, selection=None):
    jobs = []
    for method in TRADITIONAL if split == "tuning" else config["methods"]:
        params = (
            candidates(method)
            if split == "tuning"
            else ([selection[method]["parameters"]] if method in TRADITIONAL else [{}])
        )
        for index, parameters in enumerate(params):
            candidate = (
                index
                if split == "tuning"
                else (selection[method]["candidate"] if method in TRADITIONAL else 0)
            )
            for seed in config["training_seeds"] if method in LEARNED else [None]:
                for scenario in config["scenarios"]:
                    for traffic in config[f"{split}_seeds"]:
                        jobs.append(
                            {
                                "output": str(output),
                                "method": method,
                                "split": split,
                                "parameters": parameters,
                                "candidate": candidate,
                                "seed": seed,
                                "scenario": scenario,
                                "traffic": traffic,
                            }
                        )
    return jobs


def run(config_path, output, workers, stage):
    config = load_config(config_path)
    initialize(config, output)
    started = time.perf_counter()
    if stage in ("all", "tune"):
        rows = pool_map(evaluate_one, evaluation_jobs(config, output, "tuning"), workers)
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "tuning.csv", index=False)
        selected = {}
        for method in TRADITIONAL:
            ranking = []
            for index, parameters in enumerate(candidates(method)):
                subset = frame[(frame.method == method) & (frame.candidate == index)]
                if subset.valid_complete.all():
                    score = float((subset.mean_wait + 0.5 * subset.p95_wait).mean())
                    ranking.append((score, index, parameters))
            if not ranking:
                raise RuntimeError(f"No complete tuning candidate for {method}; inspect failures")
            score, index, parameters = min(ranking, key=lambda r: (r[0], r[1]))
            selected[method] = {"candidate": index, "parameters": parameters, "tuning_score": score}
        save_json(output / "selection.json", selected)
    if stage in ("all", "train"):
        jobs = [
            {"output": str(output), "method": m, "seed": seed}
            for m in LEARNED
            for seed in config["training_seeds"]
        ]
        rows = pool_map(train_one, jobs, workers)
        pd.DataFrame(rows).to_csv(output / "training.csv", index=False)
    if stage in ("all", "evaluate"):
        selected = json.loads((output / "selection.json").read_text())
        rows = pool_map(
            evaluate_one, evaluation_jobs(config, output, "validation", selected), workers
        )
        pd.DataFrame(rows).to_csv(output / "validation.csv", index=False)
    save_json(
        output / f"execution_{stage}.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "stage": stage,
            "test_evaluated": False,
            "workers": workers,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/arterial.yaml")
    parser.add_argument("--output", default="outputs/arterial-50k")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--stage", choices=("all", "tune", "train", "evaluate"), default="all")
    args = parser.parse_args()
    run(Path(args.config), output_directory(args.output), args.workers, args.stage)
