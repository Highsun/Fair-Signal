"""Frozen baseline-assisted DQN development study, independent of historical runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from arterial_study import pool_map
from arterial_study import source_digest as historical_digest

from fair_signal.arterial.demand import SCENARIOS
from fair_signal.config import EnvConfig
from fair_signal.demand import save_json, sha256
from fair_signal.environment import write_rows
from fair_signal.residual.environment import ResidualEnv
from fair_signal.runtime import output_directory, project_root

BASELINES = ("actuated", "max_pressure", "aging")
RULES = ("always_extend", "always_urgent")
LEARNED = ("residual_dqn", "fair_residual_dqn")
METHODS = BASELINES + RULES + LEARNED


def source_digest():
    root = project_root()
    digest = hashlib.sha256(historical_digest().encode())
    files = sorted((root / "src/fair_signal/residual").glob("*.py"))
    files += [root / "scripts/residual_study.py"]
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_config(path):
    config = yaml.safe_load(path.read_text())
    env = EnvConfig(**config["env"])
    env.validate()
    groups = [config[k] for k in ("training_seeds", "validation_seeds", "reserved_test_seeds")]
    if any(not s or len(s) != len(set(s)) for s in groups):
        raise ValueError("Seeds must be nonempty and unique")
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError("Seed groups overlap")
    if set(config["methods"]) != set(METHODS) or set(config["scenarios"]) != set(SCENARIOS):
        raise ValueError("Complete predefined methods and scenarios are required")
    if not config["training_steps"] > config["learning_starts"] >= config["batch_size"]:
        raise ValueError("Insufficient update budget")
    if (config["training_steps"] - config["learning_starts"]) % config["train_freq"]:
        raise ValueError("Budget must align with update interval")
    if (
        config["extension_seconds"] < env.decision_seconds
        or config["extension_seconds"] % env.decision_seconds
        or not np.isfinite(config["urgent_threshold"])
        or config["urgent_threshold"] <= 0
        or not np.isfinite(config["fair_weight"])
        or config["fair_weight"] < 0
        or config["guard_threshold"] <= 0
    ):
        raise ValueError("Invalid correction settings")
    return config


def initialize(config, output):
    reference = output_directory(config["reference"])
    old = json.loads((reference / "study.json").read_text())
    if old["source_sha256"] != historical_digest() or old["test_evaluated"]:
        raise ValueError("Historical source or test isolation changed")
    if old["config"]["env"] != config["env"]:
        raise ValueError("Physical experiment settings must match reference")
    if not set(config["validation_seeds"]) <= set(old["config"]["validation_seeds"]):
        raise ValueError("Use existing validation traffic only")
    identity = {
        "config": config,
        "source_sha256": source_digest(),
        "historical_core_sha256": historical_digest(),
        "test_evaluated": False,
        "reference_hashes": {
            name: sha256(reference / name)
            for name in ("study.json", "selection.json", "validation.csv")
        },
        "selection": f"fixed {config['training_steps']}-step endpoint; no new model selection",
    }
    output.mkdir(parents=True, exist_ok=True)
    study = output / "study.json"
    if study.exists() and json.loads(study.read_text()) != identity:
        raise ValueError("Different source/config/reference: choose a new output")
    if not study.exists():
        save_json(study, identity)
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
                    for n in (
                        "torch",
                        "stable-baselines3",
                        "gymnasium",
                        "traci",
                        "numpy",
                    )
                },
            },
        )
    for source in (reference / "network").iterdir():
        if source.is_file():
            destination = output / "network" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(source) != sha256(destination):
                raise ValueError("Copied network changed")
            shutil.copyfile(source, destination)
    network = output / "network/arterial.net.xml"
    manifest = json.loads((output / "network/manifest.json").read_text())
    if sha256(network) != manifest["sha256"]:
        raise ValueError("Frozen network checksum mismatch")
    for scenario in config["scenarios"]:
        for traffic in config["validation_seeds"]:
            relative = Path(f"cases/validation/{scenario}/{traffic}/vehicles.rou.xml")
            source, destination = reference / relative, output / relative
            manifest = json.loads((source.parent / "manifest.json").read_text())
            if sha256(source) != manifest["sha256"]:
                raise ValueError("Frozen traffic changed")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != sha256(source):
                raise ValueError("Copied traffic changed")
            shutil.copyfile(source, destination)
            shutil.copyfile(source.parent / "manifest.json", destination.parent / "manifest.json")
    selection_path = output / "selection.json"
    if selection_path.exists() and sha256(selection_path) != sha256(reference / "selection.json"):
        raise ValueError("Copied baseline selection changed")
    shutil.copyfile(reference / "selection.json", selection_path)


def make_env(output, method, directory, **kwargs):
    config = json.loads((output / "study.json").read_text())["config"]
    selection = json.loads((output / "selection.json").read_text())
    baseline_method = method if method in BASELINES else "actuated"
    return ResidualEnv(
        EnvConfig(**config["env"]),
        output / "network/arterial.net.xml",
        directory,
        baseline_method=baseline_method,
        baseline_parameters=selection[baseline_method]["parameters"],
        extension_seconds=config["extension_seconds"],
        urgent_threshold=config["urgent_threshold"],
        fair_weight=config["fair_weight"] if method == "fair_residual_dqn" else 0,
        guard_threshold=config["guard_threshold"],
        **kwargs,
    )


def train_one(job):
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    output, method, seed = Path(job["output"]), job["method"], job["seed"]
    config = json.loads((output / "study.json").read_text())["config"]
    directory = output / f"models/{method}/seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path, report = directory / "model.zip", directory / "training.json"
    if report.exists():
        saved = json.loads(report.read_text())
        if saved["source_sha256"] != source_digest() or saved["model_sha256"] != sha256(path):
            raise ValueError("Trained model changed")
        return saved

    class Log(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def _on_step(self):
            if (
                not np.isfinite(self.locals["new_obs"]).all()
                or not np.isfinite(self.locals["rewards"]).all()
            ):
                raise RuntimeError("Nonfinite observation/reward")
            info = self.locals["infos"][0]
            for key in ("queue_cost", "wait_cost", "guard_overrides", "correction_changes"):
                self.logger.record_mean(f"traffic/{key}", info[key])
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
                    "correction_effective",
                ):
                    self.logger.record(f"traffic/{key}", float(metrics[key]))
            return True

    env = Monitor(
        make_env(output, method, directory / "episodes"), filename=str(directory / "monitor.csv")
    )
    started = time.perf_counter()
    print(f"Train {method} seed={seed} steps={config['training_steps']}", flush=True)
    try:
        model = DQN(
            "MlpPolicy",
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
        model.logger.dump(model.num_timesteps)
        changed = any(
            not torch.equal(a, b) for a, b in zip(before, model.q_net.parameters(), strict=True)
        )
        if model._n_updates <= 0 or not changed:
            raise RuntimeError("DQN did not update weights")
        model.save(path)
        loaded = DQN.load(path, device="cpu")
        for name in ("q_net", "q_net_target"):
            left, right = getattr(model, name).state_dict(), getattr(loaded, name).state_dict()
            if not all(torch.equal(left[k], right[k]) for k in left):
                raise RuntimeError("Reload changed network weights")
        if any(not torch.isfinite(p).all() for p in model.q_net.parameters()):
            raise RuntimeError("Nonfinite model parameters")
        result = {
            "method": method,
            "training_seed": seed,
            "decision_steps": model.num_timesteps,
            "gradient_updates": model._n_updates,
            "weights_changed": changed,
            "save_reload_weights_match": True,
            "parameter_count": sum(p.numel() for p in model.q_net.parameters()),
            "training_seconds": time.perf_counter() - started,
            "episodes": len(callback.rows),
            "complete_episodes": sum(r["valid_complete"] for r in callback.rows),
            "source_sha256": source_digest(),
            "model_sha256": sha256(path),
        }
        save_json(report, result)
    finally:
        env.close()
    print(f"Finished {method} seed={seed}: {result['training_seconds']:.1f}s", flush=True)
    return result


def case_directory(output, method, seed, scenario, traffic):
    return output / f"evaluation/{method}/seed_{seed}/{scenario}/{traffic}"


def reference_result(config, method, scenario, traffic):
    reference = output_directory(config["reference"])
    selected = json.loads((reference / "selection.json").read_text())[method]["candidate"]
    directory = reference / (
        f"evaluation/validation/{method}/candidate_{selected}/seed_None/{scenario}/{traffic}"
    )
    return json.loads((directory / "result.json").read_text())


def evaluate_one(job):
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    output, method, seed = Path(job["output"]), job["method"], job["seed"]
    scenario, traffic = job["scenario"], job["traffic"]
    config = json.loads((output / "study.json").read_text())["config"]
    directory = case_directory(output, method, seed, scenario, traffic)
    directory.mkdir(parents=True, exist_ok=True)
    route = output / f"cases/validation/{scenario}/{traffic}/vehicles.rou.xml"
    path = output / f"models/{method}/seed_{seed}/model.zip" if method in LEARNED else None
    identity = {
        "source_sha256": source_digest(),
        "config": config,
        "model_sha256": sha256(path) if path else None,
        "route_sha256": sha256(route),
        "network_sha256": sha256(output / "network/arterial.net.xml"),
        "selection_sha256": sha256(output / "selection.json"),
    }
    result_path = directory / "result.json"
    if result_path.exists():
        if json.loads((directory / "identity.json").read_text()) != identity:
            raise ValueError("Evaluation identity changed")
        return json.loads(result_path.read_text())
    save_json(directory / "identity.json", identity)
    model = DQN.load(path, device="cpu") if path else None
    env = make_env(
        output,
        method,
        directory,
        route_file=route,
        scenario=scenario,
        sumo_seed=traffic,
        trace=True,
    )
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
                break
        metrics = info["metrics"]
        if method in BASELINES:
            original = reference_result(config, method, scenario, traffic)
            if any(metrics[k] != value for k, value in original.items() if k in metrics):
                raise RuntimeError("Uncorrected baseline no longer reproduces historical metrics")
        result = {
            "method": method,
            "training_seed": seed,
            "scenario": scenario,
            "traffic_seed": traffic,
            "route_sha256": sha256(route),
            "model_sha256": identity["model_sha256"],
            "baseline_matches_reference": method in BASELINES,
            **metrics,
        }
        save_json(result_path, result)
        return result
    finally:
        env.close()


def evaluation_jobs(config, output):
    return [
        {
            "output": str(output),
            "method": method,
            "seed": seed,
            "scenario": scenario,
            "traffic": traffic,
        }
        for method in METHODS
        for seed in (config["training_seeds"] if method in LEARNED else [None])
        for scenario in config["scenarios"]
        for traffic in config["validation_seeds"]
    ]


def run(config_path, output, workers, stage):
    config = load_config(config_path)
    initialize(config, output)
    started = time.perf_counter()
    if stage in ("all", "train"):
        jobs = [
            {"output": str(output), "method": method, "seed": seed}
            for method in LEARNED
            for seed in config["training_seeds"]
        ]
        pd.DataFrame(pool_map(train_one, jobs, workers)).to_csv(
            output / "training.csv", index=False
        )
    if stage in ("all", "evaluate"):
        rows = pool_map(evaluate_one, evaluation_jobs(config, output), workers)
        pd.DataFrame(rows).to_csv(output / "validation.csv", index=False)
    save_json(
        output / f"execution_{stage}.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "stage": stage,
            "workers": workers,
            "test_evaluated": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/residual.yaml")
    parser.add_argument("--output", default="outputs/residual-50k")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    args = parser.parse_args()
    run(Path(args.config), output_directory(args.output), args.workers, args.stage)
