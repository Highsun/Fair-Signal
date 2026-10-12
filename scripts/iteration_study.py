"""Predeclared robustness, fairness and matched information study; test opens last."""

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
from residual_study import source_digest as residual_digest

from fair_signal.arterial.demand import SCENARIOS
from fair_signal.config import EnvConfig
from fair_signal.demand import save_json, sha256
from fair_signal.environment import write_rows
from fair_signal.iteration.demand import generate_scaled_routes
from fair_signal.iteration.environment import IterationEnv
from fair_signal.iteration.policies import MatchedDQNPolicy
from fair_signal.runtime import output_directory, project_root

BASELINES = ("actuated", "max_pressure", "aging")
OLD = ("residual_dqn", "fair_residual_dqn")
NEW = ("excess_025", "crossing_30", "crossing_90", "matched_full", "matched_local")
METHODS = BASELINES + OLD + NEW


def source_digest():
    root = project_root()
    digest = hashlib.sha256(residual_digest().encode())
    for path in [
        *sorted((root / "src/fair_signal/iteration").glob("*.py")),
        Path(__file__).resolve(),
    ]:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_config(path):
    config = yaml.safe_load(path.read_text())
    EnvConfig(**config["env"]).validate()
    groups = [config[k] for k in ("training_seeds", "development_seeds", "test_seeds")]
    if any(not x or len(set(x)) != len(x) for x in groups):
        raise ValueError("Empty/duplicate seeds")
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError("Overlapping seed groups")
    if set(config["variants"]) != set(NEW):
        raise ValueError("Predeclared variants required")
    if config["training_steps"] <= config["learning_starts"]:
        raise ValueError("Insufficient training budget")
    if (config["training_steps"] - config["learning_starts"]) % config["train_freq"]:
        raise ValueError("Budget must align with gradient updates")
    return config


def initialize(config, output):
    reference = output_directory(config["reference"])
    prior = json.loads((reference / "study.json").read_text())
    if prior["source_sha256"] != residual_digest() or prior["test_evaluated"]:
        raise ValueError("Frozen residual source changed or reference test was used")
    if config["env"] != prior["config"]["env"]:
        raise ValueError("Physical settings changed")
    reserved = set(prior["config"]["reserved_test_seeds"])
    if not set(config["test_seeds"]) <= reserved or set(config["development_seeds"]) & reserved:
        raise ValueError("Reserved test isolation violated")
    model_hashes = {}
    for method in OLD:
        for seed in config["training_seeds"]:
            directory = reference / f"models/{method}/seed_{seed}"
            report = json.loads((directory / "training.json").read_text())
            digest = sha256(directory / "model.zip")
            if report["source_sha256"] != residual_digest() or report["model_sha256"] != digest:
                raise ValueError("Historical model identity changed")
            model_hashes[f"{method}/{seed}"] = digest
    identity = {
        "reference_model_hashes": model_hashes,
        "config": config,
        "source_sha256": source_digest(),
        "residual_source_sha256": residual_digest(),
        "reference_study_sha256": sha256(reference / "study.json"),
        "reference_selection_sha256": sha256(reference / "selection.json"),
        "endpoint": "fixed; no checkpoint or seed selection",
    }
    output.mkdir(parents=True, exist_ok=True)
    study = output / "study.json"
    if study.exists() and json.loads(study.read_text()) != identity:
        raise ValueError("Source/config/reference changed: use another output")
    if not study.exists():
        save_json(study, identity)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
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
    for source in (reference / "network").iterdir():
        if source.is_file():
            dest = output / "network" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha256(dest) != sha256(source):
                raise ValueError("Network copy changed")
            shutil.copyfile(source, dest)
    selection = output / "selection.json"
    if selection.exists() and sha256(selection) != sha256(reference / "selection.json"):
        raise ValueError("Baseline parameters changed")
    shutil.copyfile(reference / "selection.json", selection)


def make_env(output, method, directory, **kwargs):
    config = json.loads((output / "study.json").read_text())["config"]
    selection = json.loads((output / "selection.json").read_text())
    variant = config["variants"].get(method, {})
    base = method if method in BASELINES else "actuated"
    return IterationEnv(
        EnvConfig(**config["env"]),
        output / "network/arterial.net.xml",
        directory,
        baseline_method=base,
        baseline_parameters=selection[base]["parameters"],
        extension_seconds=config["extension_seconds"],
        urgent_threshold=config["urgent_threshold"],
        guard_threshold=config["guard_threshold"],
        fair_weight=variant.get("fair_weight", 1.0 if method == "fair_residual_dqn" else 0.0),
        crossing_weight=variant.get("crossing_weight", 0.0),
        **kwargs,
    )


def model_path(output, method, seed):
    config = json.loads((output / "study.json").read_text())["config"]
    root = output_directory(config["reference"]) if method in OLD else output
    return root / f"models/{method}/seed_{seed}/model.zip"


def cases(config, split):
    if split == "test":
        return [
            dict(
                id=f"{s}_{t}",
                scenario=s,
                traffic=t,
                multiplier=1.0,
                window=[600, 1200],
                group="test",
            )
            for s in SCENARIOS
            for t in config["test_seeds"]
        ]
    ordinary = [
        dict(
            id=f"{s}_rate{r:g}_{t}",
            scenario=s,
            traffic=t,
            multiplier=r,
            window=[600, 1200],
            group=f"rate_{r:g}",
        )
        for r in config["demand_multipliers"]
        for s in SCENARIOS
        for t in config["development_seeds"]
    ]
    shifted = [
        dict(
            id=f"downstream_slowdown_window{w[0]}_{t}",
            scenario="downstream_slowdown",
            traffic=t,
            multiplier=1.0,
            window=w,
            group=f"window_{w[0]}",
        )
        for w in config["shifted_windows"]
        for t in config["development_seeds"]
    ]
    return ordinary + shifted


def prepare_cases(config, output, split):
    records = cases(config, split)
    for case in records:
        directory = output / f"cases/{split}/{case['id']}"
        route = directory / "vehicles.rou.xml"
        if route.exists():
            manifest = json.loads((directory / "manifest.json").read_text())
            expected = {
                "scenario": case["scenario"],
                "split": "validation" if split == "development" else "test",
                "seed": case["traffic"],
                "demand_multiplier": case["multiplier"],
                "demand_seconds": config["env"]["demand_seconds"],
            }
            if (
                any(manifest[k] != v for k, v in expected.items())
                or sha256(route) != manifest["sha256"]
            ):
                raise ValueError("Case manifest/route changed")
        else:
            generate_scaled_routes(
                directory,
                case["scenario"],
                "validation" if split == "development" else "test",
                case["traffic"],
                config["env"]["demand_seconds"],
                case["multiplier"],
            )
    save_json(output / f"cases_{split}.json", records)
    return records


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
            for key in (
                "queue_cost",
                "wait_cost",
                "crossing_cost",
                "first_crossings",
                "guard_overrides",
                "correction_changes",
            ):
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
        variant = config["variants"][method]
        policy = "MlpPolicy" if variant["policy"] == "mlp" else MatchedDQNPolicy
        policy_kwargs = {"net_arch": config["hidden"]}
        if variant["policy"] != "mlp":
            policy_kwargs["local_information"] = variant["policy"] == "local"
        model = DQN(
            policy,
            env,
            seed=seed,
            device="cpu",
            verbose=0,
            policy_kwargs=policy_kwargs,
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


def evaluate_one(job):
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    output, split = Path(job["output"]), job["split"]
    method, seed, case = job["method"], job["seed"], job["case"]
    directory = output / f"evaluation/{split}/{method}/seed_{seed}/{case['id']}"
    directory.mkdir(parents=True, exist_ok=True)
    route = output / f"cases/{split}/{case['id']}/vehicles.rou.xml"
    path = model_path(output, method, seed) if method not in BASELINES else None
    config = json.loads((output / "study.json").read_text())["config"]
    identity = {
        "source_sha256": source_digest(),
        "config": config,
        "case": case,
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
        scenario=case["scenario"],
        sumo_seed=case["traffic"],
        slowdown_window=case["window"],
        trace=True,
    )
    try:
        obs, _ = env.reset(seed=case["traffic"])
        while True:
            action = int(model.predict(obs, deterministic=True)[0]) if model else 0
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        result = {
            "split": split,
            "method": method,
            "training_seed": seed,
            "case_id": case["id"],
            "scenario": case["scenario"],
            "traffic_seed": case["traffic"],
            "group": case["group"],
            "demand_multiplier": case["multiplier"],
            "window_start": case["window"][0],
            "route_sha256": identity["route_sha256"],
            "model_sha256": identity["model_sha256"],
            **info["metrics"],
        }
        save_json(result_path, result)
        return result
    finally:
        env.close()


def evaluate(config, output, workers, split):
    records = prepare_cases(config, output, split)
    jobs = [
        dict(output=str(output), split=split, method=m, seed=seed, case=case)
        for m in METHODS
        for seed in (config["training_seeds"] if m not in BASELINES else [None])
        for case in records
    ]
    rows = pool_map(evaluate_one, jobs, workers)
    pd.DataFrame(rows).to_csv(output / f"{split}.csv", index=False)


def freeze_selection(output):
    frame = pd.read_csv(output / "development.csv")
    subset = frame[frame.group == "rate_1"]
    means = subset.groupby("method")[
        ["over_threshold_fraction", "mean_wait", "mean_system_time"]
    ].mean()
    ordinary = means.loc["residual_dqn"]
    records = []
    for method in ("crossing_30", "crossing_90"):
        row = means.loc[method]
        eligible = bool(
            subset[subset.method == method].valid_complete.all()
            and subset[subset.method == "residual_dqn"].valid_complete.all()
            and row.over_threshold_fraction < ordinary.over_threshold_fraction
            and row.mean_wait <= ordinary.mean_wait * 1.05
            and row.mean_system_time <= ordinary.mean_system_time * 1.05
        )
        records.append({"method": method, "acceptable": eligible, **row.to_dict()})
    eligible = sorted(
        (r for r in records if r["acceptable"]),
        key=lambda r: (r["over_threshold_fraction"], r["method"]),
    )
    result = {
        "source_sha256": source_digest(),
        "study_sha256": sha256(output / "study.json"),
        "development_sha256": sha256(output / "development.csv"),
        "selected_fairness": eligible[0]["method"] if eligible else None,
        "candidates": records,
        "ordinary": ordinary.to_dict(),
        "main_strategy": "residual_dqn",
        "all_predeclared_variants_tested": True,
        "test_was_unopened": not (output / "cases/test").exists(),
    }
    path = output / "freeze.json"
    if path.exists():
        previous = json.loads(path.read_text())
        result["test_was_unopened"] = previous["test_was_unopened"]
        if previous != result:
            raise ValueError("Frozen selection changed")
    else:
        if not result["test_was_unopened"]:
            raise ValueError("Test opened before selection freeze")
        save_json(path, result)
    return result


def run(config_path, output, workers, stage):
    config = read_config(config_path)
    initialize(config, output)
    started = time.perf_counter()
    if stage in ("all", "train"):
        jobs = [
            dict(output=str(output), method=m, seed=s)
            for m in NEW
            for s in config["training_seeds"]
        ]
        pd.DataFrame(pool_map(train_one, jobs, workers)).to_csv(
            output / "training.csv", index=False
        )
    if stage in ("all", "development"):
        evaluate(config, output, workers, "development")
    if stage in ("all", "freeze", "test"):
        freeze_selection(output)
    if stage in ("all", "test"):
        from summarize_iteration import audit_and_export

        # Audit development physics/accounting and identities before opening test.
        audit_and_export(output, splits=("development",))
        evaluate(config, output, workers, "test")
    save_json(
        output / f"execution_{stage}.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "stage": stage,
            "workers": workers,
            "test_evaluated": (output / "test.csv").exists(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/iteration.yaml")
    parser.add_argument("--output", default="outputs/iteration-50k")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--stage", choices=("all", "train", "development", "freeze", "test"), default="all"
    )
    args = parser.parse_args()
    run(Path(args.config), output_directory(args.output), args.workers, args.stage)
