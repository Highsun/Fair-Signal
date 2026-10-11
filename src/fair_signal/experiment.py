"""Train -> validate -> freeze choices -> test, with reusable identical route files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fair_signal.config import ExperimentConfig, load_config
from fair_signal.controllers import TraditionalController
from fair_signal.demand import build_network, generate_routes, save_json, sha256
from fair_signal.environment import TrafficEnv, write_rows
from fair_signal.objectives import policy_options
from fair_signal.runtime import output_directory, project_root


def source_digest() -> str:
    root = project_root()
    files = sorted((root / "src/fair_signal").glob("*.py")) + [
        root / "uv.lock",
        root / "pyproject.toml",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def initialize(config: ExperimentConfig, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    identity = {"config": config.to_dict(), "source_sha256": source_digest()}
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != identity:
            raise ValueError("Run directory belongs to different code/config; use a new --output")
        return
    save_json(config_path, identity)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    save_json(
        output / "provenance.json",
        {
            "git_revision": revision,
            "git_dirty": bool(dirty),
            "source_sha256": source_digest(),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": 1,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "stable-baselines3", "gymnasium", "traci", "numpy")
            },
            "selection_rule": "mean(mean_wait + 0.5*p95_wait); require complete cases; "
            "all failed: candidate 0 is diagnostic only",
        },
    )


def case_path(output: Path, config: ExperimentConfig, split: str, scenario: str, seed: int) -> Path:
    directory = output / "cases" / split / scenario / str(seed)
    path = directory / "vehicles.rou.xml"
    if not path.exists():
        generate_routes(directory, scenario, split, seed, config.env.demand_seconds)
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(path) != manifest["sha256"]:
        raise ValueError(f"Route checksum mismatch: {path}")
    return path


def candidates(config: ExperimentConfig, method: str) -> list[dict]:
    if method == "fixed":
        return [{"durations": row} for row in config.fixed_candidates]
    if method == "actuated":
        return [{"minimum": value, "hysteresis": 1.0} for value in config.actuated_min_candidates]
    if method == "aging":
        return [
            {"minimum": value, "hysteresis": 1.0, "aging_weight": weight}
            for value in config.actuated_min_candidates
            for weight in config.aging_candidates
        ]
    if method in ("dqn_fair", "dqn_fair_state"):
        return [{"fair_weight": value} for value in config.fair_weights]
    if method in ("dqn_dense", "dqn_dense_state"):
        return [{"fair_weight": config.dense_fair_weight}]
    return [{"fair_weight": 0.0}]


def model_path(output: Path, method: str, candidate: int, seed: int) -> Path:
    return output / "models" / method / f"candidate_{candidate}" / f"seed_{seed}" / "model.zip"


def train(config: ExperimentConfig, output: Path, network: Path) -> None:
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)

    class EpisodeLog(BaseCallback):
        def __init__(self, target: Path, fair_weight: float):
            super().__init__()
            self.target = target
            self.rewards: list[float] = []
            self.rows: list[dict] = []
            self.fair_weight = fair_weight
            self.costs: list[tuple[float, float]] = []

        def _on_step(self) -> bool:
            if not np.isfinite(self.locals["new_obs"]).all():
                raise RuntimeError("Nonfinite observation")
            if not np.isfinite(self.locals["rewards"]).all():
                raise RuntimeError("Nonfinite reward")
            self.rewards.append(float(self.locals["rewards"][0]))
            info = self.locals["infos"][0]
            self.costs.append((info["queue_cost"], info["wait_cost"]))
            metrics = self.locals["infos"][0].get("metrics")
            if metrics is not None:
                costs = np.asarray(self.costs)
                diagnostic = {
                    "mean_queue_cost": float(costs[:, 0].mean()),
                    "mean_fair_cost": float(costs[:, 1].mean()),
                    "mean_weighted_fair_cost": float(self.fair_weight * costs[:, 1].mean()),
                    "fair_cost_active_fraction": float((costs[:, 1] > 0).mean()),
                }
                self.rows.append(
                    {
                        "step": self.num_timesteps,
                        "episode_reward": sum(self.rewards),
                        **diagnostic,
                        **metrics,
                    }
                )
                for key, value in diagnostic.items():
                    self.logger.record(f"traffic/{key}", value)
                self.logger.record("traffic/mean_wait", metrics["mean_wait"])
                self.logger.record("traffic/p95_wait", metrics["p95_wait"])
                self.logger.record("traffic/valid_complete", int(metrics["valid_complete"]))
                self.rewards.clear()
                self.costs.clear()
                write_rows(self.target, self.rows)
                if len(self.rows) % 5 == 0:
                    print(
                        f"  step={self.num_timesteps} mean_wait={metrics['mean_wait']:.1f}s "
                        f"complete={metrics['valid_complete']}",
                        flush=True,
                    )
            return True

    for method in config.methods:
        if not method.startswith("dqn_"):
            continue
        for index, params in enumerate(candidates(config, method)):
            for seed in config.training_seeds:
                path = model_path(output, method, index, seed)
                report_path = path.parent / "training.json"
                if report_path.exists() and path.exists():
                    if json.loads(report_path.read_text())["model_sha256"] != sha256(path):
                        raise ValueError(f"Model checksum mismatch: {path}")
                    print(f"Reuse trained {method} candidate={index} seed={seed}", flush=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"Train {method} candidate={index} seed={seed} steps={config.training_steps}",
                    flush=True,
                )
                env = Monitor(
                    TrafficEnv(
                        config.env,
                        network,
                        path.parent / "episodes",
                        **policy_options(method, params),
                    ),
                    filename=str(path.parent / "monitor.csv"),
                )
                started = time.perf_counter()
                try:
                    model = DQN(
                        "MlpPolicy",
                        env,
                        seed=seed,
                        device="cpu",
                        verbose=0,
                        policy_kwargs={"net_arch": config.hidden},
                        learning_rate=config.learning_rate,
                        buffer_size=config.buffer_size,
                        learning_starts=config.learning_starts,
                        batch_size=config.batch_size,
                        gamma=config.gamma,
                        train_freq=config.train_freq,
                        gradient_steps=1,
                        target_update_interval=config.target_update_interval,
                        exploration_fraction=config.exploration_fraction,
                        exploration_final_eps=0.05,
                        tensorboard_log=str(path.parent / "tensorboard"),
                    )
                    before = [p.detach().clone() for p in model.q_net.parameters()]
                    callback = EpisodeLog(path.parent / "learning.csv", params["fair_weight"])
                    model.learn(config.training_steps, callback=callback)
                    elapsed = time.perf_counter() - started
                    if model._n_updates == 0 or not any(
                        not torch.equal(a, b)
                        for a, b in zip(before, model.q_net.parameters(), strict=True)
                    ):
                        raise RuntimeError("No actual DQN parameter updates")
                    model.save(path)
                    restored = DQN.load(path, device="cpu")
                    if not np.array_equal(
                        model.predict(model._last_obs, deterministic=True)[0],
                        restored.predict(model._last_obs, deterministic=True)[0],
                    ):
                        raise RuntimeError("Model reload prediction mismatch")
                    save_json(
                        report_path,
                        {
                            "method": method,
                            "candidate": index,
                            "parameters": params,
                            "seed": seed,
                            "decision_steps": model.num_timesteps,
                            "gradient_updates": model._n_updates,
                            "completed_episodes": len(callback.rows),
                            "seconds": elapsed,
                            "steps_per_second": model.num_timesteps / elapsed,
                            "model_sha256": sha256(path),
                            "source_sha256": source_digest(),
                            "save_reload_action_matches": True,
                            "network_parameters": sum(p.numel() for p in model.q_net.parameters()),
                            "policy_options": policy_options(method, params),
                        },
                    )
                    print(
                        f"  trained in {elapsed:.1f}s ({model.num_timesteps / elapsed:.1f} "
                        "decision steps/s)",
                        flush=True,
                    )
                finally:
                    env.close()


def evaluate_case(
    config: ExperimentConfig,
    output: Path,
    network: Path,
    method: str,
    candidate: int,
    params: dict,
    training_seed: int | None,
    split: str,
    scenario: str,
    traffic_seed: int,
) -> dict:
    route = case_path(output, config, split, scenario, traffic_seed)
    destination = (
        output
        / "evaluation"
        / split
        / method
        / f"candidate_{candidate}"
        / f"model_{training_seed}"
        / scenario
        / str(traffic_seed)
    )
    model_file = (
        model_path(output, method, candidate, training_seed) if training_seed is not None else None
    )
    identity = {
        "route_sha256": sha256(route),
        "network_sha256": sha256(network),
        "model_sha256": sha256(model_file) if model_file else None,
        "parameters": params,
        "environment": asdict(config.env),
        "source_sha256": source_digest(),
    }
    metrics_file = destination / "episode_00000" / "metrics.json"
    if metrics_file.exists():
        if json.loads((destination / "identity.json").read_text()) != identity:
            raise ValueError(f"Evaluation identity changed: {destination}")
        metrics = json.loads(metrics_file.read_text())
    else:
        destination.mkdir(parents=True, exist_ok=True)
        save_json(destination / "identity.json", identity)
        policy = None
        if model_file:
            from stable_baselines3 import DQN

            policy = DQN.load(model_file, device="cpu")
        controller = TraditionalController(method, params) if policy is None else None
        env = TrafficEnv(
            config.env,
            network,
            destination,
            **policy_options(method, params),
            route_file=route,
            sumo_seed=traffic_seed,
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
                    metrics = info["metrics"]
                    break
        finally:
            env.close()
    return {
        "method": method,
        "candidate": candidate,
        "training_seed": training_seed,
        "scenario": scenario,
        "traffic_seed": traffic_seed,
        "split": split,
        "route_sha256": identity["route_sha256"],
        **metrics,
    }


def tune(config: ExperimentConfig, output: Path, network: Path) -> dict:
    selected, rows, scores = {}, [], []
    for method in config.methods:
        method_scores = []
        for index, params in enumerate(candidates(config, method)):
            seed_group = config.training_seeds if method.startswith("dqn_") else [None]
            cases = [
                evaluate_case(
                    config,
                    output,
                    network,
                    method,
                    index,
                    params,
                    seed,
                    "validation",
                    scenario,
                    traffic_seed,
                )
                for seed in seed_group
                for scenario in config.scenarios
                for traffic_seed in config.validation_seeds
            ]
            rows.extend(cases)
            valid = all(row["valid_complete"] and row["mean_wait"] is not None for row in cases)
            score = (
                float(np.mean([row["mean_wait"] + 0.5 * row["p95_wait"] for row in cases]))
                if valid
                else None
            )
            method_scores.append(score)
            scores.append(
                {
                    "method": method,
                    "candidate": index,
                    "parameters": params,
                    "validation_score": score,
                    "all_cases_complete": valid,
                }
            )
            print(f"Validate {method} candidate={index}: score={score}", flush=True)
        admissible = [i for i, value in enumerate(method_scores) if value is not None]
        # A failed policy is a result, not a reason to discard the rest of the experiment.
        # Do not choose among failed policies using biased finished-only wait statistics.
        best = min(admissible, key=lambda i: method_scores[i]) if admissible else 0
        if not admissible:
            print(
                f"WARNING: {method} has no complete validation candidate; "
                "candidate 0 retained for failure diagnostics only",
                flush=True,
            )
        selected[method] = {
            "candidate": best,
            "parameters": candidates(config, method)[best],
            "validation_eligible": bool(admissible),
            "diagnostic_only": not bool(admissible),
        }
    write_rows(output / "validation.csv", rows)
    save_json(output / "validation_scores.json", scores)
    save_json(output / "selection.json", selected)
    return selected


def test(config: ExperimentConfig, output: Path, network: Path) -> list[dict]:
    selected = json.loads((output / "selection.json").read_text())
    rows = []
    for method in config.methods:
        choice = selected[method]
        seed_group = config.training_seeds if method.startswith("dqn_") else [None]
        for seed in seed_group:
            for scenario in config.scenarios:
                for traffic_seed in config.test_seeds:
                    rows.append(
                        evaluate_case(
                            config,
                            output,
                            network,
                            method,
                            choice["candidate"],
                            choice["parameters"],
                            seed,
                            "test",
                            scenario,
                            traffic_seed,
                        )
                    )
            print(f"Test {method} model_seed={seed} complete", flush=True)
    write_rows(output / "test.csv", rows)
    return rows


def run_experiment(config_path: Path, output_value: str, stage: str = "all") -> Path:
    import torch

    torch.set_num_threads(1)
    config = load_config(config_path)
    output = output_directory(output_value)
    initialize(config, output)
    network = build_network(output / "network")
    for split, seeds in (("validation", config.validation_seeds), ("test", config.test_seeds)):
        for scenario in config.scenarios:
            for seed in seeds:
                case_path(output, config, split, scenario, seed)
    if stage in ("train", "all"):
        train(config, output, network)
    if stage in ("tune", "all"):
        tune(config, output, network)
    if stage in ("test", "all"):
        test(config, output, network)
        from fair_signal.analysis import analyze

        analyze(output)
    return output
