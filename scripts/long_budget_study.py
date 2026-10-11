"""One uninterrupted DQN trajectory, snapshots AFTER updates, evaluation afterwards.

The core experiment implementation stays frozen. This separate study runner is
hashed in study.json and training.json; it uses the same SB3/environment settings.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from fair_signal.config import ExperimentConfig, load_config
from fair_signal.demand import build_network, save_json, sha256
from fair_signal.environment import TrafficEnv, write_rows
from fair_signal.experiment import (
    candidates,
    case_path,
    evaluate_case,
    initialize,
    model_path,
    source_digest,
    tune,
)
from fair_signal.objectives import policy_options
from fair_signal.runtime import output_directory, project_root


class SnapshotDQN(DQN):
    """Only add disk snapshots; gradient/exploration/target updates remain SB3's."""

    snapshot_hook = None

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "snapshot_hook"]

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps, batch_size)
        if self.snapshot_hook is not None:
            self.snapshot_hook(self)


class EpisodeLog(BaseCallback):
    """Same decision-mean diagnostic contract as the frozen experiment trainer."""

    def __init__(self, path: Path, fair_weight: float):
        super().__init__()
        self.path = path
        self.fair_weight = fair_weight
        self.rows = []
        self.rewards = []
        self.costs = []

    def _on_step(self) -> bool:
        if not np.isfinite(self.locals["new_obs"]).all():
            raise RuntimeError("Nonfinite observation")
        if not np.isfinite(self.locals["rewards"]).all():
            raise RuntimeError("Nonfinite reward")
        info = self.locals["infos"][0]
        self.rewards.append(float(self.locals["rewards"][0]))
        self.costs.append((info["queue_cost"], info["wait_cost"]))
        if "metrics" in info:
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
                    **info["metrics"],
                }
            )
            for key, value in diagnostic.items():
                self.logger.record(f"traffic/{key}", value)
            for key in ("mean_wait", "p95_wait", "valid_complete"):
                self.logger.record(f"traffic/{key}", info["metrics"][key])
            self.rewards.clear()
            self.costs.clear()
            write_rows(self.path, self.rows)
            if len(self.rows) % 10 == 0:
                print(
                    f"step={self.num_timesteps} wait={info['metrics']['mean_wait']:.2f}", flush=True
                )
        return True


def checkpoint_config(config: ExperimentConfig, step: int) -> ExperimentConfig:
    return replace(
        config,
        label=f"{config.label}-step-{step}",
        training_steps=step,
        exploration_fraction=config.training_steps * config.exploration_fraction / step,
        methods=config.methods
        if step == config.training_steps
        else [method for method in config.methods if method.startswith("dqn_")],
    )


def train_one(
    config: ExperimentConfig, root: Path, method: str, seed: int, steps: list[int]
) -> None:
    runner_hash = sha256(Path(__file__))
    terminal = model_path(root / f"step_{steps[-1]}", method, 0, seed).with_name("training.json")
    if terminal.exists():
        for step in steps:
            path = model_path(root / f"step_{step}", method, 0, seed)
            record = json.loads(path.with_name("training.json").read_text())
            if record["runner_sha256"] != runner_hash or record["model_sha256"] != sha256(path):
                raise ValueError("Snapshot runner/model mismatch")
        print(f"Reuse complete trajectory: {method} seed={seed}", flush=True)
        return
    directory = root / "trajectories" / method / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    params = candidates(config, method)[0]
    env = Monitor(
        TrafficEnv(
            config.env,
            root / "network/intersection.net.xml",
            directory / "episodes",
            **policy_options(method, params),
        ),
        filename=str(directory / "monitor.csv"),
    )
    started = time.perf_counter()
    pending = {}
    callback = EpisodeLog(directory / "learning.csv", params["fair_weight"])

    def snapshot(model):
        step = model.num_timesteps
        if step not in steps or step in pending:
            return
        path = model_path(root / f"step_{step}", method, 0, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        pending[step] = {
            "method": method,
            "candidate": 0,
            "parameters": params,
            "seed": seed,
            "decision_steps": step,
            "gradient_updates": model._n_updates,
            "completed_episodes": len(callback.rows),
            "seconds": time.perf_counter() - started,
            "model_sha256": sha256(path),
            "source_sha256": source_digest(),
            "runner_sha256": runner_hash,
            "trajectory_budget": config.training_steps,
            "exploration_decay_steps": config.training_steps * config.exploration_fraction,
            "network_parameters": sum(p.numel() for p in model.q_net.parameters()),
            "policy_options": policy_options(method, params),
            "probe_observation": model._last_obs.tolist(),
            "probe_action": model.predict(model._last_obs, deterministic=True)[0].tolist(),
        }
        print(f"Snapshot after update: {method} seed={seed} step={step}", flush=True)

    try:
        model = SnapshotDQN(
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
            tensorboard_log=str(directory / "tensorboard"),
        )
        before = [p.detach().clone() for p in model.q_net.parameters()]
        model.snapshot_hook = snapshot
        model.learn(config.training_steps, callback=callback)
        if not any(
            not torch.equal(a, b) for a, b in zip(before, model.q_net.parameters(), strict=True)
        ):
            raise RuntimeError("No actual gradient updates")
        if set(pending) != set(steps):
            raise RuntimeError("Snapshot missing; steps must align with gradient updates")
    finally:
        env.close()
    # Loading invokes seed initialization; do it only AFTER the entire trajectory.
    for step in steps:
        path = model_path(root / f"step_{step}", method, 0, seed)
        record = pending[step]
        restored = DQN.load(path, device="cpu")
        if restored.num_timesteps != step or restored._n_updates != record["gradient_updates"]:
            raise RuntimeError("Snapshot saved before its declared update")
        if not all(
            torch.isfinite(value).all().item() for value in restored.policy.state_dict().values()
        ):
            raise RuntimeError("Nonfinite model weights")
        actual = restored.predict(np.asarray(record.pop("probe_observation")), deterministic=True)[
            0
        ]
        if not np.array_equal(actual, record.pop("probe_action")):
            raise RuntimeError("Snapshot reload action mismatch")
        record["save_reload_action_matches"] = True
        record["weights_finite"] = True
        record["steps_per_second"] = step / record["seconds"]
        save_json(path.with_name("training.json"), record)
        rows = [row for row in callback.rows if row["step"] <= step]
        write_rows(path.with_name("learning.csv"), rows)


def seed_worker(config_path: str, output: str, seed: int, steps: list[int]) -> int:
    torch.set_num_threads(1)
    config = load_config(Path(config_path))
    root = Path(output)
    for method in config.methods:
        if not method.startswith("dqn_"):
            continue
        train_one(config, root, method, seed, steps)
        for step in steps:
            one = checkpoint_config(config, step)
            destination = root / f"step_{step}"
            for scenario in config.scenarios:
                for traffic in config.validation_seeds:
                    evaluate_case(
                        one,
                        destination,
                        destination / "network/intersection.net.xml",
                        method,
                        0,
                        candidates(config, method)[0],
                        seed,
                        "validation",
                        scenario,
                        traffic,
                    )
        print(f"Trajectory trained and validated: {method} seed={seed}", flush=True)
    return seed


def run(config_path: Path, root: Path, steps: list[int], workers: int) -> None:
    config = load_config(config_path)
    if steps != sorted(set(steps)) or steps[-1] != config.training_steps:
        raise ValueError("Snapshots must be increasing and end at the full budget")
    if any(step % config.train_freq or step <= config.learning_starts for step in steps):
        raise ValueError("Snapshots must align with actual gradient updates")
    if config.training_steps * config.exploration_fraction > steps[0]:
        raise ValueError("Exploration must finish no later than the first checkpoint")
    identity = {
        "runner_sha256": sha256(Path(__file__)),
        "checkpoints": steps,
        "source_sha256": source_digest(),
        "config": config.to_dict(),
        "evaluation_timing": "after uninterrupted trajectory",
        "test_evaluated": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    if (root / "study.json").exists() and json.loads((root / "study.json").read_text()) != identity:
        raise ValueError("Study runner/configuration changed; use a new output directory")
    save_json(root / "study.json", identity)
    initialize(config, root)
    build_network(root / "network")
    for step in steps:
        one = checkpoint_config(config, step)
        destination = root / f"step_{step}"
        initialize(one, destination)
        build_network(destination / "network")
        for scenario in config.scenarios:
            for seed in config.validation_seeds:
                case_path(destination, one, "validation", scenario, seed)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(seed_worker, str(config_path), str(root), seed, steps)
            for seed in config.training_seeds
        ]
        terminal = root / f"step_{steps[-1]}"
        for method in config.methods:
            if method.startswith("dqn_"):
                continue
            for scenario in config.scenarios:
                for traffic in config.validation_seeds:
                    evaluate_case(
                        checkpoint_config(config, steps[-1]),
                        terminal,
                        terminal / "network/intersection.net.xml",
                        method,
                        0,
                        candidates(config, method)[0],
                        None,
                        "validation",
                        scenario,
                        traffic,
                    )
        for future in as_completed(futures):
            print(f"Seed finished: {future.result()}", flush=True)
    for step in steps:
        destination = root / f"step_{step}"
        tune(
            checkpoint_config(config, step),
            destination,
            destination / "network/intersection.net.xml",
        )
    if not (root / "execution.json").exists():
        records = [
            json.loads(path.read_text())
            for path in (root / f"step_{steps[-1]}/models").glob("*/*/*/training.json")
        ]
        save_json(
            root / "execution.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "training_seconds": sum(record["seconds"] for record in records),
                "independent_trajectories": len(records),
                "saved_snapshots": len(records) * len(steps),
                "workers": workers,
                "test_evaluated": False,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/long-budget.yaml")
    parser.add_argument("--output", default="outputs/long-budget-150k")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[50000, 100000, 150000])
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    config_path = (project_root() / args.config).resolve()
    if not config_path.is_relative_to(project_root()):
        parser.error("Configuration must stay inside the repository")
    run(config_path, output_directory(args.output), args.checkpoints, args.workers)
