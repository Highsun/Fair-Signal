"""Validated, serializable experiment settings shared by all methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

METHODS = (
    "fixed",
    "actuated",
    "aging",
    "dqn_queue",
    "dqn_wait",
    "dqn_fair",
    "dqn_dense",
    "dqn_dense_state",
    "dqn_fair_state",
)
SCENARIOS = ("balanced", "imbalanced", "changing_peak")


@dataclass
class EnvConfig:
    demand_seconds: int = 3600
    drain_seconds: int = 1800
    decision_seconds: int = 5
    min_green: int = 10
    max_green: int = 60
    yellow_seconds: int = 3
    all_red_seconds: int = 1
    wait_threshold: float = 90.0
    queue_scale: float = 80.0
    wait_scale: float = 180.0

    def validate(self) -> None:
        for key in (
            "demand_seconds",
            "drain_seconds",
            "decision_seconds",
            "min_green",
            "max_green",
            "yellow_seconds",
            "all_red_seconds",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.min_green >= self.max_green:
            raise ValueError("min_green must be smaller than max_green")
        if self.decision_seconds > self.min_green:
            raise ValueError("decision_seconds must be <= min_green")
        if min(self.queue_scale, self.wait_scale, self.wait_threshold) <= 0:
            raise ValueError("Reward/observation scales and threshold must be positive")


@dataclass
class ExperimentConfig:
    label: str = "pilot"
    env: EnvConfig = field(default_factory=EnvConfig)
    training_steps: int = 10000
    training_seeds: list[int] = field(default_factory=lambda: [11])
    validation_seeds: list[int] = field(default_factory=lambda: [1001, 1002])
    test_seeds: list[int] = field(default_factory=lambda: [2001, 2002, 2003])
    scenarios: list[str] = field(default_factory=lambda: list(SCENARIOS))
    methods: list[str] = field(default_factory=lambda: list(METHODS[:6]))
    fair_weights: list[float] = field(default_factory=lambda: [0.1, 0.3, 1.0])
    dense_fair_weight: float = 0.3
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    learning_starts: int = 1000
    buffer_size: int = 50000
    batch_size: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 1000
    exploration_fraction: float = 0.5
    fixed_candidates: list[list[int]] = field(
        default_factory=lambda: [[25, 10, 25, 10], [20, 10, 40, 10], [35, 10, 35, 10]]
    )
    actuated_min_candidates: list[int] = field(default_factory=lambda: [10, 20])
    aging_candidates: list[float] = field(default_factory=lambda: [0.1, 0.3])

    def validate(self) -> None:
        self.env.validate()
        if not self.methods or set(self.methods) - set(METHODS):
            raise ValueError(f"methods must be drawn from {METHODS}")
        if not self.scenarios or set(self.scenarios) - set(SCENARIOS):
            raise ValueError(f"scenarios must be drawn from {SCENARIOS}")
        groups = (self.training_seeds, self.validation_seeds, self.test_seeds)
        if any(not seeds or len(set(seeds)) != len(seeds) for seeds in groups):
            raise ValueError("Seed groups must be nonempty and contain no duplicates")
        if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
            raise ValueError("Training, validation and test seeds must be disjoint")
        if any(type(seed) is not int or seed < 0 for seeds in groups for seed in seeds):
            raise ValueError("Seeds must be nonnegative integers")
        for key in (
            "training_steps",
            "buffer_size",
            "batch_size",
            "train_freq",
            "target_update_interval",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 <= self.learning_starts < self.training_steps:
            raise ValueError("learning_starts must be below training_steps")
        if not self.hidden or min(self.hidden) <= 0:
            raise ValueError("hidden must contain positive layer sizes")
        if not 0 < self.gamma <= 1 or not 0 < self.exploration_fraction <= 1:
            raise ValueError("Invalid discount or exploration_fraction")
        if self.learning_rate <= 0 or self.batch_size > self.buffer_size:
            raise ValueError("Invalid learning rate or replay size")
        if not self.fair_weights or min(self.fair_weights) < 0:
            raise ValueError("fair_weights must be nonnegative")
        if not 0 <= self.dense_fair_weight < float("inf"):
            raise ValueError("dense_fair_weight must be finite and nonnegative")
        if not self.fixed_candidates or any(
            len(row) != 4 or min(row) < self.env.min_green or max(row) > self.env.max_green
            for row in self.fixed_candidates
        ):
            raise ValueError("Each fixed plan needs four durations within green limits")
        if not self.actuated_min_candidates or any(
            not self.env.min_green <= value <= self.env.max_green
            for value in self.actuated_min_candidates
        ):
            raise ValueError("Invalid actuated minimum")
        if not self.aging_candidates or min(self.aging_candidates) < 0:
            raise ValueError("Invalid aging weights")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Experiment configuration must be a mapping")
    data = dict(data)
    data["env"] = EnvConfig(**data.get("env", {}))
    config = ExperimentConfig(**data)
    config.validate()
    return config
