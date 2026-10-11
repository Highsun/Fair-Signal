"""Transparent traditional policies, all using the same environment signal layer."""

from __future__ import annotations

import numpy as np

from fair_signal.demand import PHASE_LANES
from fair_signal.environment import TrafficEnv


class TraditionalController:
    def __init__(self, method: str, parameters: dict):
        self.method, self.parameters = method, parameters

    def predict(self, env: TrafficEnv) -> int:
        phase = env.signal.phase
        if env.signal.mode != "green":
            return env.signal.target
        if self.method == "fixed":
            if env.signal.elapsed >= self.parameters["durations"][phase]:
                return (phase + 1) % 4
            return phase
        if env.signal.elapsed < self.parameters["minimum"]:
            return phase
        scores = np.array([env.queues[list(lanes)].sum() for lanes in PHASE_LANES])
        demand = np.array([env.counts[list(lanes)].sum() for lanes in PHASE_LANES])
        if self.method == "aging":
            scores += self.parameters["aging_weight"] * np.array(
                [env.request_ages[list(lanes)].max() for lanes in PHASE_LANES]
            )
        # Demand tie-break prevents waiting for a stop before serving an approaching car.
        scores += 0.01 * demand
        target = int(np.argmax(scores))
        if scores[target] <= scores[phase] + self.parameters.get("hysteresis", 1.0):
            return phase
        return target
