"""Strong transparent baselines with identical signal constraints and service guards."""

from __future__ import annotations

import numpy as np

from fair_signal.arterial.demand import INCOMING, JUNCTIONS, OUTGOING
from fair_signal.arterial.environment import encode_action
from fair_signal.demand import DIRECTIONS, EXITS, PHASE_LANES


def candidates(method):
    if method == "coordinated_fixed":
        return [
            {"durations": d, "offset": offset}
            for d in ([15, 10, 45, 10], [20, 10, 35, 10])
            for offset in (0, 25, 50)
        ]
    if method in ("actuated", "max_pressure"):
        return [{"minimum": m, "hysteresis": 1.0} for m in (10, 20)]
    if method == "aging":
        return [{"minimum": 10, "hysteresis": 1.0, "aging_weight": w} for w in (0.1, 0.3)]
    raise ValueError("Unknown traditional controller")


def scheduled_phase(time, durations, offset):
    position = (time - offset) % sum(d + 4 for d in durations)
    for p, duration in enumerate(durations):
        if position < duration:
            return p
        if position < duration + 4:
            return (p + 1) % 4  # Request clearance at the scheduled green end.
        position -= duration + 4
    raise RuntimeError("Unreachable schedule position")


class ArterialController:
    def __init__(self, method, parameters):
        self.method, self.parameters = method, parameters

    def pressure_scores(self, env, j):
        capacity = np.array([env.capacities[lane] for lane in INCOMING[j]])
        out_capacity = np.array([env.capacities[lane] for lane in OUTGOING[j]])
        upstream = env.counts[j] / capacity
        downstream = env.out_counts[j] / out_capacity
        lane_pressure = []
        for i, d in enumerate(DIRECTIONS):
            right, straight, left = (DIRECTIONS.index(dest) * 2 for dest in EXITS[d])
            lane_pressure.extend(
                (
                    upstream[2 * i] - (0.125 * downstream[right] + 0.875 * downstream[straight]),
                    upstream[2 * i + 1] - downstream[left + 1],
                )
            )
        # Equal saturation-flow weights; nominal cars-per-lane makes hysteresis interpretable.
        return np.array([sum(lane_pressure[i] for i in lanes) * 40 for lanes in PHASE_LANES])

    def predict(self, env):
        actions = []
        for j in JUNCTIONS:
            signal = env.signals[j]
            if signal.mode != "green":
                actions.append(signal.target)
                continue
            if self.method == "coordinated_fixed":
                actions.append(
                    scheduled_phase(
                        env.time,
                        self.parameters["durations"],
                        self.parameters["offset"] if j == "b" else 0,
                    )
                )
                continue
            if signal.elapsed < self.parameters["minimum"]:
                actions.append(signal.phase)
                continue
            if self.method == "max_pressure":
                scores = self.pressure_scores(env, j)
            else:
                scores = np.array([env.queues[j][list(lanes)].sum() for lanes in PHASE_LANES])
                scores += 0.01 * np.array(
                    [env.counts[j][list(lanes)].sum() for lanes in PHASE_LANES]
                )
                if self.method == "aging":
                    scores += self.parameters["aging_weight"] * np.array(
                        [env.ages[j][list(lanes)].max() for lanes in PHASE_LANES]
                    )
            target = int(np.argmax(scores))
            actions.append(
                target
                if scores[target] > scores[signal.phase] + self.parameters["hysteresis"]
                else signal.phase
            )
        return encode_action(actions)
