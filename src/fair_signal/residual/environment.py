"""Baseline-assisted actions; the historical arterial environment stays unchanged."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from fair_signal.arterial.controllers import ArterialController
from fair_signal.arterial.demand import JUNCTIONS
from fair_signal.arterial.environment import (
    ArterialEnv,
    decode_action,
    encode_action,
    service_action,
)
from fair_signal.demand import PHASE_LANES, save_json
from fair_signal.environment import write_rows

OBSERVATION_DIM = 162


def corrected_phase(signal, counts, waits, baseline, action, used, limit, urgent_threshold):
    """Return a requested physical phase and extension seconds charged this step."""
    if action not in (0, 1, 2):
        raise ValueError("Correction action must be 0, 1 or 2")
    if signal.mode != "green" or signal.elapsed < signal.config.min_green:
        return baseline, 0
    if action == 1:
        interval = signal.config.decision_seconds
        if (
            baseline != signal.phase
            and sum(counts[i] for i in PHASE_LANES[signal.phase]) > 0
            and used + interval <= limit
            and signal.elapsed + interval <= signal.config.max_green
        ):
            return signal.phase, interval
    elif action == 2:
        scores = [
            max((waits[i] for i in lanes if counts[i] > 0), default=0) for lanes in PHASE_LANES
        ]
        target = int(np.argmax(scores))
        if scores[target] >= urgent_threshold:
            return target, 0
    return baseline, 0


class ResidualEnv(ArterialEnv):
    def __init__(
        self,
        *args,
        baseline_parameters,
        baseline_method="actuated",
        extension_seconds=10,
        urgent_threshold=90.0,
        **kwargs,
    ):
        self.baseline = ArterialController(baseline_method, baseline_parameters)
        self.extension_seconds = extension_seconds
        self.urgent_threshold = urgent_threshold
        super().__init__(*args, **kwargs)
        if (
            type(extension_seconds) is not int
            or extension_seconds < self.config.decision_seconds
            or extension_seconds % self.config.decision_seconds
            or not np.isfinite(urgent_threshold)
            or urgent_threshold <= 0
        ):
            raise ValueError("Invalid bounded correction settings")
        self.action_space = gym.spaces.Discrete(9)
        self.physical_action_space = gym.spaces.Discrete(16)
        self.observation_space = gym.spaces.Box(0, 1, (OBSERVATION_DIM,), np.float32)

    def reset(self, **kwargs):
        self.extension_used = dict.fromkeys(JUNCTIONS, 0)
        self.extension_keys = dict.fromkeys(JUNCTIONS, None)
        self.decisions = []
        return super().reset(**kwargs)

    def _observation(self):
        extras = []
        suggestions = decode_action(self.baseline.predict(self))
        for j, phase in zip(JUNCTIONS, suggestions, strict=True):
            signal = self.signals[j]
            key = (signal.switches, signal.phase)
            if self.extension_keys[j] != key:
                self.extension_used[j], self.extension_keys[j] = 0, key
            extras.extend(np.eye(4)[phase])
            extras.append(self.extension_used[j] / self.extension_seconds)
        return np.concatenate((super()._observation(), extras)).astype(np.float32)

    def step(self, action):
        if self.connection is None or self.summary is not None:
            raise RuntimeError("Call reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError("Joint correction must be in [0, 8]")
        choices = divmod(int(action), 3)
        baseline = decode_action(self.baseline.predict(self))
        phases = []
        for j, choice, base in zip(JUNCTIONS, choices, baseline, strict=True):
            phase, spent = corrected_phase(
                self.signals[j],
                self.counts[j],
                self.max_waits[j],
                base,
                choice,
                self.extension_used[j],
                self.extension_seconds,
                self.urgent_threshold,
            )
            guarded_base = service_action(
                self.signals[j], self.counts[j], self.ages[j], base, self.guard_threshold
            )
            guarded_phase = service_action(
                self.signals[j], self.counts[j], self.ages[j], phase, self.guard_threshold
            )
            # Charge only an extension that survives the shared service guard.
            charged = spent if guarded_phase == phase and guarded_phase != guarded_base else 0
            self.extension_used[j] += charged
            self.decisions.append(
                {
                    "second": self.time,
                    "junction": j,
                    "choice": choice,
                    "baseline": base,
                    "proposed": phase,
                    "guarded_baseline": guarded_base,
                    "guarded_proposed": guarded_phase,
                    "changed": int(phase != base),
                    "effective": int(guarded_phase != guarded_base),
                    "extension_charged": charged,
                    "extension_used": self.extension_used[j],
                    "switches_before": self.signals[j].switches,
                }
            )
            phases.append(phase)
        # The inherited implementation validates physical phase pairs, whereas
        # the public learning interface validates correction pairs above.
        correction_space = self.action_space
        self.action_space = self.physical_action_space
        try:
            result = super().step(encode_action(phases))
        finally:
            self.action_space = correction_space
        result[-1]["correction_changes"] = sum(r["effective"] for r in self.decisions[-2:])
        return result

    def _finish(self, truncated):
        super()._finish(truncated)
        rows = self.decisions
        self.summary.update(
            {
                "correction_node_decisions": len(rows),
                "correction_changes": sum(r["changed"] for r in rows),
                "correction_effective": sum(r["effective"] for r in rows),
                "extension_seconds_charged": sum(r["extension_charged"] for r in rows),
                **{f"choice_{i}": sum(r["choice"] == i for r in rows) for i in range(3)},
            }
        )
        save_json(self.episode_dir / "metrics.json", self.summary)
        write_rows(self.episode_dir / "corrections.csv", rows)
