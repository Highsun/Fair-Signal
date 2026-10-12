"""Count each first threshold crossing once; retain frozen vehicle accounting."""

import numpy as np

from fair_signal.demand import save_json
from fair_signal.environment import write_rows
from fair_signal.iteration.physics import ScheduledArterialEnv
from fair_signal.residual.environment import ResidualEnv


class IterationEnv(ResidualEnv, ScheduledArterialEnv):
    def __init__(self, *args, crossing_weight=0.0, slowdown_window=None, **kwargs):
        self.crossing_weight = crossing_weight
        self.slowdown_window = slowdown_window or (600, 1200)
        if not np.isfinite(crossing_weight) or crossing_weight < 0:
            raise ValueError("Invalid threshold crossing weight")
        if not 0 <= self.slowdown_window[0] < self.slowdown_window[1]:
            raise ValueError("Invalid slowdown window")
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        self.crossing_total = 0
        self.crossing_step = 0
        self.crossing_seconds = []
        return super().reset(**kwargs)

    def _sample_network(self, served):
        waits = super()._sample_network(served)
        # Waiting increments are integer seconds. Each vehicle reaches this value once.
        crossings = sum(w == int(np.floor(self.config.wait_threshold)) + 1 for w in waits)
        self.crossing_total += crossings
        self.crossing_step += crossings
        if self.trace:
            self.crossing_seconds.append(crossings)
        return waits

    def step(self, action):
        self.crossing_step = 0
        start = self.time
        obs, reward, terminated, truncated, info = super().step(action)
        cost = self.crossing_step / (self.config.queue_scale * (self.time - start))
        info["crossing_cost"] = cost
        info["first_crossings"] = self.crossing_step
        return obs, reward - self.crossing_weight * cost, terminated, truncated, info

    def _finish(self, truncated):
        super()._finish(truncated)
        if self.trace:
            for row, crossings in zip(self.trace_rows, self.crossing_seconds, strict=True):
                row["first_crossings"] = crossings
            write_rows(self.episode_dir / "trace.csv", self.trace_rows)
        self.summary["first_crossings"] = self.crossing_total
        self.summary["mean_excess_wait"] = (
            sum(max(w - self.config.wait_threshold, 0) for w in self.total_waits.values())
            / self.planned
            if self.planned
            else 0.0
        )
        save_json(self.episode_dir / "metrics.json", self.summary)
