"""Frozen arterial physics; only the explicit slowdown window expression differs.

The step body is derived from ArterialEnv.step. Its place between ResidualEnv and
ArterialEnv in the MRO preserves the historical correction implementation.
"""

import numpy as np

from fair_signal.arterial.demand import JUNCTIONS
from fair_signal.arterial.environment import ArterialEnv, decode_action, service_action
from fair_signal.demand import PHASE_LANES


class ScheduledArterialEnv(ArterialEnv):
    def step(self, action):
        if self.connection is None or self.summary is not None:
            raise RuntimeError("Call reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError("Invalid joint action")
        proposed = decode_action(int(action))
        executed = tuple(
            service_action(self.signals[j], self.counts[j], self.ages[j], p, self.guard_threshold)
            for j, p in zip(JUNCTIONS, proposed, strict=True)
        )
        overrides = sum(p != e for p, e in zip(proposed, executed, strict=True))
        self.guard_overrides += overrides
        for j, p in zip(JUNCTIONS, executed, strict=True):
            self.signals[j].request(p)
        costs, fair_costs = [], []
        terminated = truncated = False
        for _ in range(self.config.decision_seconds):
            slow = (
                self.current_scenario == "downstream_slowdown"
                and self.slowdown_window[0] <= self.time < self.slowdown_window[1]
            )
            if slow != self.slowdown_active:
                self.connection.edge.setMaxSpeed("b_e", 1.5 if slow else 13.89)
                self.slowdown_active = slow
            served = {}
            for j, signal in self.signals.items():
                self.connection.trafficlight.setRedYellowGreenState(j, signal.before_second())
                served[j] = PHASE_LANES[signal.phase] if signal.mode == "green" else ()
            self.connection.simulationStep()
            self.time += 1
            waits = self._sample_network(served)
            queue_cost = (self.queue_now + self.pending_now) / self.config.queue_scale
            # Exact increase of sum(max(lifetime_stop - threshold, 0)) / queue_scale.
            fair_cost = sum(w > self.config.wait_threshold for w in waits) / self.config.queue_scale
            costs.append(queue_cost)
            fair_costs.append(fair_cost)
            if self.trace:
                row = {
                    "second": self.time,
                    "queue": self.queue_now,
                    "pending": self.pending_now,
                    "queue_cost": queue_cost,
                    "fair_cost": fair_cost,
                    "guard_overrides": overrides,
                    "slowdown": slow,
                }
                for k, j in enumerate(JUNCTIONS):
                    s = self.signals[j]
                    row.update(
                        {
                            f"{j}_{name}": value
                            for name, value in {
                                "phase": s.phase,
                                "target": s.target,
                                "mode": s.mode,
                                "elapsed": s.elapsed,
                                "proposed": proposed[k],
                                "executed": executed[k],
                                "max_age": float(self.ages[j].max()),
                            }.items()
                        }
                    )
                self.trace_rows.append(row)
            for signal in self.signals.values():
                signal.after_second()
            terminated = (
                self.time >= self.config.demand_seconds
                and self.connection.simulation.getMinExpectedNumber() == 0
            )
            truncated = (
                not terminated
                and self.time >= self.config.demand_seconds + self.config.drain_seconds
            )
            if terminated or truncated:
                break
        obs = self._observation()
        info = {
            "queue_cost": float(np.mean(costs)),
            "wait_cost": float(np.mean(fair_costs)),
            "guard_overrides": overrides,
        }
        if terminated or truncated:
            self._finish(truncated)
            info["metrics"] = self.summary
        return (
            obs,
            -info["queue_cost"] - self.fair_weight * info["wait_cost"],
            terminated,
            truncated,
            info,
        )
