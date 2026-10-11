"""Joint 16-action control with unique-vehicle accounting and a shared service guard."""

from __future__ import annotations

import subprocess
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import numpy as np
import traci
import traci.constants as tc
from sumolib.miscutils import getFreeSocketPort

from fair_signal.arterial.demand import (
    ALL_LANES,
    CONNECTOR_LANES,
    INCOMING,
    JUNCTIONS,
    ORIGINS,
    OUTGOING,
    SCENARIOS,
    generate_routes,
)
from fair_signal.demand import PHASE_LANES, save_json
from fair_signal.environment import TrafficEnv, gui_arguments, write_rows
from fair_signal.signals import Signal

LOCAL_DIM = 76


def decode_action(action: int) -> tuple[int, int]:
    if not 0 <= action < 16:
        raise ValueError("Joint action must be in [0, 15]")
    return divmod(action, 4)


def encode_action(phases) -> int:
    a, b = phases
    if a not in range(4) or b not in range(4):
        raise ValueError("Each phase must be in [0, 3]")
    return int(a * 4 + b)


def service_action(signal, counts, ages, proposed: int, threshold: float) -> int:
    phase_ages = [
        max((ages[i] for i in lanes if counts[i] > 0), default=0) for lanes in PHASE_LANES
    ]
    urgent = int(np.argmax(phase_ages))
    if (
        phase_ages[urgent] >= threshold
        and signal.mode == "green"
        and signal.elapsed >= signal.config.min_green
    ):
        return urgent
    return proposed


class ArterialEnv(TrafficEnv):
    """Separate environment; inherits only common startup options and process cleanup.

    Both ordinary and fair DQN see IDENTICAL 152-dimensional observations. All
    controllers use the same guard during both training and evaluation. The fair
    cost is the lifetime excess-stop increment, counted once per physical vehicle.
    """

    def __init__(
        self,
        config,
        network: Path,
        output: Path,
        *,
        fair_weight=0.0,
        route_file=None,
        scenario=None,
        sumo_seed=0,
        trace=False,
        guard_threshold=180.0,
        gui=False,
        gui_delay=100,
        start_paused=True,
    ):
        super().__init__(
            config,
            network,
            output,
            fair_weight=fair_weight,
            route_file=route_file,
            sumo_seed=sumo_seed,
            trace=trace,
            gui=gui,
            gui_delay=gui_delay,
            start_paused=start_paused,
        )
        self.scenario = scenario
        self.guard_threshold = guard_threshold
        if not np.isfinite(fair_weight) or fair_weight < 0 or guard_threshold <= 0:
            raise ValueError("Invalid fairness weight/service guard")
        self.action_space = gym.spaces.Discrete(16)
        self.observation_space = gym.spaces.Box(0, 1, (LOCAL_DIM * 2,), np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.close()
        self.episode += 1
        self.episode_dir = self.output / f"episode_{self.episode:05d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.current_scenario = self.scenario or str(self.np_random.choice(SCENARIOS))
        if self.route_file is None:
            self.current_sumo_seed = int(self.np_random.integers(0, 2**31 - 1))
            route = generate_routes(
                self.episode_dir,
                self.current_scenario,
                "train",
                self.current_sumo_seed,
                self.config.demand_seconds,
            )
        else:
            route, self.current_sumo_seed = self.route_file, self.sumo_seed
        vehicles = ET.parse(route).getroot().findall("vehicle")
        self.planned = len(vehicles)
        self.planned_times = np.sort([float(v.get("depart")) for v in vehicles])
        self.trip_file = self.episode_dir / "tripinfo.xml"
        command = [
            "sumo-gui" if self.gui else "sumo",
            "-n",
            str(self.network),
            "-r",
            str(route),
            "--seed",
            str(self.current_sumo_seed),
            "--step-length",
            "1",
            "--time-to-teleport",
            "-1",
            "--max-depart-delay",
            "-1",
            "--tripinfo-output",
            str(self.trip_file),
            "--tripinfo-output.write-unfinished",
            "true",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--log",
            str(self.episode_dir / "sumo.log"),
        ]
        if self.gui:
            command += gui_arguments(self.gui_delay, self.start_paused)
            port = getFreeSocketPort()
            self.gui_process = subprocess.Popen(command + ["--remote-port", str(port)])
            try:
                self.connection = traci.connect(port, proc=self.gui_process, waitBetweenRetries=0.1)
                self.connection.getVersion()
            except BaseException:
                self.close()
                raise
        else:
            label = f"arterial-{uuid.uuid4().hex}"
            traci.start(command, label=label, doSwitch=False, stdout=None)
            self.connection = traci.getConnection(label)
        self.signals = {j: Signal(self.config) for j in JUNCTIONS}
        for j, signal in self.signals.items():
            self.connection.trafficlight.setRedYellowGreenState(j, signal.state)
        self.lengths = {lane: self.connection.lane.getLength(lane) for lane in ALL_LANES}
        self.capacities = {lane: self.lengths[lane] / 7.5 for lane in ALL_LANES}
        for lane in ALL_LANES:
            self.connection.lane.subscribe(
                lane,
                (
                    tc.LAST_STEP_VEHICLE_ID_LIST,
                    tc.LAST_STEP_VEHICLE_NUMBER,
                    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                ),
            )
        self.counts = {j: np.zeros(8) for j in JUNCTIONS}
        self.queues = {j: np.zeros(8) for j in JUNCTIONS}
        self.ages = {j: np.zeros(8) for j in JUNCTIONS}
        self.max_waits = {j: np.zeros(8) for j in JUNCTIONS}
        self.segments = {j: np.zeros((8, 3)) for j in JUNCTIONS}
        self.out_counts = {j: np.zeros(8) for j in JUNCTIONS}
        self.total_waits = {}
        self.trace_rows = []
        self.summary = None
        self.time = self.departed = self.arrived = self.teleports = self.collisions = 0
        self.pending_now = self.queue_now = self.peak_queue = self.queue_sum = 0
        self.pending_sum = self.arrived_in_window = self.guard_overrides = 0
        self.connector_queue_sum = self.connector_congested_seconds = 0
        self.connector_peak_occupancy = 0.0
        self.slowdown_active = False
        return self._observation(), {}

    def _sample_network(self, served):
        conn = self.connection
        new = conn.simulation.getDepartedIDList()
        self.departed += len(new)
        self.arrived += len(conn.simulation.getArrivedIDList())
        self.teleports += conn.simulation.getStartingTeleportNumber()
        self.collisions += conn.simulation.getCollidingVehiclesNumber()
        for v in new:
            conn.vehicle.subscribe(v, (tc.VAR_SPEED, tc.VAR_LANEPOSITION))
        vehicle_data = conn.vehicle.getAllSubscriptionResults()
        stopped = []
        for v, data in vehicle_data.items():
            self.total_waits[v] = self.total_waits.get(v, 0) + (data[tc.VAR_SPEED] <= 0.1)
            if data[tc.VAR_SPEED] <= 0.1:
                stopped.append(self.total_waits[v])
        lane_data = conn.lane.getAllSubscriptionResults()
        for j in JUNCTIONS:
            self.segments[j][:] = 0
            for i, lane in enumerate(INCOMING[j]):
                data = lane_data[lane]
                self.counts[j][i] = data[tc.LAST_STEP_VEHICLE_NUMBER]
                self.queues[j][i] = data[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
                waits = []
                for v in data[tc.LAST_STEP_VEHICLE_ID_LIST]:
                    if v not in vehicle_data:
                        continue
                    values = vehicle_data[v]
                    section = min(int(values[tc.VAR_LANEPOSITION] * 3 / self.lengths[lane]), 2)
                    self.segments[j][i, section] += 1
                    if values[tc.VAR_SPEED] <= 0.1:
                        waits.append(self.total_waits[v])
                self.max_waits[j][i] = max(waits, default=0)
                self.ages[j][i] = (
                    self.ages[j][i] + 1 if self.counts[j][i] > 0 and i not in served[j] else 0
                )
            self.out_counts[j] = np.array(
                [lane_data[lane][tc.LAST_STEP_VEHICLE_NUMBER] for lane in OUTGOING[j]], dtype=float
            )
        self.queue_now = len(stopped)  # Includes exit/internal lanes; each vehicle counted ONCE.
        self.pending_now = int(np.searchsorted(self.planned_times, self.time, side="left"))
        self.pending_now -= self.departed
        if self.pending_now < 0:
            raise RuntimeError("Vehicle departure accounting became negative")
        occupied = [
            lane_data[lane][tc.LAST_STEP_VEHICLE_NUMBER] / self.capacities[lane]
            for lane in CONNECTOR_LANES
        ]
        if self.time <= self.config.demand_seconds:
            self.queue_sum += self.queue_now
            self.pending_sum += self.pending_now
            self.peak_queue = max(self.peak_queue, self.queue_now)
            self.arrived_in_window = self.arrived
            self.connector_queue_sum += sum(
                lane_data[lane][tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for lane in CONNECTOR_LANES
            )
            self.connector_congested_seconds += max(occupied) >= 0.8
            self.connector_peak_occupancy = max(self.connector_peak_occupancy, max(occupied))
        return stopped

    def _observation(self):
        values = []
        for j in JUNCTIONS:
            signal = self.signals[j]
            capacity = np.array([self.capacities[lane] for lane in INCOMING[j]])
            out_capacity = np.array([self.capacities[lane] for lane in OUTGOING[j]])
            values.extend(
                np.concatenate(
                    (
                        np.eye(4)[signal.phase],
                        np.eye(4)[signal.target],
                        np.eye(3)[("green", "yellow", "all_red").index(signal.mode)],
                        [signal.elapsed / self.config.max_green],
                        self.counts[j] / capacity,
                        self.queues[j] / capacity,
                        (self.segments[j] / (capacity[:, None] / 3)).ravel(),
                        self.out_counts[j] / out_capacity,
                        self.max_waits[j] / (self.max_waits[j] + self.config.wait_scale),
                        self.ages[j] / (self.ages[j] + self.config.wait_scale),
                    )
                )
            )
        return np.clip(values, 0, 1).astype(np.float32)

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
                and self.config.demand_seconds / 3 <= self.time < 2 * self.config.demand_seconds / 3
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

    def _finish(self, truncated):
        running = self.connection.vehicle.getIDCount()
        self.close()
        rows = []
        for trip in ET.parse(self.trip_file).getroot().findall("tripinfo"):
            vehicle = trip.get("id")
            row = {
                "vehicle": vehicle,
                "origin": vehicle.split("_")[0],
                "completed": float(trip.get("arrival", "-1")) >= 0
                and trip.get("vaporized", "") in ("", "false", "0"),
            }
            row.update(
                {
                    k: float(trip.get(k, "0"))
                    for k in ("waitingTime", "duration", "departDelay", "waitingCount", "timeLoss")
                }
            )
            if row["waitingTime"] != self.total_waits.get(vehicle, 0):
                raise RuntimeError("Online lifetime stop time differs from SUMO TripInfo")
            rows.append(row)
        complete = [r for r in rows if r["completed"]]
        waits = [r["waitingTime"] for r in complete]
        mean = lambda values: float(np.mean(values)) if values else None  # noqa: E731
        summary = {
            "planned": self.planned,
            "departed": self.departed,
            "arrived": self.arrived,
            "completed": len(complete),
            "running": running,
            "pending_insertion": self.planned - self.departed,
            "abnormal_removals": self.departed - self.arrived - running,
            "teleport_events": self.teleports,
            "collision_events": self.collisions,
            "truncated": truncated,
            "simulated_seconds": self.time,
            "drain_seconds": max(0, self.time - self.config.demand_seconds),
            "throughput_demand_window": self.arrived_in_window,
            "mean_wait": mean(waits),
            "p95_wait": float(np.percentile(waits, 95)) if waits else None,
            "max_wait": max(waits, default=None),
            "over_threshold_fraction": mean([w > self.config.wait_threshold for w in waits]),
            "mean_travel_time": mean([r["duration"] for r in complete]),
            "mean_system_time": mean([r["duration"] + r["departDelay"] for r in complete]),
            "mean_depart_delay": mean([r["departDelay"] for r in complete]),
            "mean_wait_including_insertion": mean(
                [r["waitingTime"] + r["departDelay"] for r in complete]
            ),
            "mean_stops": mean([r["waitingCount"] for r in complete]),
            "mean_queue": self.queue_sum / self.config.demand_seconds,
            "peak_queue": self.peak_queue,
            "mean_pending": self.pending_sum / self.config.demand_seconds,
            "connector_mean_queue": self.connector_queue_sum / self.config.demand_seconds,
            "connector_congested_fraction": self.connector_congested_seconds
            / self.config.demand_seconds,
            "connector_peak_occupancy": float(self.connector_peak_occupancy),
            "guard_overrides": self.guard_overrides,
            "switches": sum(s.switches for s in self.signals.values()),
            "forced_switches": sum(s.forced_switches for s in self.signals.values()),
            "unfinished_wait_sum": sum(r["waitingTime"] for r in rows if not r["completed"]),
        }
        for origin in ORIGINS:
            summary[f"wait_{origin}"] = mean(
                [r["waitingTime"] for r in complete if r["origin"] == origin]
            )
        for group, origins in (("main", ("w", "e")), ("side", ORIGINS[2:])):
            summary[f"wait_{group}"] = mean(
                [r["waitingTime"] for r in complete if r["origin"] in origins]
            )
        summary["valid_complete"] = (
            not truncated
            and summary["pending_insertion"] == 0
            and running == 0
            and self.teleports == self.collisions == 0
            and len(complete) == self.planned
            and summary["abnormal_removals"] == 0
        )
        self.summary = summary
        save_json(self.episode_dir / "metrics.json", summary)
        write_rows(self.episode_dir / "trips.csv", rows)
        write_rows(self.episode_dir / "trace.csv", self.trace_rows)
