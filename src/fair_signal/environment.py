"""Gymnasium environment with explicit clearance rules and uncensored accounting."""

from __future__ import annotations

import csv
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import numpy as np
import traci
import traci.constants as tc
from sumolib.miscutils import getFreeSocketPort

from fair_signal.config import SCENARIOS, EnvConfig
from fair_signal.demand import DIRECTIONS, LANES, PHASE_LANES, generate_routes, save_json
from fair_signal.runtime import configure_sumo
from fair_signal.signals import Signal


def trip_metrics(path: Path, threshold: float) -> tuple[dict, list[dict]]:
    """TripInfo waitingTime is a lifetime total, unlike TraCI's rolling memory."""
    rows = []
    for trip in ET.parse(path).getroot().findall("tripinfo"):
        rows.append(
            {
                "vehicle": trip.get("id"),
                "direction": trip.get("id", "")[0],
                "completed": float(trip.get("arrival", "-1")) >= 0
                and trip.get("vaporized", "") in ("", "false", "0"),
                **{
                    key: float(trip.get(key, "0"))
                    for key in ("waitingTime", "duration", "departDelay", "timeLoss")
                },
            }
        )
    complete = [r for r in rows if r["completed"]]
    waits = [r["waitingTime"] for r in complete]
    mean = lambda values: float(np.mean(values)) if values else None  # noqa: E731
    summary = {
        "completed": len(complete),
        "mean_wait": mean(waits),
        "p95_wait": float(np.percentile(waits, 95)) if waits else None,
        "max_wait": max(waits) if waits else None,
        "over_threshold_fraction": mean([w > threshold for w in waits]),
        "mean_travel_time": mean([r["duration"] for r in complete]),
        "mean_depart_delay": mean([r["departDelay"] for r in complete]),
        "mean_system_time": mean([r["duration"] + r["departDelay"] for r in complete]),
        "mean_wait_including_insertion": mean(
            [r["waitingTime"] + r["departDelay"] for r in complete]
        ),
        "unfinished_wait_sum": sum(r["waitingTime"] for r in rows if not r["completed"]),
        "unfinished_wait_max": max(
            (r["waitingTime"] for r in rows if not r["completed"]), default=0
        ),
    }
    for direction in DIRECTIONS:
        summary[f"wait_{direction}"] = mean(
            [r["waitingTime"] for r in complete if r["direction"] == direction]
        )
    return summary, rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gui_arguments(delay_ms: int, start_paused: bool) -> list[str]:
    """Presentation options only; they never change the simulator time step."""
    if type(delay_ms) is not int or not 0 <= delay_ms <= 10000:
        raise ValueError("GUI delay must be an integer between 0 and 10000 milliseconds")
    arguments = ["--delay", str(delay_ms), "--quit-on-end"]
    if not start_paused:
        arguments.append("--start")
    return arguments


class TrafficEnv(gym.Env):
    """One-second simulator, five-second decisions; actions request green phases.

    This custom TraCI environment replaces the SUMO-RL wrapper for experiments
    because we need explicit max-green enforcement, a drain phase and trip totals.
    SUMO-RL remains available for the independent installation smoke test.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: EnvConfig,
        network: Path,
        output: Path,
        enhanced: bool = False,
        fair_weight: float = 0.0,
        route_file: Path | None = None,
        scenario: str | None = None,
        sumo_seed: int = 0,
        gui: bool = False,
        trace: bool = False,
        gui_delay: int = 200,
        start_paused: bool = True,
    ):
        super().__init__()
        config.validate()
        configure_sumo()
        self.config, self.network, self.output = config, network, output
        self.enhanced, self.fair_weight = enhanced, fair_weight
        self.route_file, self.scenario = route_file, scenario
        self.sumo_seed, self.gui, self.trace = sumo_seed, gui, trace
        self.gui_delay, self.start_paused = gui_delay, start_paused
        gui_arguments(gui_delay, start_paused)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(38 if enhanced else 22,), dtype=np.float32
        )
        self.connection = None
        self.gui_process: subprocess.Popen | None = None
        self.episode = -1
        self.summary: dict | None = None
        self.signal = Signal(config)
        self.queues = np.zeros(8)
        self.counts = np.zeros(8)
        self.head_waits = np.zeros(8)
        self.request_ages = np.zeros(8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.episode += 1
        self.episode_dir = self.output / f"episode_{self.episode:05d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        if self.route_file is None:
            scenario = self.scenario or str(self.np_random.choice(SCENARIOS))
            demand_seed = int(self.np_random.integers(0, 2**31 - 1))
            route_file = generate_routes(
                self.episode_dir, scenario, "train", demand_seed, self.config.demand_seconds
            )
            self.current_sumo_seed = demand_seed
        else:
            route_file = self.route_file
            self.current_sumo_seed = self.sumo_seed
        self.planned = len(ET.parse(route_file).getroot().findall("vehicle"))
        self.trip_file = self.episode_dir / "tripinfo.xml"
        cmd = [
            "sumo-gui" if self.gui else "sumo",
            "-n",
            str(self.network),
            "-r",
            str(route_file),
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
            cmd += gui_arguments(self.gui_delay, self.start_paused)
            if self.start_paused:
                print(
                    "SUMO GUI starts paused. Click Play (▶) in its toolbar to begin; "
                    "then Stop/Single Step controls remain available.",
                    flush=True,
                )
        label = f"fair-signal-{uuid.uuid4().hex}"
        if self.gui:
            # A paused GUI blocks TraCI commands, including startup and close.
            # Keep our process handle so Ctrl+C can still clean it up.
            port = getFreeSocketPort()
            self.gui_process = subprocess.Popen(cmd + ["--remote-port", str(port)])
            try:
                self.connection = traci.connect(port, proc=self.gui_process, waitBetweenRetries=0.1)
                self.connection.getVersion()
            except BaseException:
                self.close()
                raise
        else:
            traci.start(cmd, label=label, doSwitch=False, stdout=None)
            self.connection = traci.getConnection(label)
        self.signal = Signal(self.config)
        self.connection.trafficlight.setRedYellowGreenState("t", self.signal.state)
        self.capacity = np.array([self.connection.lane.getLength(lane) / 7.5 for lane in LANES])
        for lane in LANES:
            self.connection.lane.subscribe(
                lane,
                (
                    tc.LAST_STEP_VEHICLE_ID_LIST,
                    tc.LAST_STEP_VEHICLE_NUMBER,
                    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                ),
            )
        self.time = self.departed = self.arrived = self.teleports = self.collisions = 0
        self.arrived_in_window = 0
        self.queues, self.counts = np.zeros(8), np.zeros(8)
        self.head_waits, self.request_ages = np.zeros(8), np.zeros(8)
        self.total_waits: dict[str, float] = {}
        self.active: set[str] = set()
        self.trace_rows: list[dict] = []
        self.queue_sum = self.queue_peak = self.drain_queue_sum = self.drain_samples = 0
        self.summary = None
        return self._observation(), {}

    def _sample(self, served: tuple[int, ...]) -> None:
        conn = self.connection
        new_ids = conn.simulation.getDepartedIDList()
        arrived_ids = conn.simulation.getArrivedIDList()
        self.departed += len(new_ids)
        self.arrived += len(arrived_ids)
        self.teleports += conn.simulation.getStartingTeleportNumber()
        self.collisions += conn.simulation.getCollidingVehiclesNumber()
        self.active.update(new_ids)
        self.active.difference_update(arrived_ids)
        for vehicle in new_ids:
            conn.vehicle.subscribe(vehicle, (tc.VAR_SPEED, tc.VAR_LANEPOSITION))
        vehicle_data = conn.vehicle.getAllSubscriptionResults()
        for vehicle, values in vehicle_data.items():
            self.total_waits[vehicle] = self.total_waits.get(vehicle, 0.0) + (
                values[tc.VAR_SPEED] <= 0.1
            )
        lane_data = conn.lane.getAllSubscriptionResults()
        for i, lane in enumerate(LANES):
            data = lane_data[lane]
            self.queues[i] = data[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
            self.counts[i] = data[tc.LAST_STEP_VEHICLE_NUMBER]
            stopped = [
                v
                for v in data[tc.LAST_STEP_VEHICLE_ID_LIST]
                if v in vehicle_data and vehicle_data[v][tc.VAR_SPEED] <= 0.1
            ]
            head = max(stopped, key=lambda v: vehicle_data[v][tc.VAR_LANEPOSITION], default=None)
            self.head_waits[i] = self.total_waits[head] if head is not None else 0.0
            self.request_ages[i] = (
                self.request_ages[i] + 1 if self.counts[i] > 0 and i not in served else 0
            )
        queue = float(self.queues.sum())
        if self.time <= self.config.demand_seconds:
            self.arrived_in_window = self.arrived
            self.queue_sum += queue
            self.queue_peak = max(self.queue_peak, queue)
        else:
            self.drain_queue_sum += queue
            self.drain_samples += 1

    def _observation(self) -> np.ndarray:
        # During clearance, phase is the old phase; target and mode expose the transition.
        phase = np.eye(4)[self.signal.target if self.signal.mode != "green" else self.signal.phase]
        base = np.concatenate(
            (
                phase,
                [self.signal.elapsed / self.config.max_green, float(self.signal.mode != "green")],
                self.counts / self.capacity,
                self.queues / self.capacity,
            )
        )
        if self.enhanced:
            base = np.concatenate(
                (
                    base,
                    self.head_waits / self.config.wait_scale,
                    self.request_ages / self.config.wait_scale,
                )
            )
        return np.clip(base, 0, 1).astype(np.float32)

    def step(self, action):
        if self.connection is None or self.summary is not None:
            raise RuntimeError("reset() is required before stepping a finished environment")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid phase action: {action}")
        self.signal.request(int(action))
        rewards, queue_costs, wait_costs = [], [], []
        terminated = truncated = False
        for _ in range(self.config.decision_seconds):
            state = self.signal.before_second()
            self.connection.trafficlight.setRedYellowGreenState("t", state)
            served = PHASE_LANES[self.signal.phase] if self.signal.mode == "green" else ()
            self.connection.simulationStep()
            self.time += 1
            self._sample(served)
            queue_cost = float(self.queues.sum() / self.config.queue_scale)
            wait_cost = float(
                np.maximum(self.head_waits - self.config.wait_threshold, 0).sum()
                / (8 * self.config.wait_scale)
            )
            queue_costs.append(queue_cost)
            wait_costs.append(wait_cost)
            rewards.append(-queue_cost - self.fair_weight * wait_cost)
            if self.trace:
                self.trace_rows.append(
                    {
                        "second": self.time,
                        "phase": self.signal.phase,
                        "mode": self.signal.mode,
                        "target": self.signal.target,
                        "green_elapsed": self.signal.elapsed,
                        "queue": float(self.queues.sum()),
                        "head_wait_max": float(self.head_waits.max()),
                        "request_age_max": float(self.request_ages.max()),
                        "requested_action": int(action),
                    }
                )
            self.signal.after_second()
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
        observation = self._observation()
        info = {"queue_cost": float(np.mean(queue_costs)), "wait_cost": float(np.mean(wait_costs))}
        if terminated or truncated:
            self._finish(truncated)
            info["metrics"] = self.summary
        return observation, float(np.mean(rewards)), terminated, truncated, info

    def _finish(self, truncated: bool) -> None:
        conn = self.connection
        running = conn.vehicle.getIDCount()
        pending = self.planned - self.departed
        # Closing flushes unfinished TripInfo. Never classify them as completed trips.
        self.close()
        summary, trips = trip_metrics(self.trip_file, self.config.wait_threshold)
        summary.update(
            {
                "planned": self.planned,
                "departed": self.departed,
                "arrived": self.arrived,
                "throughput_demand_window": self.arrived_in_window,
                "running": running,
                "pending_insertion": pending,
                "abnormal_removals": self.departed - self.arrived - running,
                "teleport_events": self.teleports,
                "collision_events": self.collisions,
                "truncated": truncated,
                "simulated_seconds": self.time,
                "drain_seconds": max(0, self.time - self.config.demand_seconds),
                "mean_queue": self.queue_sum / self.config.demand_seconds,
                "peak_queue": self.queue_peak,
                "drain_mean_queue": self.drain_queue_sum / self.drain_samples
                if self.drain_samples
                else 0.0,
                "switches": self.signal.switches,
                "forced_switches": self.signal.forced_switches,
                "blocked_requests": self.signal.blocked_requests,
            }
        )
        summary["valid_complete"] = (
            not truncated
            and pending == 0
            and running == 0
            and self.teleports == 0
            and self.collisions == 0
            and summary["completed"] == self.planned
            and summary["abnormal_removals"] == 0
        )
        self.summary = summary
        save_json(self.episode_dir / "metrics.json", summary)
        write_rows(self.episode_dir / "trips.csv", trips)
        write_rows(self.episode_dir / "trace.csv", self.trace_rows)

    def close(self):
        if self.gui_process is not None and sys.exc_info()[0] is not None:
            self._stop_gui_process()
        if self.connection is not None:
            try:
                self.connection.close()
            except (traci.FatalTraCIError, OSError):
                if not self.gui:
                    raise
            finally:
                self.connection = None
        self._stop_gui_process()

    def _stop_gui_process(self) -> None:
        if self.gui_process is not None:
            process, self.gui_process = self.gui_process, None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
