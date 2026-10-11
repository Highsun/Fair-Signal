"""Replay frozen control unchanged and inspect terminal vehicles before SUMO closes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from arterial_study import LEARNED, TRADITIONAL, source_digest
from view_arterial import replay

from fair_signal.arterial.demand import INCOMING, JUNCTIONS
from fair_signal.arterial.environment import ArterialEnv
from fair_signal.demand import GREEN_STATES, PHASE_LANES, save_json, sha256
from fair_signal.runtime import output_directory, project_root


class TerminalSnapshotEnv(ArterialEnv):
    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.demand_rows = []
        return result

    def _sample_network(self, served):
        before = {j: self.counts[j].copy() for j in JUNCTIONS}
        result = super()._sample_network(served)
        if self.time > self.config.demand_seconds:
            for junction in JUNCTIONS:
                signal = self.signals[junction]
                lanes = list(PHASE_LANES[signal.phase])
                self.demand_rows.append(
                    {
                        "second": self.time,
                        "junction": junction,
                        "phase": signal.phase,
                        "mode": signal.mode,
                        "count_before": float(before[junction][lanes].sum()),
                        "count_after": float(self.counts[junction][lanes].sum()),
                    }
                )
        return result

    def _finish(self, truncated):
        connection = self.connection
        fleet = []
        for vehicle in connection.vehicle.getIDList():
            fleet.append(
                {
                    "vehicle": vehicle,
                    "lane": connection.vehicle.getLaneID(vehicle),
                    "position": connection.vehicle.getLanePosition(vehicle),
                    "speed": connection.vehicle.getSpeed(vehicle),
                    "route": connection.vehicle.getRoute(vehicle),
                    "route_index": connection.vehicle.getRouteIndex(vehicle),
                    "next_signals": connection.vehicle.getNextTLS(vehicle),
                    "best_lanes": connection.vehicle.getBestLanes(vehicle),
                    "lifetime_stop": self.total_waits.get(vehicle, 0),
                }
            )
        snapshot = {
            "second": self.time,
            "truncated": truncated,
            "vehicles": fleet,
            "signals": {
                j: {
                    "phase": self.signals[j].phase,
                    "mode": self.signals[j].mode,
                    "target": self.signals[j].target,
                    "elapsed": self.signals[j].elapsed,
                    "lanes": list(INCOMING[j]),
                    "counts": self.counts[j].tolist(),
                    "queues": self.queues[j].tolist(),
                    "request_ages": self.ages[j].tolist(),
                }
                for j in JUNCTIONS
            },
        }
        save_json(self.episode_dir / "terminal_snapshot.json", snapshot)
        pd.DataFrame(self.demand_rows).to_csv(self.episode_dir / "phase_demand.csv", index=False)
        super()._finish(truncated)


def export_diagnostic(output, method, seed, scenario, traffic):
    episode = output / f"viewer/{method}/{seed}/{scenario}/{traffic}/episode_00000"
    destination = (
        project_root()
        / f"docs/results/{output.name}/diagnostic-{method}-{seed}-{scenario}-{traffic}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((episode / "terminal_snapshot.json").read_text())
    for vehicle in snapshot["vehicles"]:
        vehicle["needed_phases"] = [
            {
                "junction": signal,
                "link_index": index,
                "phase": next(p for p, state in enumerate(GREEN_STATES) if state[index] == "G"),
            }
            for signal, index, _, _ in vehicle["next_signals"]
        ]
    save_json(destination / "terminal_snapshot.json", snapshot)
    frame = pd.read_csv(episode / "trace.csv")
    demand = pd.read_csv(episode / "phase_demand.csv")
    config = json.loads((output / "study.json").read_text())["config"]
    decisions = frame.iloc[:: config["env"].get("decision_seconds", 5)]
    rows = []
    for seconds in (900, 600, 300):
        tail = frame[frame.second > frame.second.max() - seconds]
        actions = decisions[decisions.second > frame.second.max() - seconds]
        for junction in JUNCTIONS:
            for phase in range(4):
                selected = demand[
                    (demand.second > frame.second.max() - seconds)
                    & (demand.junction == junction)
                    & (demand.phase == phase)
                    & (demand["mode"] == "green")
                ]
                rows.append(
                    {
                        "last_seconds": seconds,
                        "junction": junction,
                        "phase": phase,
                        "green_seconds": int(
                            (
                                (tail[f"{junction}_mode"] == "green")
                                & (tail[f"{junction}_phase"] == phase)
                            ).sum()
                        ),
                        "proposed_decisions": int((actions[f"{junction}_proposed"] == phase).sum()),
                        "executed_decisions": int((actions[f"{junction}_executed"] == phase).sum()),
                        "empty_green_seconds": int(
                            ((selected.count_before == 0) & (selected.count_after == 0)).sum()
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(destination / "phase_allocation.csv", index=False)
    shutil.copyfile(episode / "metrics.json", destination / "metrics.json")
    save_json(
        destination / "identity.json",
        {
            "method": method,
            "training_seed": seed,
            "scenario": scenario,
            "traffic_seed": traffic,
            "source_sha256": source_digest(),
            "diagnostic_script_sha256": sha256(Path(__file__)),
            "viewer_script_sha256": sha256(project_root() / "scripts/view_arterial.py"),
            "replay_all_metrics_equal": True,
            "simulation_limit_unchanged": True,
            "model_sha256": sha256(output / f"models/{method}/seed_{seed}/model.zip")
            if method in LEARNED
            else None,
            "route_sha256": sha256(
                output / f"cases/validation/{scenario}/{traffic}/vehicles.rou.xml"
            ),
        },
    )
    print(f"Exported unchanged-control terminal diagnostic: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/arterial-50k")
    parser.add_argument("--method", choices=LEARNED + TRADITIONAL, default="joint_dqn")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--scenario", default="downstream_slowdown")
    parser.add_argument("--traffic", type=int, default=2003)
    args = parser.parse_args()
    replay(
        output_directory(args.output),
        args.method,
        args.seed,
        args.scenario,
        args.traffic,
        0,
        headless=True,
        env_class=TerminalSnapshotEnv,
    )
    export_diagnostic(
        output_directory(args.output), args.method, args.seed, args.scenario, args.traffic
    )
