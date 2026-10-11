"""Strict quality gates, model-seed summaries and paired comparisons for the arterial study."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from arterial_study import LEARNED, TRADITIONAL, candidates, source_digest
from sumolib.net import readNet
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fair_signal.arterial.validation import audit_signal_rows  # noqa: E402
from fair_signal.config import EnvConfig  # noqa: E402
from fair_signal.demand import save_json, sha256  # noqa: E402
from fair_signal.runtime import output_directory, project_root  # noqa: E402

METRICS = (
    "mean_wait",
    "p95_wait",
    "over_threshold_fraction",
    "mean_system_time",
    "mean_stops",
    "mean_wait_including_insertion",
    "mean_depart_delay",
    "mean_queue",
    "mean_pending",
    "wait_main",
    "wait_side",
    "connector_mean_queue",
    "connector_congested_fraction",
    "guard_overrides",
    "switches",
    "cross_mean_wait",
    "cross_mean_system_time",
    "cross_mean_stops",
)
LABELS = {
    "coordinated_fixed": "Coordinated\nfixed",
    "actuated": "Actuated",
    "aging": "Aging",
    "max_pressure": "Max\npressure",
    "local_dqn": "Local-Q\nDQN",
    "joint_dqn": "Joint\nDQN",
    "fair_joint_dqn": "Fair joint\nDQN",
}


def summarize(output):
    saved = json.loads((output / "study.json").read_text())
    if saved["source_sha256"] != source_digest() or saved["test_evaluated"]:
        raise ValueError("Study code changed or test isolation violated")
    config = saved["config"]
    frame = pd.read_csv(output / "validation.csv")
    frame["training_seed"] = frame.training_seed.fillna(-1).astype(int)
    expected = {
        (m, seed, s, t)
        for m in config["methods"]
        for seed in (config["training_seeds"] if m in LEARNED else [-1])
        for s in config["scenarios"]
        for t in config["validation_seeds"]
    }
    actual = list(
        frame[["method", "training_seed", "scenario", "traffic_seed"]].itertuples(
            index=False, name=None
        )
    )
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Validation matrix is incomplete/duplicated")
    if (output / "cases/test").exists() or (output / "evaluation/test").exists():
        raise ValueError("Reserved test artifacts exist")
    tuning = pd.read_csv(output / "tuning.csv")
    tune_expected = {
        (m, i, s, t)
        for m in TRADITIONAL
        for i in range(len(candidates(m)))
        for s in config["scenarios"]
        for t in config["tuning_seeds"]
    }
    tune_actual = list(
        tuning[["method", "candidate", "scenario", "traffic_seed"]].itertuples(
            index=False, name=None
        )
    )
    if len(tune_actual) != len(set(tune_actual)) or set(tune_actual) != tune_expected:
        raise ValueError("Tuning matrix is incomplete/duplicated")
    selection = json.loads((output / "selection.json").read_text())
    for method in TRADITIONAL:
        rank = []
        for index, parameters in enumerate(candidates(method)):
            part = tuning[(tuning.method == method) & (tuning.candidate == index)]
            if part.valid_complete.all():
                rank.append(
                    (float((part.mean_wait + 0.5 * part.p95_wait).mean()), index, parameters)
                )
        score, index, params = min(rank, key=lambda r: (r[0], r[1]))
        chosen = selection[method]
        if (
            index != chosen["candidate"]
            or params != chosen["parameters"]
            or not np.isclose(score, chosen["tuning_score"])
        ):
            raise ValueError("Baseline selection does not reproduce")
    traces_checked = intervals = 0
    hashes = {}
    cohorts = []
    for split, cases in (("tuning", tuning), ("validation", frame)):
        for row in cases.itertuples():
            seed = (
                None
                if pd.isna(row.training_seed) or row.training_seed == -1
                else int(row.training_seed)
            )
            directory = (
                output
                / f"evaluation/{split}/{row.method}/candidate_{row.candidate}"
                / f"seed_{seed}/{row.scenario}/{row.traffic_seed}"
            )
            identity = json.loads((directory / "identity.json").read_text())
            route = output / f"cases/{split}/{row.scenario}/{row.traffic_seed}/vehicles.rou.xml"
            route_hash = sha256(route)
            if identity["source_sha256"] != source_digest() or row.route_sha256 != route_hash:
                raise ValueError("Case source/route checksum mismatch")
            if identity["route_sha256"] != route_hash or identity["config"] != config:
                raise ValueError("Case identity changed")
            if identity["network_sha256"] != sha256(output / "network/arterial.net.xml"):
                raise ValueError("Case network changed")
            hashes[f"{split}/{row.scenario}/{row.traffic_seed}"] = route_hash
            if (
                row.planned != row.departed + row.pending_insertion
                or row.departed != row.arrived + row.running + row.abnormal_removals
            ):
                raise ValueError("Fleet accounting mismatch")
            trace = pd.read_csv(directory / "episode_00000/trace.csv")
            if len(trace) != row.simulated_seconds or not np.array_equal(
                trace.second, np.arange(1, len(trace) + 1)
            ):
                raise ValueError("Non-contiguous simulation trace")
            env_config = EnvConfig(**config["env"])
            limits = {
                "green": env_config.max_green,
                "yellow": env_config.yellow_seconds,
                "all_red": env_config.all_red_seconds,
            }
            for junction in ("a", "b"):
                upper = trace[f"{junction}_mode"].map(limits)
                elapsed = trace[f"{junction}_elapsed"]
                if upper.isna().any() or (elapsed < 0).any() or (elapsed >= upper).any():
                    raise ValueError("Invalid physical timer, including final partial intervals")
            demand = trace.iloc[: config["env"]["demand_seconds"]]
            if not np.isclose(demand.queue.mean(), row.mean_queue):
                raise ValueError("Demand-window queue does not reproduce from trace")
            if not np.isclose(demand.pending.mean(), row.mean_pending):
                raise ValueError("Pending insertion does not reproduce from trace")
            decision_rows = trace.iloc[:: config["env"].get("decision_seconds", 5)]
            if decision_rows.guard_overrides.sum() != row.guard_overrides:
                raise ValueError("Guard intervention counter does not reproduce")
            trips = ET.parse(directory / "episode_00000/tripinfo.xml").getroot().findall("tripinfo")
            if sum(float(t.get("waitingTime")) for t in trips) != trace.queue.sum():
                raise ValueError("Full-network waiting integral differs from TripInfo")
            if split == "validation":
                cross_ids = {
                    vehicle.get("id")
                    for vehicle in ET.parse(route).getroot().findall("vehicle")
                    if len(vehicle.find("route").get("edges").split()) == 3
                }
                complete_cross = [
                    trip
                    for trip in trips
                    if trip.get("id") in cross_ids
                    and float(trip.get("arrival", "-1")) >= 0
                    and trip.get("vaporized", "") in ("", "false", "0")
                ]
                cohort = {
                    "method": row.method,
                    "training_seed": -1 if seed is None else seed,
                    "scenario": row.scenario,
                    "traffic_seed": row.traffic_seed,
                    "cross_planned": len(cross_ids),
                    "cross_completed": len(complete_cross),
                }
                for metric, fields in (
                    ("cross_mean_wait", ("waitingTime",)),
                    ("cross_mean_system_time", ("duration", "departDelay")),
                    ("cross_mean_stops", ("waitingCount",)),
                ):
                    cohort[metric] = (
                        float(
                            np.mean(
                                [
                                    sum(float(trip.get(field)) for field in fields)
                                    for trip in complete_cross
                                ]
                            )
                        )
                        if complete_cross
                        else None
                    )
                cohorts.append(cohort)
            intervals += audit_signal_rows(trace.to_dict("records"), env_config)
            traces_checked += 1
    frame = frame.merge(
        pd.DataFrame(cohorts),
        on=["method", "training_seed", "scenario", "traffic_seed"],
        validate="one_to_one",
    )
    events = []
    training = []
    for method in LEARNED:
        for seed in config["training_seeds"]:
            directory = output / f"models/{method}/seed_{seed}"
            report = json.loads((directory / "training.json").read_text())
            if report["source_sha256"] != source_digest() or report["model_sha256"] != sha256(
                directory / "model.zip"
            ):
                raise ValueError("Training/model identity mismatch")
            if (
                report["decision_steps"] != config["training_steps"]
                or report["gradient_updates"]
                != (config["training_steps"] - config["learning_starts"]) // config["train_freq"]
            ):
                raise ValueError("Unexpected training/update budget")
            if not report["weights_changed"] or not report["save_reload_action_matches"]:
                raise ValueError("Training/reload check failed")
            validation_hashes = frame[
                (frame.method == method) & (frame.training_seed == seed)
            ].model_sha256
            if not (validation_hashes == report["model_sha256"]).all():
                raise ValueError("Evaluated weights do not match frozen model")
            event_count = 0
            tags = set()
            for event in (directory / "tensorboard").rglob("events.out.tfevents.*"):
                accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0}).Reload()
                for tag in accumulator.Tags()["scalars"]:
                    values = accumulator.Scalars(tag)
                    if any(not np.isfinite(v.value) for v in values):
                        raise ValueError("Nonfinite TensorBoard scalar")
                    event_count += len(values)
                    tags.add(tag)
            if "train/loss" not in tags:
                raise ValueError("Missing DQN loss history")
            events.append(
                {
                    "method": method,
                    "training_seed": seed,
                    "finite_scalars": event_count,
                    "tags": sorted(tags),
                }
            )
            training.append(report)
    summaries = []
    per_seed = []
    for scope in ["all", *config["scenarios"]]:
        subset = frame if scope == "all" else frame[frame.scenario == scope]
        for method in config["methods"]:
            part = subset[subset.method == method]
            valid = bool(part.valid_complete.all())
            summary = {
                "scenario": scope,
                "method": method,
                "cases": len(part),
                "complete_cases": int(part.valid_complete.sum()),
                "eligible": valid,
            }
            seed_means = part.groupby("training_seed")[list(METRICS)].mean()
            for metric in METRICS:
                summary[metric] = float(seed_means[metric].mean()) if valid else None
                summary[f"{metric}_seed_sd"] = (
                    float(seed_means[metric].std(ddof=1)) if valid and len(seed_means) > 1 else None
                )
            summaries.append(summary)
            for seed, values in seed_means.iterrows():
                per_seed.append(
                    {
                        "scenario": scope,
                        "method": method,
                        "training_seed": int(seed),
                        "eligible": valid,
                        **{k: float(values[k]) if valid else None for k in METRICS},
                    }
                )
    summary = pd.DataFrame(summaries)
    paired = []
    for method in LEARNED:
        for reference in (
            "actuated",
            "max_pressure",
            "coordinated_fixed",
            "local_dqn",
            "joint_dqn",
        ):
            if method == reference:
                continue
            left = frame[frame.method == method]
            right = frame[frame.method == reference]
            keys = ["scenario", "traffic_seed"] + (
                ["training_seed"] if reference in LEARNED else []
            )
            joined = left.merge(
                right,
                on=keys,
                suffixes=("", "_reference"),
                validate="one_to_one" if reference in LEARNED else "many_to_one",
            )
            for scope in ["all", *config["scenarios"]]:
                part = joined if scope == "all" else joined[joined.scenario == scope]
                valid = bool(part.valid_complete.all() and part.valid_complete_reference.all())
                row = {
                    "method": method,
                    "reference": reference,
                    "scenario": scope,
                    "eligible": valid,
                }
                for metric in METRICS:
                    delta = part[metric] - part[f"{metric}_reference"]
                    row[f"delta_{metric}"] = float(delta.mean()) if valid else None
                paired.append(row)
    destination = project_root() / "docs/results" / output.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("study.json", "provenance.json", "execution_all.json"):
        if (output / name).exists():
            shutil.copyfile(output / name, destination / name)
    frame.to_csv(destination / "validation.csv", index=False)
    tuning.to_csv(destination / "tuning.csv", index=False)
    summary.to_csv(destination / "summary.csv", index=False)
    pd.DataFrame(per_seed).to_csv(destination / "per_seed.csv", index=False)
    pd.DataFrame(paired).to_csv(destination / "paired_differences.csv", index=False)
    pd.DataFrame(training).to_csv(destination / "training.csv", index=False)
    save_json(destination / "selection.json", selection)
    quality = {
        "source_sha256": source_digest(),
        "summary_script_sha256": sha256(Path(__file__)),
        "validation_cases": len(frame),
        "tuning_cases": len(tuning),
        "complete_validation_cases": int(frame.valid_complete.sum()),
        "complete_tuning_cases": int(tuning.valid_complete.sum()),
        "traces_audited": traces_checked,
        "physical_intervals_audited": intervals,
        "running": int(frame.running.sum()),
        "pending_insertion": int(frame.pending_insertion.sum()),
        "collision_events": int(frame.collision_events.sum()),
        "teleport_events": int(frame.teleport_events.sum()),
        "test_generated_or_evaluated": False,
        "route_sha256": hashes,
        "tensorboard": events,
        "input_directory": str(output.relative_to(project_root())),
        "inputs": {
            name: sha256(output / name)
            for name in (
                "study.json",
                "validation.csv",
                "tuning.csv",
                "selection.json",
                "training.csv",
            )
        },
    }
    save_json(destination / "quality.json", quality)
    global_rows = summary[summary.scenario == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, metric, title, multiplier in zip(
        axes,
        ("mean_wait", "mean_system_time", "over_threshold_fraction"),
        ("Mean stop wait (s)", "End-to-end system time (s)", "Stop wait > 90s (%)"),
        (1, 1, 100),
        strict=True,
    ):
        positions = np.arange(len(global_rows))
        vals = global_rows[metric].to_numpy(dtype=float) * multiplier
        errors = global_rows[f"{metric}_seed_sd"].astype(float).fillna(0).to_numpy() * multiplier
        axis.bar(
            positions,
            vals,
            yerr=errors,
            capsize=3,
            color=["#458b74"] * 4 + ["#7896c6", "#4468a2", "#b65b69"],
        )
        for index, eligible in enumerate(global_rows.eligible):
            if not eligible:
                axis.text(index, 0, "incomplete", rotation=90, ha="center", va="bottom")
        axis.set_xticks(positions, [LABELS[m] for m in global_rows.method], rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Two-junction development validation; all controllers guarded\n"
        "Error bars: SD of model-seed means; traditional bars have no model-seed error"
    )
    fig.tight_layout()
    fig.savefig(destination / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, color in zip(LEARNED, ("#7896c6", "#4468a2", "#b65b69"), strict=True):
        for index, seed in enumerate(config["training_seeds"]):
            path = output / f"models/{method}/seed_{seed}/learning.csv"
            if not path.exists():
                continue
            episodes = pd.read_csv(path)
            for axis, metric in zip(axes, ("valid_complete", "mean_queue"), strict=True):
                values = episodes[metric]
                if metric == "mean_queue":
                    values = values + episodes.mean_pending
                axis.plot(
                    episodes.step,
                    values.rolling(10, min_periods=1).mean(),
                    color=color,
                    linestyle=("-", "--", ":")[index % 3],
                    alpha=0.8,
                    label=LABELS[method].replace("\n", " ") if index == 0 else None,
                )
    axes[0].set_ylabel("Complete episode fraction (rolling 10)")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Stopped + pending (rolling 10 episodes)")
    for axis in axes:
        axis.set_xlabel("Joint decision steps")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Training diagnostics; line styles distinguish model seeds")
    fig.tight_layout()
    fig.savefig(destination / "training.png", dpi=160)
    plt.close(fig)
    network = readNet(str(output / "network/arterial.net.xml"))
    fig, axis = plt.subplots(figsize=(9, 5))
    for edge in network.getEdges():
        xy = np.asarray(edge.getShape())
        axis.plot(xy[:, 0], xy[:, 1], color="#aaaaaa", linewidth=3, zorder=1)
    for edge_id in ("w_a", "a_b", "b_e"):
        xy = np.asarray(network.getEdge(edge_id).getShape())
        axis.plot(xy[:, 0], xy[:, 1], color="#4468a2", linewidth=2, zorder=2)
    for node in network.getNodes():
        x, y = node.getCoord()
        signal = node.getID() in ("a", "b")
        axis.scatter(
            [x], [y], color="#b65b69" if signal else "#555555", s=65 if signal else 20, zorder=3
        )
        axis.annotate(node.getID().upper(), (x, y), xytext=(8, 8), textcoords="offset points")
    x_a, y_a = network.getNode("a").getCoord()
    x_b, _ = network.getNode("b").getCoord()
    axis.annotate("", (x_a, y_a - 100), (x_b, y_a - 100), arrowprops={"arrowstyle": "<->"})
    axis.text((x_a + x_b) / 2, y_a - 90, "350 m", ha="center")
    axis.set_title("Original two-junction network; blue: an example cross-junction route")
    axis.set_aspect("equal")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(destination / "network.png", dpi=160)
    plt.close(fig)
    lines = [
        "# 两路口开发验证汇总",
        "",
        "全部方法共享服务护栏；数值为同一批开发车流的平均值。",
        "",
        "| 方法 | 完整回合 | 平均等待(s) | P95(s) | 超90秒(%) | 行程含插入(s) | 停车次数 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in global_rows.itertuples():
        values = [
            f"{getattr(row, k) * factor:.3f}" if row.eligible else "不排名"
            for k, factor in (
                ("mean_wait", 1),
                ("p95_wait", 1),
                ("over_threshold_fraction", 100),
                ("mean_system_time", 1),
                ("mean_stops", 1),
            )
        ]
        lines.append(
            f"| {row.method} | {row.complete_cases}/{row.cases} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "详细场景、模型种子波动和配对差值见同目录 CSV；未使用最终测试。",
        "",
        "![Comparison](comparison.png)",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines))
    quality["exported_files"] = {
        name: sha256(destination / name)
        for name in (
            "validation.csv",
            "tuning.csv",
            "summary.csv",
            "per_seed.csv",
            "paired_differences.csv",
            "training.csv",
            "selection.json",
            "study.json",
            "provenance.json",
            "execution_all.json",
            "comparison.png",
            "training.png",
            "network.png",
            "report.md",
        )
        if (destination / name).exists()
    }
    save_json(destination / "quality.json", quality)
    print(
        global_rows[
            [
                "method",
                "complete_cases",
                "cases",
                "mean_wait",
                "p95_wait",
                "over_threshold_fraction",
                "mean_system_time",
                "mean_stops",
            ]
        ].to_string(index=False)
    )
    print(
        f"Audited {traces_checked} traces / {intervals} physical intervals; exported {destination}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/arterial-50k")
    args = parser.parse_args()
    summarize(output_directory(args.output))
