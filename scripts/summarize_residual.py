"""Audit the complete residual study and export compact, failure-aware results."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from residual_study import (
    BASELINES,
    LEARNED,
    METHODS,
    case_directory,
    historical_digest,
    reference_result,
    source_digest,
)
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
    "wait_main",
    "wait_side",
    "guard_overrides",
    "correction_effective",
    "correction_rate",
    "extension_seconds_charged",
    "cross_mean_wait",
    "cross_mean_system_time",
    "cross_mean_stops",
)
LABELS = {
    "actuated": "Actuated",
    "max_pressure": "Max pressure",
    "aging": "Aging",
    "always_extend": "Always extend",
    "always_urgent": "Always urgent",
    "residual_dqn": "Residual DQN",
    "fair_residual_dqn": "Fair residual DQN",
}


def audit_case(directory, row, config):
    env_config = EnvConfig(**config["env"])
    trace = pd.read_csv(directory / "episode_00000/trace.csv")
    decisions = pd.read_csv(directory / "episode_00000/corrections.csv")
    if not np.array_equal(trace.second, np.arange(1, row.simulated_seconds + 1)):
        raise ValueError("Noncontiguous physical trace")
    interval = env_config.decision_seconds
    expected = {(t, j) for t in range(0, len(trace), interval) for j in ("a", "b")}
    actual = list(decisions[["second", "junction"]].itertuples(index=False, name=None))
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Correction decision matrix incomplete/duplicated")
    for j in ("a", "b"):
        upper = trace[f"{j}_mode"].map(
            {
                "green": env_config.max_green,
                "yellow": env_config.yellow_seconds,
                "all_red": env_config.all_red_seconds,
            }
        )
        if (
            upper.isna().any()
            or (trace[f"{j}_elapsed"] >= upper).any()
            or (trace[f"{j}_elapsed"] < 0).any()
        ):
            raise ValueError("Invalid physical interval timer")
        part = decisions[decisions.junction == j].sort_values("second")
        first = trace.iloc[::interval]
        if not np.array_equal(part.proposed, first[f"{j}_proposed"]) or not np.array_equal(
            part.guarded_proposed, first[f"{j}_executed"]
        ):
            raise ValueError("Decision requests differ from physical trace")
        yellow = trace[f"{j}_mode"] == "yellow"
        switches = (yellow & ~yellow.shift(fill_value=False)).cumsum()
        before = [0 if t == 0 else switches.iloc[t - 1] for t in part.second]
        if not np.array_equal(part.switches_before, before):
            raise ValueError("Correction budget reset does not match physical switches")
    charged = decisions.groupby(["junction", "switches_before"]).extension_charged.cumsum()
    if (
        not np.array_equal(charged, decisions.extension_used)
        or (decisions.extension_used > config["extension_seconds"]).any()
    ):
        raise ValueError("Extension budget exceeded or reset within a green interval")
    if not decisions.extension_charged.isin([0, interval]).all():
        raise ValueError("Invalid extension charge")
    if not decisions.choice.isin([0, 1, 2]).all():
        raise ValueError("Invalid correction action")
    if not np.array_equal(
        decisions.changed, (decisions.proposed != decisions.baseline).astype(int)
    ):
        raise ValueError("Correction counter mismatch")
    if not np.array_equal(
        decisions.effective, (decisions.guarded_proposed != decisions.guarded_baseline).astype(int)
    ):
        raise ValueError("Effective correction counter mismatch")
    counts = {
        "correction_node_decisions": len(decisions),
        "correction_changes": decisions.changed.sum(),
        "correction_effective": decisions.effective.sum(),
        "extension_seconds_charged": decisions.extension_charged.sum(),
        **{f"choice_{i}": int((decisions.choice == i).sum()) for i in range(3)},
    }
    if any(getattr(row, k) != value for k, value in counts.items()):
        raise ValueError("Episode correction totals differ from decisions")
    if trace.iloc[::interval].guard_overrides.sum() != row.guard_overrides:
        raise ValueError("Guard total differs from trace")
    demand = trace.iloc[: env_config.demand_seconds]
    if not np.isclose(demand.queue.mean(), row.mean_queue) or not np.isclose(
        demand.pending.mean(), row.mean_pending
    ):
        raise ValueError("Queue/pending statistics differ from trace")
    trips = ET.parse(directory / "episode_00000/tripinfo.xml").getroot().findall("tripinfo")
    if sum(float(t.get("waitingTime")) for t in trips) != trace.queue.sum():
        raise ValueError("Network stop integral differs from TripInfo")
    if row.planned != row.departed + row.pending_insertion or (
        row.departed != row.arrived + row.running + row.abnormal_removals
    ):
        raise ValueError("Vehicle accounting mismatch")
    return audit_signal_rows(trace.to_dict("records"), env_config), trips


def summarize(output):
    saved = json.loads((output / "study.json").read_text())
    if saved["source_sha256"] != source_digest() or saved["test_evaluated"]:
        raise ValueError("Source changed or test isolation violated")
    if saved["historical_core_sha256"] != historical_digest():
        raise ValueError("Historical core changed")
    config = saved["config"]
    reference = output_directory(config["reference"])
    for name, digest in saved["reference_hashes"].items():
        if sha256(reference / name) != digest:
            raise ValueError("Historical reference changed")
    for directory in (output, reference):
        if (directory / "cases/test").exists() or (directory / "evaluation/test").exists():
            raise ValueError("Reserved test artifacts exist")
    if sha256(output / "selection.json") != saved["reference_hashes"]["selection.json"]:
        raise ValueError("Baseline selection changed")
    frame = pd.read_csv(output / "validation.csv")
    frame["training_seed"] = frame.training_seed.fillna(-1).astype(int)
    expected = {
        (m, seed, s, t)
        for m in METHODS
        for seed in (config["training_seeds"] if m in LEARNED else [-1])
        for s in config["scenarios"]
        for t in config["validation_seeds"]
    }
    actual = list(
        frame[["method", "training_seed", "scenario", "traffic_seed"]].itertuples(
            index=False, name=None
        )
    )
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Validation matrix incomplete/duplicated")
    intervals = reference_matches = 0
    inputs = {}
    cohorts = []
    network_hash = sha256(output / "network/arterial.net.xml")
    if network_hash != sha256(reference / "network/arterial.net.xml"):
        raise ValueError("Reference network changed")
    for row in frame.itertuples():
        seed = None if row.training_seed == -1 else int(row.training_seed)
        directory = case_directory(output, row.method, seed, row.scenario, row.traffic_seed)
        identity = json.loads((directory / "identity.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        route = output / f"cases/validation/{row.scenario}/{row.traffic_seed}/vehicles.rou.xml"
        relative_route = route.relative_to(output)
        if sha256(route) != sha256(reference / relative_route):
            raise ValueError("Reference route changed")
        model = output / f"models/{row.method}/seed_{seed}/model.zip"
        model_hash = sha256(model) if row.method in LEARNED else None
        expected_identity = {
            "source_sha256": source_digest(),
            "config": config,
            "model_sha256": model_hash,
            "route_sha256": sha256(route),
            "network_sha256": network_hash,
            "selection_sha256": sha256(output / "selection.json"),
        }
        if identity != expected_identity or row.route_sha256 != sha256(route):
            raise ValueError("Case source/network/route/model identity changed")
        for key, value in result.items():
            if key in ("training_seed", "model_sha256"):
                continue
            actual_value = getattr(row, key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not np.isclose(actual_value, value, equal_nan=True):
                    raise ValueError(f"CSV/raw metrics differ: {key}")
            elif actual_value != value:
                raise ValueError(f"CSV/raw result differs: {key}")
        if row.method in BASELINES:
            old = reference_result(config, row.method, row.scenario, row.traffic_seed)
            if any(result[k] != value for k, value in old.items() if k in result):
                raise ValueError("Baseline no longer exactly reproduces reference")
            if result["correction_effective"] != 0:
                raise ValueError("Uncorrected baseline was corrected")
            reference_matches += 1
        checked, trips = audit_case(directory, row, config)
        intervals += checked
        ids = {
            v.get("id")
            for v in ET.parse(route).getroot().findall("vehicle")
            if len(v.find("route").get("edges").split()) == 3
        }
        cross = [
            t
            for t in trips
            if t.get("id") in ids
            and float(t.get("arrival", "-1")) >= 0
            and t.get("vaporized", "") in ("", "false", "0")
        ]
        cohort = {
            "method": row.method,
            "training_seed": row.training_seed,
            "scenario": row.scenario,
            "traffic_seed": row.traffic_seed,
            "cross_planned": len(ids),
            "cross_completed": len(cross),
        }
        for name, fields in (
            ("cross_mean_wait", ("waitingTime",)),
            ("cross_mean_system_time", ("duration", "departDelay")),
            ("cross_mean_stops", ("waitingCount",)),
        ):
            cohort[name] = float(np.mean([sum(float(t.get(f)) for f in fields) for t in cross]))
        cohorts.append(cohort)
        for name in (
            "identity.json",
            "result.json",
            "episode_00000/trace.csv",
            "episode_00000/corrections.csv",
            "episode_00000/tripinfo.xml",
        ):
            inputs[str((directory / name).relative_to(output))] = sha256(directory / name)
        inputs[str(relative_route)] = sha256(route)
    frame = frame.merge(
        pd.DataFrame(cohorts),
        on=["method", "training_seed", "scenario", "traffic_seed"],
        validate="one_to_one",
    )
    frame["correction_rate"] = frame.correction_effective / frame.correction_node_decisions
    training = pd.read_csv(output / "training.csv")
    train_expected = {(m, s) for m in LEARNED for s in config["training_seeds"]}
    train_actual = list(training[["method", "training_seed"]].itertuples(index=False, name=None))
    if set(train_actual) != train_expected or len(train_actual) != len(train_expected):
        raise ValueError("Training matrix incomplete/duplicated")
    events = []
    for row in training.itertuples():
        directory = output / f"models/{row.method}/seed_{row.training_seed}"
        report = json.loads((directory / "training.json").read_text())
        if report["source_sha256"] != source_digest() or report["model_sha256"] != sha256(
            directory / "model.zip"
        ):
            raise ValueError("Trained model identity changed")
        if (
            report["decision_steps"] != config["training_steps"]
            or report["gradient_updates"]
            != (config["training_steps"] - config["learning_starts"]) // config["train_freq"]
        ):
            raise ValueError("Unexpected update budget")
        if not report["weights_changed"] or not report["save_reload_weights_match"]:
            raise ValueError("Model update/reload verification failed")
        if not (
            frame[
                (frame.method == row.method) & (frame.training_seed == row.training_seed)
            ].model_sha256
            == report["model_sha256"]
        ).all():
            raise ValueError("Validation used different weights")
        tags, count = set(), 0
        for path in (directory / "tensorboard").rglob("events.out.tfevents.*"):
            acc = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
            for tag in acc.Tags()["scalars"]:
                values = acc.Scalars(tag)
                if any(not np.isfinite(v.value) for v in values):
                    raise ValueError("Nonfinite TensorBoard scalar")
                tags.add(tag)
                count += len(values)
        if "train/loss" not in tags:
            raise ValueError("Missing loss history")
        events.append(
            {
                "method": row.method,
                "seed": row.training_seed,
                "finite_scalars": count,
                "tags": sorted(tags),
            }
        )
        inputs[str((directory / "model.zip").relative_to(output))] = report["model_sha256"]
        inputs[str((directory / "training.json").relative_to(output))] = sha256(
            directory / "training.json"
        )
    summaries, per_seed, paired = [], [], []
    for scope in ("all", *config["scenarios"]):
        subset = frame if scope == "all" else frame[frame.scenario == scope]
        for method in METHODS:
            part = subset[subset.method == method]
            eligible = bool(part.valid_complete.all())
            means = part.groupby("training_seed")[list(METRICS)].mean()
            record = {
                "scenario": scope,
                "method": method,
                "cases": len(part),
                "complete_cases": int(part.valid_complete.sum()),
                "eligible": eligible,
            }
            for metric in METRICS:
                record[metric] = float(means[metric].mean()) if eligible else None
                record[f"{metric}_seed_sd"] = (
                    float(means[metric].std(ddof=1)) if (eligible and len(means) > 1) else None
                )
            summaries.append(record)
            for seed, values in means.iterrows():
                per_seed.append(
                    {
                        "scenario": scope,
                        "method": method,
                        "training_seed": seed,
                        "eligible": eligible,
                        **{k: float(values[k]) if eligible else None for k in METRICS},
                    }
                )
        for method in LEARNED:
            for reference_method in (*BASELINES, "always_extend", "always_urgent"):
                part = subset[subset.method == method].merge(
                    subset[subset.method == reference_method],
                    on=["scenario", "traffic_seed"],
                    suffixes=("", "_reference"),
                    validate="many_to_one",
                )
                eligible = bool(part.valid_complete.all() and part.valid_complete_reference.all())
                differences = (
                    part[list(METRICS)].to_numpy()
                    - part[[f"{m}_reference" for m in METRICS]].to_numpy()
                )
                delta_frame = pd.DataFrame(differences, columns=list(METRICS))
                delta_frame["training_seed"] = part.training_seed.to_numpy()
                seed_deltas = delta_frame.groupby("training_seed").mean()
                paired.append(
                    {
                        "scenario": scope,
                        "method": method,
                        "reference": reference_method,
                        "eligible": eligible,
                        **{
                            f"delta_{m}": float(seed_deltas[m].mean()) if eligible else None
                            for m in METRICS
                        },
                        **{
                            f"delta_{m}_seed_sd": float(seed_deltas[m].std(ddof=1))
                            if (eligible and len(seed_deltas) > 1)
                            else None
                            for m in METRICS
                        },
                    }
                )
    summary = pd.DataFrame(summaries)
    destination = project_root() / "docs/results" / output.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("study.json", "selection.json", "provenance.json", "execution_all.json"):
        if (output / name).exists():
            shutil.copyfile(output / name, destination / name)
    for name, data in (
        ("validation", frame),
        ("summary", summary),
        ("training", training),
        ("per_seed", pd.DataFrame(per_seed)),
        ("paired_differences", pd.DataFrame(paired)),
    ):
        data.to_csv(destination / f"{name}.csv", index=False)
    global_rows = summary[summary.scenario == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, title, scale in zip(
        axes,
        ("mean_wait", "mean_system_time", "over_threshold_fraction"),
        ("Mean stop wait (s)", "End-to-end system time (s)", "Wait > 90s (%)"),
        (1, 1, 100),
        strict=True,
    ):
        positions = np.arange(len(global_rows))
        axis.bar(
            positions,
            global_rows[metric].astype(float) * scale,
            yerr=global_rows[f"{metric}_seed_sd"].astype(float).fillna(0) * scale,
            capsize=3,
            color=["#458b74"] * 3 + ["#b99868"] * 2 + ["#4468a2", "#b65b69"],
        )
        axis.set_xticks(positions, [LABELS[m] for m in global_rows.method], rotation=40, ha="right")
        for index, eligible in enumerate(global_rows.eligible):
            if not eligible:
                axis.text(index, 0, "incomplete", rotation=90, ha="center", va="bottom")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Baseline-assisted DQN: development validation\n"
        "Error bars: SD of model-seed means; no model-seed error for rule controllers"
    )
    fig.tight_layout()
    fig.savefig(destination / "comparison.png", dpi=160)
    plt.close(fig)
    report = [
        "# Baseline-assisted DQN: audited development results",
        "",
        "| Method | Complete | Mean wait | P95 | >90s | System time | Correction rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in global_rows.itertuples():
        values = (
            [
                f"{row.mean_wait:.2f}",
                f"{row.p95_wait:.2f}",
                f"{100 * row.over_threshold_fraction:.2f}%",
                f"{row.mean_system_time:.2f}",
                f"{100 * row.correction_rate:.2f}%",
            ]
            if row.eligible
            else ["—"] * 5
        )
        report.append(
            f"| {row.method} | {row.complete_cases}/{row.cases} | " + " | ".join(values) + " |"
        )
    (destination / "report.md").write_text("\n".join(report) + "\n")
    save_json(
        destination / "quality.json",
        {
            "source_sha256": source_digest(),
            "historical_core_sha256": historical_digest(),
            "summary_script_sha256": sha256(Path(__file__)),
            "validation_cases": len(frame),
            "complete_validation_cases": int(frame.valid_complete.sum()),
            "baseline_cases_exactly_reproduced": reference_matches,
            "traces_audited": len(frame),
            "physical_intervals_audited": intervals,
            "running": int(frame.running.sum()),
            "pending_insertion": int(frame.pending_insertion.sum()),
            "teleport_events": int(frame.teleport_events.sum()),
            "collision_events": int(frame.collision_events.sum()),
            "test_generated_or_evaluated": False,
            "tensorboard": events,
            "input_directory": str(output.relative_to(project_root())),
            "inputs": {
                **inputs,
                **{
                    name: sha256(output / name)
                    for name in (
                        "study.json",
                        "validation.csv",
                        "training.csv",
                        "selection.json",
                    )
                },
            },
            "exported_files": {
                p.name: sha256(p)
                for p in destination.iterdir()
                if p.is_file() and p.name != "quality.json"
            },
        },
    )
    print("\n".join(report), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/residual-50k")
    args = parser.parse_args()
    summarize(output_directory(args.output))
