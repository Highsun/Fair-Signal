"""Audit identities, physics, reward events and all predeclared experiment cells."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from iteration_study import BASELINES, METHODS, NEW, OLD, cases, model_path, source_digest
from residual_study import source_digest as residual_digest
from summarize_residual import audit_case
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fair_signal.demand import save_json, sha256  # noqa: E402
from fair_signal.runtime import output_directory, project_root  # noqa: E402

METRICS = (
    "mean_wait",
    "mean_system_time",
    "over_threshold_fraction",
    "p95_wait",
    "mean_excess_wait",
    "mean_stops",
    "wait_main",
    "wait_side",
    "mean_depart_delay",
)
LABELS = {
    "actuated": "Actuated",
    "max_pressure": "Max pressure",
    "aging": "Aging",
    "residual_dqn": "Residual DQN",
    "fair_residual_dqn": "Original fair DQN",
    "excess_025": "Excess weight .25",
    "crossing_30": "Crossing K=30",
    "crossing_90": "Crossing K=90",
    "matched_full": "Matched full",
    "matched_local": "Matched local",
}


def summaries(frame):
    rows = []
    for (split, group), part in frame.groupby(["split", "group"], sort=False):
        for scenario in ("all", *part.scenario.unique()):
            scope = part if scenario == "all" else part[part.scenario == scenario]
            for method in METHODS:
                data = scope[scope.method == method]
                eligible = bool(len(data) and data.valid_complete.all())
                seeds = data.groupby("training_seed")[list(METRICS)].mean()
                row = dict(
                    split=split,
                    group=group,
                    scenario=scenario,
                    method=method,
                    cases=len(data),
                    complete_cases=int(data.valid_complete.sum()),
                    eligible=eligible,
                )
                for m in METRICS:
                    row[m] = float(seeds[m].mean()) if eligible else None
                    row[f"{m}_seed_sd"] = (
                        float(seeds[m].std()) if eligible and len(seeds) > 1 else None
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(frame):
    rows = []
    comparisons = [
        ("residual_dqn", "actuated"),
        ("residual_dqn", "max_pressure"),
        ("fair_residual_dqn", "residual_dqn"),
        ("crossing_30", "residual_dqn"),
        ("crossing_90", "residual_dqn"),
        ("crossing_30", "excess_025"),
        ("crossing_90", "excess_025"),
        ("excess_025", "residual_dqn"),
        ("matched_full", "matched_local"),
    ]
    for method, reference in comparisons:
        a, b = frame[frame.method == method], frame[frame.method == reference]
        keys = ["split", "case_id"] + ([] if reference in BASELINES else ["training_seed"])
        paired = a.merge(b, on=keys, suffixes=("", "_reference"), validate="many_to_one")
        for r in paired.itertuples():
            row = dict(
                split=r.split,
                group=r.group,
                scenario=r.scenario,
                method=method,
                reference=reference,
                case_id=r.case_id,
                training_seed=r.training_seed,
                eligible=bool(r.valid_complete and r.valid_complete_reference),
            )
            for m in METRICS:
                row[f"delta_{m}"] = (
                    getattr(r, m) - getattr(r, m + "_reference") if row["eligible"] else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_results(summary, destination):
    # Standard-window normal-demand development vs untouched final test.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row_index, (split, group) in enumerate((("development", "rate_1"), ("test", "test"))):
        data = summary[
            (summary.split == split) & (summary.group == group) & (summary.scenario == "all")
        ]
        if data.empty:
            continue
        data = data.set_index("method").reindex(METHODS)
        for axis, metric, title, scale in zip(
            axes[row_index],
            ("mean_wait", "over_threshold_fraction", "mean_excess_wait"),
            ("Mean stop wait (s)", "Wait > 90s (%)", "Mean excess stop wait (s)"),
            (1, 100, 1),
            strict=True,
        ):
            axis.bar(
                np.arange(len(data)),
                data[metric] * scale,
                yerr=data[f"{metric}_seed_sd"].astype(float).fillna(0) * scale,
                capsize=2,
                color=["#458b74"] * 3
                + ["#4468a2", "#b65b69", "#c4a373", "#d59445", "#ba6c25", "#6863a7", "#9e93c6"],
            )
            axis.set_xticks(
                np.arange(len(data)), [LABELS[m] for m in data.index], rotation=40, ha="right"
            )
            for i, eligible in enumerate(data.eligible):
                if not eligible:
                    axis.text(i, 0, "incomplete", rotation=90, ha="center")
            axis.set_title(f"{split}: {title}")
            axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen 50k endpoints; SD of three model-seed means, not confidence intervals")
    fig.tight_layout()
    fig.savefig(destination / "comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, scenario in zip(axes, ("steady", "tidal", "downstream_slowdown"), strict=True):
        for method in ("actuated", "max_pressure", "residual_dqn", "fair_residual_dqn"):
            data = summary[
                (summary.split == "development")
                & (summary.scenario == scenario)
                & summary.group.str.startswith("rate_")
                & (summary.method == method)
            ].copy()
            data["rate"] = data.group.str.removeprefix("rate_").astype(float)
            data = data.sort_values("rate")
            axis.plot(data.rate, data.mean_wait, marker="o", label=LABELS[method])
        axis.set_title(scenario)
        axis.set_xlabel("Demand / original validation demand")
        axis.set_ylabel("Mean stop wait (s)")
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "generalization.png", dpi=150)
    plt.close(fig)


def audit_and_export(output, splits=("development", "test")):
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    study = json.loads((output / "study.json").read_text())
    if (
        study["source_sha256"] != source_digest()
        or study["residual_source_sha256"] != residual_digest()
    ):
        raise ValueError("Frozen source changed")
    config = study["config"]
    reference = output_directory(config["reference"])
    if sha256(reference / "study.json") != study["reference_study_sha256"]:
        raise ValueError("Reference study changed")
    if sha256(output / "selection.json") != study["reference_selection_sha256"]:
        raise ValueError("Baseline selection changed")
    reference_config = json.loads((reference / "study.json").read_text())["config"]
    inputs = {}
    training = pd.read_csv(output / "training.csv")
    train_actual = list(training[["method", "training_seed"]].itertuples(index=False, name=None))
    train_expected = {(m, s) for m in NEW for s in config["training_seeds"]}
    if set(train_actual) != train_expected or len(train_actual) != len(train_expected):
        raise ValueError("Training matrix incomplete/duplicated")
    events = []
    for method in (*OLD, *NEW):
        for seed in config["training_seeds"]:
            path = model_path(output, method, seed)
            report = json.loads((path.parent / "training.json").read_text())
            expected_source = residual_digest() if method in OLD else source_digest()
            if sha256(path) != report["model_sha256"] or report["source_sha256"] != expected_source:
                raise ValueError("Model identity changed")
            if (
                method in OLD
                and sha256(path) != study["reference_model_hashes"][f"{method}/{seed}"]
            ):
                raise ValueError("Old model changed")
            model = DQN.load(path, device="cpu")
            budget = reference_config if method in OLD else config
            if (
                model.num_timesteps != budget["training_steps"]
                or model._n_updates
                != (budget["training_steps"] - budget["learning_starts"]) // budget["train_freq"]
            ):
                raise ValueError("Serialized training budget differs")
            if model.observation_space.shape != (162,) or model.action_space.n != 9:
                raise ValueError("Model dimensions differ")
            if sum(p.numel() for p in model.q_net.parameters()) != report["parameter_count"]:
                raise ValueError("Parameter count differs")
            if any(not torch.isfinite(p).all() for p in model.policy.parameters()):
                raise ValueError("Nonfinite model")
            if not report["weights_changed"] or not report["save_reload_weights_match"]:
                raise ValueError("Untrained/reload mismatch")
            if method.startswith("matched_"):
                masks = model.q_net.information_masks
                expected_masks = torch.ones(2, 162)
                if method == "matched_local":
                    expected_masks.zero_()
                    expected_masks[0, :76] = expected_masks[0, 152:157] = 1
                    expected_masks[1, 76:152] = expected_masks[1, 157:162] = 1
                if not torch.equal(masks.cpu(), expected_masks):
                    raise ValueError("Information mask changed after serialization")
            tags, count = set(), 0
            for event in (path.parent / "tensorboard").rglob("events.out.tfevents.*"):
                acc = EventAccumulator(str(event), size_guidance={"scalars": 0}).Reload()
                for tag in acc.Tags()["scalars"]:
                    values = acc.Scalars(tag)
                    if any(not np.isfinite(v.value) for v in values):
                        raise ValueError("Nonfinite TensorBoard value")
                    count += len(values)
                    tags.add(tag)
            if "train/loss" not in tags:
                raise ValueError("Missing training loss")
            if method in NEW and "traffic/crossing_cost" not in tags:
                raise ValueError("Missing crossing logging")
            events.append(dict(method=method, seed=seed, finite_scalars=count, tags=sorted(tags)))
            inputs[f"model/{method}/{seed}"] = sha256(path)
    frames, intervals = [], 0
    for split in splits:
        frame = pd.read_csv(output / f"{split}.csv")
        frame["training_seed"] = frame.training_seed.fillna(-1).astype(int)
        actual = list(
            frame[["method", "training_seed", "case_id"]].itertuples(index=False, name=None)
        )
        records = cases(config, split)
        expected = {
            (m, s, c["id"])
            for m in METHODS
            for s in ([-1] if m in BASELINES else config["training_seeds"])
            for c in records
        }
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("Evaluation matrix incomplete/duplicated")
        case_map = {c["id"]: c for c in records}
        for row in frame.itertuples():
            seed = None if row.training_seed == -1 else row.training_seed
            directory = output / f"evaluation/{split}/{row.method}/seed_{seed}/{row.case_id}"
            route = output / f"cases/{split}/{row.case_id}/vehicles.rou.xml"
            path = model_path(output, row.method, seed) if row.method not in BASELINES else None
            identity = json.loads((directory / "identity.json").read_text())
            expected_identity = dict(
                source_sha256=source_digest(),
                config=config,
                case=case_map[row.case_id],
                model_sha256=sha256(path) if path else None,
                route_sha256=sha256(route),
                network_sha256=sha256(output / "network/arterial.net.xml"),
                selection_sha256=sha256(output / "selection.json"),
            )
            if identity != expected_identity or row.route_sha256 != sha256(route):
                raise ValueError("Evaluation source/route/model identity changed")
            raw = json.loads((directory / "result.json").read_text())
            for key, value in raw.items():
                if key in ("training_seed", "model_sha256"):
                    continue
                actual_value = getattr(row, key)
                if value is None:
                    if not pd.isna(actual_value):
                        raise ValueError("Null metric differs")
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not np.isclose(actual_value, value):
                        raise ValueError(f"CSV/raw mismatch {key}")
                elif actual_value != value:
                    raise ValueError(f"CSV/raw mismatch {key}")
            checked, trips, _ = audit_case(directory, row, config)
            intervals += checked
            waits = np.array([float(t.get("waitingTime")) for t in trips])
            crossing_count = int((waits > 90).sum())
            trace = pd.read_csv(directory / "episode_00000/trace.csv")
            if (
                row.first_crossings != crossing_count
                or trace.first_crossings.sum() != crossing_count
            ):
                raise ValueError("First crossing count differs from TripInfo/trace")
            if not np.isclose(row.mean_excess_wait, np.maximum(waits - 90, 0).sum() / row.planned):
                raise ValueError("Excess wait differs from TripInfo")
            case = case_map[row.case_id]
            want_slow = (trace.second > case["window"][0]) & (trace.second <= case["window"][1])
            if case["scenario"] != "downstream_slowdown":
                want_slow[:] = False
            if not np.array_equal(trace.slowdown, want_slow):
                raise ValueError("Actual slowdown window differs")
            if row.method in BASELINES and row.correction_effective != 0:
                raise ValueError("Baseline received correction")
            for name in (
                "identity.json",
                "result.json",
                "episode_00000/trace.csv",
                "episode_00000/corrections.csv",
                "episode_00000/tripinfo.xml",
            ):
                inputs[str((directory / name).relative_to(output))] = sha256(directory / name)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    summary = summaries(frame)
    destination = project_root() / "docs/results" / output.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "study.json",
        "provenance.json",
        "selection.json",
        "freeze.json",
        "cases_development.json",
        "cases_test.json",
        "execution_all.json",
    ):
        if (output / name).exists():
            shutil.copyfile(output / name, destination / name)
    frame.to_csv(destination / "evaluation.csv", index=False)
    summary.to_csv(destination / "summary.csv", index=False)
    training.to_csv(destination / "training.csv", index=False)
    paired_rows(frame).to_csv(destination / "paired.csv", index=False)
    per_seed = frame.groupby(["split", "group", "scenario", "method", "training_seed"])[
        list(METRICS)
    ].mean()
    per_seed.to_csv(destination / "per_seed.csv")
    plot_results(summary, destination)
    save_json(
        destination / "quality.json",
        {
            "purpose": "connectivity_check_not_performance_evidence"
            if config["training_steps"] != 50000
            else ("development_and_frozen_test" if "test" in splits else "development_before_test"),
            "source_sha256": source_digest(),
            "residual_source_sha256": residual_digest(),
            "summary_script_sha256": sha256(Path(__file__)),
            "evaluation_cases": len(frame),
            "complete_cases": int(frame.valid_complete.sum()),
            "splits": list(splits),
            "traces_audited": len(frame),
            "physical_intervals_audited": intervals,
            "running": int(frame.running.sum()),
            "pending_insertion": int(frame.pending_insertion.sum()),
            "collisions": int(frame.collision_events.sum()),
            "teleports": int(frame.teleport_events.sum()),
            "abnormal_removals": int(frame.abnormal_removals.sum()),
            "freeze_sha256": sha256(output / "freeze.json")
            if (output / "freeze.json").exists()
            else None,
            "tensorboard": events,
            "input_hashes": inputs,
            "exported_files": {
                p.name: sha256(p)
                for p in destination.iterdir()
                if p.is_file() and p.name != "quality.json"
            },
        },
    )
    print(
        summary[(summary.scenario == "all") & summary.group.isin(["rate_1", "test"])][
            ["split", "method", "complete_cases", "cases", *METRICS]
        ].to_string(index=False),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/iteration-50k")
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    audit_and_export(
        output_directory(args.output),
        ("development",) if args.development_only else ("development", "test"),
    )
