"""Verify uninterrupted trajectories and summarize fixed validation checkpoints."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from summarize_validation import export
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from fair_signal.analysis import METRICS, hierarchical_summary
from fair_signal.demand import EXITS, save_json, sha256
from fair_signal.experiment import source_digest
from fair_signal.runtime import project_root


def audit_starvation(run: Path, frame: pd.DataFrame, destination: Path) -> None:
    rows = []
    for row in frame[~frame.valid_complete].itertuples():
        point = run / f"step_{row.checkpoint}"
        episode = (
            point
            / "evaluation/validation"
            / row.method
            / "candidate_0"
            / f"model_{int(row.training_seed)}"
            / row.scenario
            / str(row.traffic_seed)
            / "episode_00000"
        )
        trace = pd.read_csv(episode / "trace.csv")
        tail = trace[trace.second > trace.second.max() - 600]
        route = ET.parse(
            point / f"cases/validation/{row.scenario}/{row.traffic_seed}/vehicles.rou.xml"
        ).getroot()
        routes = {
            item.attrib["id"]: item.attrib["edges"].split() for item in route.findall("route")
        }
        vehicles = {
            item.attrib["id"]: routes[item.attrib["route"]] for item in route.findall("vehicle")
        }
        for trip in ET.parse(episode / "tripinfo.xml").getroot().findall("tripinfo"):
            if float(trip.attrib["arrival"]) >= 0:
                continue
            edges = vehicles[trip.attrib["id"]]
            direction, target = edges[0].split("_")[0], edges[-1].split("_")[-1]
            phase = (0 if direction in ("n", "s") else 2) + int(EXITS[direction].index(target) == 2)
            rows.append(
                {
                    "checkpoint": row.checkpoint,
                    "method": row.method,
                    "training_seed": int(row.training_seed),
                    "scenario": row.scenario,
                    "traffic_seed": row.traffic_seed,
                    "vehicle": trip.attrib["id"],
                    "required_phase": phase,
                    "last600_required_green_seconds": int(
                        ((tail["mode"] == "green") & (tail.phase == phase)).sum()
                    ),
                    "unfinished_wait": float(trip.attrib["waitingTime"]),
                    "last600_head_wait_max": float(tail.head_wait_max.max()),
                }
            )
    pd.DataFrame(rows).to_csv(destination / "starvation_audit.csv", index=False)


def main() -> None:
    torch.set_num_threads(1)
    root = project_root()
    run = root / "outputs/long-budget-150k"
    destination = root / "docs/results/long-budget-150k"
    study = json.loads((run / "study.json").read_text())
    if study["runner_sha256"] != sha256(root / "scripts/long_budget_study.py"):
        raise ValueError("Frozen study runner changed")
    if study["source_sha256"] != source_digest():
        raise ValueError("Frozen core source changed")
    if study["checkpoints"] != [50000, 100000, 150000]:
        raise ValueError("Unexpected checkpoint plan")
    if (run / "cases/test").exists() or (run / "test.csv").exists():
        raise ValueError("Reserved test used")
    destination.mkdir(parents=True, exist_ok=True)
    frames, models, quality = [], [], []
    for step in study["checkpoints"]:
        point = run / f"step_{step}"
        if (point / "cases/test").exists():
            raise ValueError("Reserved test routes generated")
        record = export(point, destination / f"step_{step}")
        expected_cases = 120 if step == 150000 else 90
        if record["trained_models"] != 6 or record["validation_cases"] != expected_cases:
            raise ValueError("Unexpected validation matrix")
        quality.append({"checkpoint": step, **record})
        frame = pd.read_csv(point / "validation.csv")
        frame["checkpoint"] = step
        frames.append(frame)
        for path in sorted((point / "models").glob("*/*/*/training.json")):
            training = json.loads(path.read_text())
            if (
                training["runner_sha256"] != study["runner_sha256"]
                or training["gradient_updates"] != (step - 1000) // 4
                or training["trajectory_budget"] != 150000
                or training["exploration_decay_steps"] != 25000
                or not training["weights_finite"]
            ):
                raise ValueError("Training trajectory/checkpoint mismatch")
            models.append({"checkpoint": step, **training})
    frame = pd.concat(frames, ignore_index=True)
    if not frame.groupby(["scenario", "traffic_seed"]).route_sha256.nunique().eq(1).all():
        raise ValueError("Different routes between checkpoints")
    frame.to_csv(destination / "validation.csv", index=False)
    audit_starvation(run, frame, destination)
    save_json(destination / "quality.json", quality)
    save_json(destination / "training.json", models)
    for name in ("study.json", "provenance.json", "execution.json"):
        shutil.copy2(run / name, destination / name)

    # Earlier 50k runs used the same environment, seeds, objective and epsilon decay.
    # The frozen prefix must match both online/target weights and every metric.
    prefix_checks = []
    for method in ("dqn_queue", "dqn_fair"):
        for seed in (11, 22, 33):
            old_path = (
                root / f"outputs/improvement-50k/models/{method}/candidate_0/seed_{seed}/model.zip"
            )
            new_path = run / f"step_50000/models/{method}/candidate_0/seed_{seed}/model.zip"
            old_record = json.loads(old_path.with_name("training.json").read_text())
            if old_record["model_sha256"] != sha256(old_path):
                raise ValueError("Historical model checksum changed")
            old = DQN.load(old_path, device="cpu").policy.state_dict()
            new = DQN.load(new_path, device="cpu").policy.state_dict()
            identical = old.keys() == new.keys() and all(
                torch.equal(old[key], new[key]) for key in old
            )
            if not identical:
                raise ValueError(f"50k prefix weights differ: {method} seed={seed}")
            prefix_checks.append(
                {
                    "method": method,
                    "training_seed": seed,
                    "online_target_identical": True,
                    "old_model_sha256": sha256(old_path),
                    "new_model_sha256": sha256(new_path),
                }
            )
    old_path = root / "outputs/improvement-50k/validation.csv"
    archived_old = root / "docs/results/improvement-50k/validation.csv"
    if old_path.read_text() != archived_old.read_text():
        raise ValueError("Historical frozen validation changed")
    old = pd.read_csv(old_path)
    old = old[old.method.isin(["dqn_queue", "dqn_fair"])]
    new = frame[frame.checkpoint == 50000].drop(columns="checkpoint")
    keys = ["method", "candidate", "training_seed", "scenario", "traffic_seed"]
    paired_prefix = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    if len(paired_prefix) != 90:
        raise ValueError("Missing 50k prefix cases")
    for column in old.columns.difference(keys):
        a, b = paired_prefix[f"{column}_old"], paired_prefix[f"{column}_new"]
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise ValueError(f"50k prefix evaluation changed: {column}")
    previous = pd.read_csv(old_path)
    previous = previous[previous.method.isin(["actuated", "aging"])]
    baselines = frame[(frame.checkpoint == 150000) & frame.method.isin(["actuated", "aging"])]
    matched = previous.merge(baselines, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    if len(matched) != 30:
        raise ValueError("Missing historical baseline cases")
    for column in previous.columns.difference(keys):
        a, b = matched[f"{column}_old"], matched[f"{column}_new"]
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise ValueError(f"Historical baseline changed: {column}")
    save_json(
        destination / "prefix_regression.json",
        {
            "models": prefix_checks,
            "identical_validation_cases": 90,
            "identical_baseline_cases": 30,
            "old_validation_sha256": sha256(old_path),
            "old_source_sha256": json.loads((old_path.parent / "config.json").read_text())[
                "source_sha256"
            ],
            "new_source_sha256": study["source_sha256"],
            "all_metrics_identical": True,
        },
    )

    per_models, scenario_summaries, global_rows = [], [], []
    for step, group in frame.groupby("checkpoint"):
        _, scenario_summary = hierarchical_summary(group)
        scenario_summary = scenario_summary.rename(columns={"test_cases": "validation_cases"})
        scenario_summary["checkpoint"] = step
        scenario_summaries.append(scenario_summary)
        normalized = group.assign(training_seed=group.training_seed.fillna(-1).astype(int))
        per_model = normalized.groupby(["method", "training_seed"])[METRICS].mean()
        complete = normalized.groupby(["method", "training_seed"]).valid_complete.all()
        per_model.loc[~complete, METRICS] = np.nan
        per_model["all_cases_complete"] = complete
        per_model["checkpoint"] = step
        per_models.append(per_model.reset_index())
        for method, values in per_model.groupby("method"):
            valid = bool(values.all_cases_complete.all())
            row = {"checkpoint": step, "method": method, "all_cases_complete": valid}
            for metric in METRICS:
                row[metric] = values[metric].mean() if valid else np.nan
                row[f"{metric}_training_std"] = values[metric].std(ddof=1) if valid else np.nan
            global_rows.append(row)
    per_model = pd.concat(per_models, ignore_index=True)
    summary = pd.DataFrame(global_rows)
    per_model.to_csv(destination / "per_model.csv", index=False)
    summary.to_csv(destination / "global_summary.csv", index=False)
    pd.concat(scenario_summaries).to_csv(destination / "scenario_summary.csv", index=False)
    pairs = []
    for step in study["checkpoints"]:
        point = frame[frame.checkpoint == step]
        for reference in ("dqn_queue", "actuated", "aging"):
            if reference not in point.method.unique():
                continue
            for seed in (11, 22, 33):
                a = point[(point.method == "dqn_fair") & (point.training_seed == seed)]
                b = point[point.method == reference]
                if reference.startswith("dqn_"):
                    b = b[b.training_seed == seed]
                match = a.merge(
                    b,
                    on=["scenario", "traffic_seed"],
                    suffixes=("_fair", "_reference"),
                    validate="one_to_one",
                )
                if (
                    len(match) != 15
                    or not match.route_sha256_fair.eq(match.route_sha256_reference).all()
                ):
                    raise ValueError("Missing matched validation reference")
                valid = bool(
                    match.valid_complete_fair.all() and match.valid_complete_reference.all()
                )
                pairs.append(
                    {
                        "checkpoint": step,
                        "reference": reference,
                        "training_seed": seed,
                        "paired_cases": 15,
                        "all_cases_complete": valid,
                        **{
                            metric: float(
                                (match[f"{metric}_fair"] - match[f"{metric}_reference"]).mean()
                            )
                            if valid
                            else np.nan
                            for metric in METRICS
                        },
                    }
                )
    pd.DataFrame(pairs).to_csv(destination / "fair_minus_reference_by_seed.csv", index=False)

    tensorboard = []
    reward_diagnostics = []
    for directory in sorted((run / "trajectories").glob("*/seed_*/tensorboard")):
        paths = sorted(directory.glob("DQN_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not paths:
            raise ValueError("Missing TensorBoard run")
        path = paths[-1]
        learning = pd.read_csv(directory.parent / "learning.csv")
        weights = np.diff(np.r_[0, learning.step.to_numpy()])[-10:]
        tail = learning.tail(10)
        averages = {
            metric: float(np.average(tail[metric], weights=weights))
            for metric in (
                "mean_queue_cost",
                "mean_weighted_fair_cost",
                "fair_cost_active_fraction",
            )
        }
        reward_diagnostics.append(
            {
                "method": directory.parent.parent.name,
                "training_seed": int(directory.parent.name.split("_")[-1]),
                "last_logged_step": int(learning.step.iloc[-1]),
                "completed_episodes": len(learning),
                "last10_weighted_fair_to_queue_ratio": (
                    averages["mean_weighted_fair_cost"] / averages["mean_queue_cost"]
                ),
                **{f"last10_{key}": value for key, value in averages.items()},
            }
        )
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        tags = events.Tags()["scalars"]
        required = ["train/loss", "traffic/mean_wait", "traffic/mean_fair_cost"]
        if not set(required).issubset(tags):
            raise ValueError("Missing TensorBoard training diagnostics")
        values = {tag: events.Scalars(tag) for tag in required}
        if not all(np.isfinite(item.value) for data in values.values() for item in data):
            raise ValueError("Nonfinite TensorBoard values")
        tensorboard.append(
            {
                "trajectory": str(path.relative_to(run)),
                "loss_samples": len(values["train/loss"]),
                "last_logged_step": values["train/loss"][-1].step,
                "required_tags_present": True,
                "scalars_finite": True,
            }
        )
    if len(tensorboard) != 6:
        raise ValueError("Expected six independent TensorBoard trajectories")
    save_json(destination / "tensorboard_check.json", tensorboard)
    pd.DataFrame(reward_diagnostics).to_csv(destination / "reward_diagnostics.csv", index=False)

    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    for ax, metric, label, scale in zip(
        axes[:3],
        ["mean_wait", "p95_wait", "over_threshold_fraction"],
        ["Mean wait (s)", "Mean case P95 (s)", "Wait >90s (%)"],
        [1, 1, 100],
        strict=True,
    ):
        for method, color in (("dqn_queue", "tab:blue"), ("dqn_fair", "tab:orange")):
            data = per_model[per_model.method == method]
            for _seed, trajectory in data.groupby("training_seed"):
                ax.plot(
                    trajectory.checkpoint / 1000,
                    trajectory[metric] * scale,
                    color=color,
                    alpha=0.3,
                    linewidth=1,
                    marker="o",
                    markersize=3,
                )
            means = summary[summary.method == method]
            ax.plot(
                means.checkpoint / 1000,
                means[metric] * scale,
                color=color,
                linewidth=2.5,
                marker="o",
                label=method,
            )
        terminal = summary[summary.checkpoint == 150000]
        for method, color in (("actuated", "tab:green"), ("aging", "tab:red")):
            value = terminal[terminal.method == method][metric].iloc[0]
            ax.axhline(value * scale, color=color, linestyle="--", alpha=0.8, label=method)
        ax.set(xlabel="Decision steps (thousands)", ylabel=label, xticks=[50, 100, 150])
        ax.grid(alpha=0.2)
    for method, color in (("dqn_queue", "tab:blue"), ("dqn_fair", "tab:orange")):
        rates = (
            frame[frame.method == method].groupby("checkpoint").valid_complete.agg(["sum", "count"])
        )
        axes[3].plot(
            rates.index / 1000, 100 * rates["sum"] / rates["count"], color=color, marker="o"
        )
        for step, row in rates.iterrows():
            axes[3].annotate(
                f"{int(row['sum'])}/{int(row['count'])}",
                (step / 1000, 100 * row["sum"] / row["count"]),
                xytext=(0, -15 if method == "dqn_queue" else 6),
                textcoords="offset points",
                fontsize=8,
            )
    axes[3].set(
        xlabel="Decision steps (thousands)",
        ylabel="Complete cases (%)",
        xticks=[50, 100, 150],
        ylim=(0, 108),
    )
    axes[3].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed epsilon decay: thin lines = seed trajectories, thick lines = means")
    fig.tight_layout()
    fig.savefig(figures / "matched_budget.png", dpi=170)
    plt.close(fig)
    lines = [
        "# 长预算实际验证汇总",
        "",
        "± 为三个训练种子的样本标准差；P95 为回合分位数的平均。",
        "检查点有相关性，主结果使用 150000 步末端；不是最好检查点选择。",
        "",
        "| 步数 | 方法 | 平均等待 s | P95 s | 超 90 秒比例 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        metrics = []
        for metric, scale, suffix in (
            ("mean_wait", 1, ""),
            ("p95_wait", 1, ""),
            ("over_threshold_fraction", 100, "%"),
        ):
            value, std = (
                getattr(row, metric) * scale,
                getattr(row, f"{metric}_training_std") * scale,
            )
            metrics.append(
                (f"{value:.2f}±{std:.2f}{suffix}" if np.isfinite(std) else f"{value:.2f}{suffix}")
                if row.all_cases_complete
                else "—"
            )
        lines.append(f"| {row.checkpoint} | {row.method} | {' | '.join(metrics)} |")
    lines += ["", "所有原始记录、失败回合、来源、分场景和成对差异在同目录。预留最终测试未评估。"]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(
        destination / "analysis.json",
        {
            "summarizer_sha256": sha256(Path(__file__)),
            "core_source_sha256": source_digest(),
            "study_runner_sha256": study["runner_sha256"],
            "historical_validation_sha256": sha256(old_path),
            "raw_checkpoint_validation_sha256": {
                str(step): sha256(run / f"step_{step}/validation.csv")
                for step in study["checkpoints"]
            },
            "independent_training_trajectories": 6,
            "saved_snapshots": 18,
            "training_decision_steps": 900000,
            "validation_cases": len(frame),
            "test_evaluated": False,
        },
    )
    print(
        summary[
            ["checkpoint", "method", "mean_wait", "p95_wait", "over_threshold_fraction"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
