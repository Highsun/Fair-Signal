"""Export a validation-only study; never read test performance or choose checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_signal.config import ExperimentConfig
from fair_signal.demand import save_json, sha256
from fair_signal.experiment import candidates, source_digest
from fair_signal.runtime import output_directory, project_root

METRICS = ["mean_wait", "p95_wait", "over_threshold_fraction", "mean_queue", "mean_depart_delay"]


def summarize(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Require the complete predeclared matrix before exposing candidate means."""
    expected = {
        (method, candidate, seed, scenario, traffic_seed)
        for method in config.methods
        for candidate in range(len(candidates(config, method)))
        for seed in (config.training_seeds if method.startswith("dqn_") else [-1])
        for scenario in config.scenarios
        for traffic_seed in config.validation_seeds
    }
    normalized = frame.copy()
    normalized["training_seed"] = pd.to_numeric(normalized.training_seed).fillna(-1).astype(int)
    keys = ["method", "candidate", "training_seed", "scenario", "traffic_seed"]
    observed = set(normalized[keys].itertuples(index=False, name=None))
    if observed != expected or len(normalized) != len(expected):
        raise ValueError("Validation matrix is incomplete, duplicated or unexpected")
    if not normalized.split.eq("validation").all():
        raise ValueError("Only validation records are allowed")
    if not normalized.groupby(["scenario", "traffic_seed"]).route_sha256.nunique().eq(1).all():
        raise ValueError("Candidates did not use identical demand")
    rows = []
    for (method, candidate, scenario), group in normalized.groupby(
        ["method", "candidate", "scenario"], sort=True
    ):
        complete = bool(group.valid_complete.all())
        record = {
            "method": method,
            "candidate": int(candidate),
            "parameters": json.dumps(candidates(config, method)[int(candidate)], sort_keys=True),
            "scenario": scenario,
            "valid_cases": int(group.valid_complete.sum()),
            "cases": len(group),
            "all_cases_complete": complete,
            "unfinished_vehicles": int(group.running.sum()),
            "pending_insertion": int(group.pending_insertion.sum()),
            "collision_events": int(group.collision_events.sum()),
            "teleport_events": int(group.teleport_events.sum()),
        }
        for metric in METRICS:
            # Equal cases per trained model make this an equal-model mean.
            record[metric] = float(group[metric].mean()) if complete else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def export(run: Path, destination: Path) -> dict:
    saved = json.loads((run / "config.json").read_text())
    if saved["source_sha256"] != source_digest():
        raise ValueError("Run source differs from current code")
    from fair_signal.config import EnvConfig

    values = saved["config"]
    values["env"] = EnvConfig(**values["env"])
    config = ExperimentConfig(**values)
    config.validate()
    if (run / "test.csv").exists():
        raise ValueError("This exporter is reserved for a validation-only study")
    frame = pd.read_csv(run / "validation.csv")
    summary = summarize(frame, config)
    selected = json.loads((run / "selection.json").read_text())
    chosen = summary[
        summary.apply(lambda row: row.candidate == selected[row.method]["candidate"], axis=1)
    ].copy()
    training = []
    for path in sorted((run / "models").glob("*/*/*/training.json")):
        record = json.loads(path.read_text())
        if record["source_sha256"] != saved["source_sha256"]:
            raise ValueError("Training source mismatch")
        if sha256(path.with_name("model.zip")) != record["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        if record["decision_steps"] != config.training_steps or record["gradient_updates"] <= 0:
            raise ValueError("Incomplete training")
        if not record["save_reload_action_matches"]:
            raise ValueError("Reload check failed")
        training.append(record)
    expected_models = sum(
        len(candidates(config, method)) * len(config.training_seeds)
        for method in config.methods
        if method.startswith("dqn_")
    )
    if len(training) != expected_models:
        raise ValueError("Missing training records")
    quality = {
        "purpose": "development_validation_only",
        "trained_models": len(training),
        "validation_cases": len(frame),
        "complete_cases": int(frame.valid_complete.sum()),
        "unfinished_vehicles": int(frame.running.sum()),
        "pending_insertion": int(frame.pending_insertion.sum()),
        "collision_events": int(frame.collision_events.sum()),
        "teleport_events": int(frame.teleport_events.sum()),
        "test_performance_evaluated": False,
        "training_seconds": sum(record["seconds"] for record in training),
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "provenance.json",
        "validation.csv",
        "validation_scores.json",
        "selection.json",
    ):
        if name.endswith(".csv"):
            # Normalize csv.writer CRLF for review without rounding numeric text.
            (destination / name).write_text(
                (run / name).read_text(encoding="utf-8"), encoding="utf-8"
            )
        else:
            shutil.copy2(run / name, destination / name)
    summary.to_csv(destination / "candidate_summary.csv", index=False)
    chosen.to_csv(destination / "selected_summary.csv", index=False)
    overall = []
    for method, group in chosen.groupby("method", sort=True):
        complete = bool(group.all_cases_complete.all())
        row = {
            "method": method,
            "candidate": int(group.candidate.iloc[0]),
            "parameters": group.parameters.iloc[0],
            "valid_cases": int(group.valid_cases.sum()),
            "cases": int(group.cases.sum()),
            "all_cases_complete": complete,
        }
        for metric in METRICS:
            row[metric] = float(group[metric].mean()) if complete else np.nan
        overall.append(row)
    pd.DataFrame(overall).to_csv(destination / "method_summary.csv", index=False)
    frame.loc[~frame.valid_complete].to_csv(destination / "failure_cases.csv", index=False)
    save_json(destination / "training.json", training)
    save_json(destination / "quality.json", quality)

    paired = []
    keys = ["scenario", "traffic_seed"]
    if len(config.training_seeds) == 1 and "dqn_fair" in selected:
        for reference in ("dqn_queue", "dqn_wait", "aging"):
            if reference not in selected:
                continue
            a = frame[
                (frame.method == "dqn_fair")
                & (frame.candidate == selected["dqn_fair"]["candidate"])
            ]
            b = frame[
                (frame.method == reference) & (frame.candidate == selected[reference]["candidate"])
            ]
            pairs = a.merge(b, on=keys, suffixes=("_fair", "_reference"), validate="one_to_one")
            complete = bool(
                pairs.valid_complete_fair.all() and pairs.valid_complete_reference.all()
            )
            row = {
                "reference": reference,
                "paired_cases": len(pairs),
                "all_cases_complete": complete,
            }
            for metric in METRICS:
                row[f"fair_minus_reference_{metric}"] = (
                    float((pairs[f"{metric}_fair"] - pairs[f"{metric}_reference"]).mean())
                    if complete
                    else np.nan
                )
            paired.append(row)
    pd.DataFrame(paired).to_csv(destination / "paired_deltas.csv", index=False)

    activation = []
    for row in frame[frame.method.str.startswith("dqn_")].itertuples():
        trace_path = (
            run
            / "evaluation/validation"
            / row.method
            / f"candidate_{row.candidate}"
            / f"model_{int(row.training_seed)}"
            / row.scenario
            / str(row.traffic_seed)
            / "episode_00000/trace.csv"
        )
        trace = pd.read_csv(trace_path)
        trace = trace[trace.second <= config.env.demand_seconds]
        if len(trace) != config.env.demand_seconds:
            raise ValueError("Missing demand-window trace seconds")
        activation.append(
            {
                "method": row.method,
                "candidate": row.candidate,
                "training_seed": int(row.training_seed),
                "scenario": row.scenario,
                "traffic_seed": row.traffic_seed,
                "fair_penalty_active_fraction": float(
                    (trace.fair_cost > 0).mean()
                    if "fair_cost" in trace
                    else (trace.head_wait_max > config.env.wait_threshold).mean()
                ),
                "head_wait_above_observation_scale_fraction": float(
                    (trace.head_wait_max > config.env.wait_scale).mean()
                ),
                "mean_normalized_queue_cost": float(trace.queue.mean() / config.env.queue_scale),
            }
        )
    pd.DataFrame(activation).to_csv(destination / "reward_activation.csv", index=False)

    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(config.scenarios), figsize=(15, 4), squeeze=False)
    for ax, scenario in zip(axes[0], config.scenarios, strict=True):
        for _, row in chosen[chosen.scenario == scenario].iterrows():
            if row.all_cases_complete:
                ax.scatter(row.mean_wait, row.p95_wait)
                ax.annotate(
                    row.method,
                    (row.mean_wait, row.p95_wait),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set(
            title=f"{scenario} (validation)",
            xlabel="Mean trip wait (s)",
            ylabel="Mean case P95 (s)",
        )
        ax.grid(alpha=0.2)
        ax.margins(x=0.18, y=0.12)
    fig.tight_layout()
    fig.savefig(figures / "validation_tradeoff.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for path in sorted((run / "models").glob("*/*/*/learning.csv")):
        values = pd.read_csv(path)
        label = "/".join(path.relative_to(run / "models").parts[:-1])
        axes[0].plot(
            values.step, values.mean_wait.rolling(5, min_periods=1).mean(), label=label, linewidth=1
        )
        axes[1].plot(
            values.step,
            values.valid_complete.astype(float).rolling(5, min_periods=1).mean(),
            label=label,
            linewidth=1,
        )
    axes[0].set(
        ylabel="Finished-trip mean wait (s)",
        title="Randomized training episodes (unfiltered, rolling 5)",
    )
    axes[1].set(ylabel="Complete episode fraction", xlabel="Decision steps", ylim=(-0.03, 1.03))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "training_diagnostics.png", dpi=160)
    plt.close(fig)

    lines = [
        f"# {config.label} 验证结果（非最终测试）",
        "",
        f"训练种子 {config.training_seeds}，每模型 {config.training_steps} 步，"
        f"需求 {config.env.demand_seconds} 秒。",
        f"共 {len(training)} 个模型，纯训练合计 {quality['training_seconds'] / 60:.2f} 分钟。",
        f"验证完整回合 {quality['complete_cases']}/{len(frame)}，"
        f"滞留车辆计数 {quality['unfinished_vehicles']}（跨方法/候选求和）。",
        "",
        "以下均为验证选参后的描述性结果，不能作为最终泛化性能。P95 为回合 P95 的平均。",
        "任一场景有不完整回合时该候选/场景性能留空；完整性仍显示。",
        "",
        "| 场景 | 方法 | 候选 | 完整回合 | 平均等待 s | 平均回合 P95 s | 超 90 秒比例 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in chosen.iterrows():
        metrics = (
            f"{row.mean_wait:.2f} | {row.p95_wait:.2f} | {row.over_threshold_fraction:.4f}"
            if row.all_cases_complete
            else "— | — | —"
        )
        lines.append(
            f"| {row.scenario} | {row.method} | {row.candidate} | "
            f"{row.valid_cases}/{row.cases} | {metrics} |"
        )
    lines += [
        "",
        "完整候选、失败回合及成对差异见同目录 CSV；所有数值来自真实运行。",
        "单训练种子不能估计训练方差；随机训练回合曲线不能证明收敛。",
        "预留最终测试种子尚未进行性能评估。",
    ]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return quality


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/stability-50k")
    parser.add_argument("--destination", default="docs/results/stability-50k")
    args = parser.parse_args()
    destination = (project_root() / args.destination).resolve()
    if not destination.is_relative_to(project_root() / "docs/results"):
        parser.error("Destination must be under docs/results")
    print(json.dumps(export(output_directory(args.run), destination), indent=2))
