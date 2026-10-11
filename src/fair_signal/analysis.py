"""Hierarchical summaries and portable static figures from actual run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_signal.demand import save_json

METRICS = [
    "mean_wait",
    "p95_wait",
    "over_threshold_fraction",
    "mean_travel_time",
    "mean_depart_delay",
    "mean_system_time",
    "mean_queue",
    "peak_queue",
    "throughput_demand_window",
    "switches",
    "forced_switches",
    "wait_n",
    "wait_s",
    "wait_e",
    "wait_w",
]


def hierarchical_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["training_seed"] = frame.training_seed.fillna(-1).astype(int)
    # Test cases first averaged within a trained model. Vehicles are never replicas.
    per_model = frame.groupby(["method", "scenario", "training_seed"])[METRICS].mean().reset_index()
    quality = (
        frame.groupby(["method", "scenario", "training_seed"])
        .valid_complete.agg(all_cases_complete="all", valid_cases="sum", test_cases="count")
        .reset_index()
    )
    per_model = per_model.merge(quality, on=["method", "scenario", "training_seed"])
    per_model.loc[~per_model.all_cases_complete, METRICS] = np.nan
    records = []
    for (method, scenario), group in per_model.groupby(["method", "scenario"]):
        record = {
            "method": method,
            "scenario": scenario,
            "trained_models": len(group) if method.startswith("dqn_") else 0,
            "all_cases_complete": bool(group.all_cases_complete.all()),
            "valid_cases": int(group.valid_cases.sum()),
            "test_cases": int(group.test_cases.sum()),
        }
        for metric in METRICS:
            record[metric] = group[metric].mean() if record["all_cases_complete"] else np.nan
            record[f"{metric}_training_std"] = (
                group[metric].std(ddof=1)
                if method.startswith("dqn_") and len(group) > 1 and record["all_cases_complete"]
                else np.nan
            )
        records.append(record)
    return per_model, pd.DataFrame(records)


def analyze(output: Path) -> dict:
    frame = pd.read_csv(output / "test.csv")
    per_model, summary = hierarchical_summary(frame)
    per_model.to_csv(output / "per_model_summary.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    scenarios = list(frame.scenario.unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios, strict=True):
        subset = summary[summary.scenario == scenario]
        for _, row in subset.iterrows():
            if not row.all_cases_complete:
                continue
            ax.scatter(row.mean_wait, row.p95_wait)
            ax.annotate(
                row.method,
                (row.mean_wait, row.p95_wait),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.set(title=scenario, xlabel="Mean trip stopped wait (s)", ylabel="Mean case P95 wait (s)")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "efficiency_fairness.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios, strict=True):
        subset = summary[summary.scenario == scenario].set_index("method")
        subset[[f"wait_{d}" for d in "nsew"]].plot.bar(ax=ax, width=0.8)
        ax.set(title=scenario, ylabel="Mean trip wait by approach (s)", xlabel="")
        ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(figures / "direction_waits.png", dpi=170)
    plt.close(fig)
    # Raw traffic metric avoids comparing returns from different reward definitions.
    fig, ax = plt.subplots(figsize=(9, 4))
    for path in sorted((output / "models").glob("*/*/*/learning.csv")):
        values = pd.read_csv(path)
        label = "/".join(path.relative_to(output / "models").parts[:-1])
        ax.plot(
            values.step, values.mean_wait.rolling(5, min_periods=1).mean(), label=label, linewidth=1
        )
    ax.set(
        xlabel="Decision steps",
        ylabel="Episode mean wait, rolling 5 (s)",
        title="Randomized training demand; diagnostic, not a matched evaluation",
    )
    if ax.lines:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "learning_wait.png", dpi=170)
    plt.close(fig)
    trace_files = sorted(
        (output / "evaluation/test").glob("*/*/*/changing_peak/*/episode_00000/trace.csv")
    )
    if trace_files:
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        # Same held-out case, first model per method, for a readable representative trace.
        seen = set()
        for path in trace_files:
            method = path.relative_to(output / "evaluation/test").parts[0]
            if method in seen:
                continue
            seen.add(method)
            trace = pd.read_csv(path)
            axes[0].plot(trace.second, trace.queue, label=method, alpha=0.8, linewidth=0.8)
            axes[1].plot(trace.second, trace.head_wait_max, label=method, alpha=0.8, linewidth=0.8)
        axes[0].set(ylabel="Queued vehicles", title="Representative changing-peak test case")
        axes[1].set(ylabel="Max lane-head wait (s)", xlabel="Simulation time (s)")
        axes[0].legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(figures / "traffic_trace.png", dpi=170)
        plt.close(fig)
        chosen = pd.read_csv(trace_files[0])
        fig, ax = plt.subplots(figsize=(10, 2.5))
        green = chosen[chosen["mode"] == "green"]
        ax.scatter(green.second, green.phase, s=2, label="Green")
        clear = chosen[chosen["mode"] != "green"]
        ax.scatter(clear.second, clear.target, s=2, c="orange", label="Clearance -> target")
        ax.set(
            xlabel="Simulation time (s)",
            ylabel="Phase",
            yticks=range(4),
            title=f"Phase trace: {trace_files[0].relative_to(output).parts[2]}",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "phase_trace.png", dpi=170)
        plt.close(fig)
    quality = {
        "test_cases": len(frame),
        "all_cases_complete": bool(frame.valid_complete.all()),
        "incomplete_cases": int((~frame.valid_complete).sum()),
        "unfinished_vehicles": int(frame.running.sum()),
        "pending_insertion": int(frame.pending_insertion.sum()),
        "teleport_events": int(frame.teleport_events.sum()),
        "collision_events": int(frame.collision_events.sum()),
    }
    save_json(output / "quality.json", quality)
    config = json.loads((output / "config.json").read_text())["config"]
    lines = [
        f"# {config['label']} 实际运行结果",
        "",
        f"每个学习配置 {config['training_steps']} 决策步；训练种子 {config['training_seeds']}。",
        f"需求注入 {config['env']['demand_seconds']} 秒。测试质量：{quality}。",
        "",
        "以下是每个回合指标的均值，P95 列为各测试回合 P95 的平均，并非合并车辆后的 P95。",
        "多模型时先在模型内部平均测试车流，再在训练种子之间汇总。",
        "",
        "| 场景 | 方法 | 完整回合 | 平均等待 s | 平均回合 P95 s | 超阈值比例 | 平均队列 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        values = (
            f"{row.mean_wait:.2f} | {row.p95_wait:.2f} | "
            f"{row.over_threshold_fraction:.3f} | {row.mean_queue:.2f}"
            if row.all_cases_complete
            else "— | — | — | —"
        )
        lines.append(
            f"| {row.scenario} | {row.method} | {row.valid_cases}/{row.test_cases} | {values} |"
        )
    lines += [
        "",
        "验证选择见 selection.json；完整原始指标见 test.csv，模型级均值见",
        "per_model_summary.csv，训练种子标准差见 summary.csv。",
        "任何方法/场景只要存在不完整回合，其汇总性能留空；删失原始指标保留在 test.csv。",
        "",
        "短跑用于验证流程和估算预算，不足以证明算法收敛或公平性改进。",
        "即使所有已完成车辆的指标较好，也须同时检查未完成/未插入车辆和入网延迟。",
        "单种子不估计训练方差；不同奖励的累计回报不能直接比较。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return quality
