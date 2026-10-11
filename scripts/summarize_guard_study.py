"""Compare distinct hybrid controllers; preserve invalid unguarded outcomes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_signal.analysis import METRICS, hierarchical_summary
from fair_signal.demand import save_json, sha256
from fair_signal.experiment import source_digest
from fair_signal.runtime import project_root


def main() -> None:
    root = project_root()
    run = root / "outputs/service-guard-150k"
    destination = root / "docs/results/service-guard-150k"
    study = json.loads((run / "study.json").read_text())
    if (
        study["runner_sha256"] != sha256(root / "scripts/guard_study.py")
        or study["guard_sha256"] != sha256(root / "scripts/service_guard.py")
        or study["source_sha256"] != source_digest()
        or study["threshold"] != 180
    ):
        raise ValueError("Guard implementation or fixed threshold changed")
    guarded = pd.read_csv(run / "validation.csv")
    expected = {
        (method + "_guard", seed, scenario, traffic)
        for method in ("dqn_queue", "dqn_fair", "actuated")
        for seed in ([11, 22, 33] if method.startswith("dqn_") else [-1])
        for scenario in study["base_config"]["config"]["scenarios"]
        for traffic in study["base_config"]["config"]["validation_seeds"]
    }
    keys = ["method", "training_seed", "scenario", "traffic_seed"]
    normalized = guarded.assign(training_seed=guarded.training_seed.fillna(-1).astype(int))
    if (
        len(guarded) != len(expected)
        or set(normalized[keys].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError("Incomplete or duplicated guard validation matrix")
    for row in guarded.itertuples():
        seed = int(row.training_seed) if pd.notna(row.training_seed) else None
        case = run / f"evaluation/{row.method}/model_{seed}/{row.scenario}/{row.traffic_seed}"
        identity = json.loads((case / "identity.json").read_text())
        if any(
            identity[key] != study[key]
            for key in ("runner_sha256", "guard_sha256", "source_sha256", "threshold")
        ):
            raise ValueError("Case code identity changed")
        if identity["route_sha256"] != row.route_sha256:
            raise ValueError("Case route identity mismatch")
        if row.base_method.startswith("dqn_"):
            if identity["model_sha256"] != study["models"][f"{row.base_method}/{seed}"]:
                raise ValueError("Guard changed model weights")
        actual = json.loads((case / "result.json").read_text())
        if bool(actual["valid_complete"]) != row.valid_complete:
            raise ValueError("Case completeness mismatch")
        for metric in [*METRICS, "max_wait", "drain_seconds"]:
            if not np.isclose(actual[metric], getattr(row, metric), rtol=1e-12, atol=1e-12):
                raise ValueError("Case traffic metric mismatch")
        trace = pd.read_csv(case / "episode_00000/trace.csv")
        segment = (
            trace["mode"].ne(trace["mode"].shift()) | trace.phase.ne(trace.phase.shift())
        ).cumsum()
        blocks = trace.groupby(segment).agg(mode=("mode", "first"), duration=("second", "size"))
        full = blocks.iloc[:-1]
        green = full[full["mode"] == "green"].duration
        if not green.between(10, 60).all():
            raise ValueError("Green duration violates shared constraints")
        for mode, duration in (("yellow", 3), ("all_red", 1)):
            if not full[full["mode"] == mode].duration.eq(duration).all():
                raise ValueError("Clearance duration violates shared constraints")
        next_mode = {"green": "yellow", "yellow": "all_red", "all_red": "green"}
        modes = blocks["mode"].tolist()
        if any(next_mode[a] != b for a, b in zip(modes[:-1], modes[1:], strict=True)):
            raise ValueError("Skipped signal clearance stage")
        actions = pd.read_csv(case / "guard_actions.csv")
        if int(actions.overridden.sum()) != row.guard_overrides:
            raise ValueError("Guard intervention count mismatch")
        if actions.loc[actions.overridden, "oldest_service_age"].lt(180).any():
            raise ValueError("Guard intervened below the fixed threshold")
    base = root / "outputs/long-budget-150k/step_150000/validation.csv"
    if sha256(base) != study["base_validation_sha256"]:
        raise ValueError("Base results changed")
    original = pd.read_csv(base)
    if not guarded.split.eq("validation").all():
        raise ValueError("Non-validation records")
    frame = pd.concat([original, guarded], ignore_index=True)
    if not frame.groupby(["scenario", "traffic_seed"]).route_sha256.nunique().eq(1).all():
        raise ValueError("Unmatched guard and baseline routes")
    destination.mkdir(parents=True, exist_ok=True)
    guarded.to_csv(destination / "validation.csv", index=False)
    guarded.loc[~guarded.valid_complete].to_csv(destination / "failure_cases.csv", index=False)
    for name in ("study.json", "execution.json"):
        shutil.copy2(run / name, destination / name)
    per_scenario, scenarios = hierarchical_summary(frame)
    per_scenario.rename(columns={"test_cases": "validation_cases"}).to_csv(
        destination / "per_scenario_model.csv", index=False
    )
    scenarios.rename(columns={"test_cases": "validation_cases"}).to_csv(
        destination / "scenario_summary.csv", index=False
    )
    metrics = [*METRICS, "max_wait", "drain_seconds"]
    frame["training_seed"] = frame.training_seed.fillna(-1).astype(int)
    per_model = frame.groupby(["method", "training_seed"])[metrics].mean()
    complete = frame.groupby(["method", "training_seed"]).valid_complete.all()
    per_model.loc[~complete, metrics] = np.nan
    per_model["all_cases_complete"] = complete
    per_model.reset_index().to_csv(destination / "per_model.csv", index=False)
    summary = []
    for method, values in per_model.groupby("method"):
        valid = bool(values.all_cases_complete.all())
        cases = frame[frame.method == method]
        summary.append(
            {
                "method": method,
                "complete_cases": int(cases.valid_complete.sum()),
                "cases": len(cases),
                "all_cases_complete": valid,
                **{metric: float(values[metric].mean()) if valid else np.nan for metric in metrics},
                **{
                    f"{metric}_training_std": float(values[metric].std(ddof=1)) if valid else np.nan
                    for metric in metrics
                },
            }
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(destination / "global_summary.csv", index=False)
    interventions = (
        guarded.groupby(["method", "training_seed"], dropna=False)[
            [
                "guard_overrides",
                "guard_override_fraction",
                "guard_overrides_demand",
                "guard_override_fraction_demand",
            ]
        ]
        .mean()
        .reset_index()
    )
    interventions["mean_overrides_drain"] = (
        interventions.guard_overrides - interventions.guard_overrides_demand
    )
    interventions.to_csv(destination / "interventions.csv", index=False)
    comparison = []
    for base_method in ("dqn_queue", "dqn_fair", "actuated"):
        a = frame[frame.method == f"{base_method}_guard"]
        b = frame[frame.method == base_method]
        matched = a.merge(b, on=keys[1:], suffixes=("_guard", "_plain"), validate="one_to_one")
        if len(matched) != len(a):
            raise ValueError("Missing unguarded control cases")
        valid = bool(matched.valid_complete_guard.all() and matched.valid_complete_plain.all())
        comparison.append(
            {
                "base_method": base_method,
                "paired_cases": len(matched),
                "rescued_cases": int(
                    (matched.valid_complete_guard & ~matched.valid_complete_plain).sum()
                ),
                "newly_failed_cases": int(
                    (~matched.valid_complete_guard & matched.valid_complete_plain).sum()
                ),
                "both_all_complete": valid,
                **{
                    f"guard_minus_plain_{metric}": float(
                        (matched[f"{metric}_guard"] - matched[f"{metric}_plain"]).mean()
                    )
                    if valid
                    else np.nan
                    for metric in metrics
                },
            }
        )
    pd.DataFrame(comparison).to_csv(destination / "guard_vs_plain.csv", index=False)
    pairs = []
    for reference in ("dqn_queue_guard", "aging", "actuated", "actuated_guard"):
        for seed in (11, 22, 33):
            a = frame[(frame.method == "dqn_fair_guard") & (frame.training_seed == seed)]
            b = frame[frame.method == reference]
            if reference.startswith("dqn_"):
                b = b[b.training_seed == seed]
            matched = a.merge(
                b,
                on=["scenario", "traffic_seed"],
                suffixes=("_fair", "_ref"),
                validate="one_to_one",
            )
            if len(matched) != 15:
                raise ValueError("Missing paired hybrid reference")
            valid = bool(matched.valid_complete_fair.all() and matched.valid_complete_ref.all())
            pairs.append(
                {
                    "reference": reference,
                    "training_seed": seed,
                    "paired_cases": 15,
                    "all_cases_complete": valid,
                    **{
                        metric: float((matched[f"{metric}_fair"] - matched[f"{metric}_ref"]).mean())
                        if valid
                        else np.nan
                        for metric in metrics
                    },
                }
            )
    pd.DataFrame(pairs).to_csv(destination / "fair_guard_minus_reference.csv", index=False)
    save_json(
        destination / "quality.json",
        {
            "purpose": "posthoc_development_hybrid_validation",
            "threshold": 180,
            "validation_cases": len(guarded),
            "complete_cases": int(guarded.valid_complete.sum()),
            "unfinished_vehicles": int(guarded.running.sum()),
            "pending_insertion": int(guarded.pending_insertion.sum()),
            "collisions": int(guarded.collision_events.sum()),
            "teleports": int(guarded.teleport_events.sum()),
            "signal_constraints_checked_cases": len(guarded),
            "training_performed": False,
            "test_evaluated": False,
            "summarizer_sha256": sha256(Path(__file__)),
            "raw_validation_sha256": sha256(run / "validation.csv"),
        },
    )
    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    order = [
        "dqn_queue",
        "dqn_fair",
        "dqn_queue_guard",
        "dqn_fair_guard",
        "actuated",
        "actuated_guard",
        "aging",
    ]
    labels = [
        "Queue",
        "Fair",
        "Queue + guard",
        "Fair + guard",
        "Actuated",
        "Actuated + guard",
        "Aging",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    quality = summary.set_index("method").loc[order]
    axes[0].bar(range(len(order)), quality.complete_cases / quality.cases * 100)
    axes[0].set(ylabel="Complete cases (%)", ylim=(0, 110))
    for index, row in enumerate(quality.itertuples()):
        axes[0].text(
            index,
            row.complete_cases / row.cases * 100 + 2,
            f"{row.complete_cases}/{row.cases}",
            ha="center",
            fontsize=8,
        )
    for ax, metric, label, scale in zip(
        axes[1:],
        ["mean_wait", "p95_wait", "over_threshold_fraction"],
        ["Mean wait (s)", "Mean case P95 (s)", "Wait >90s (%)"],
        [1, 1, 100],
        strict=True,
    ):
        for index, method in enumerate(order):
            values = per_model.loc[method, metric].to_numpy() * scale
            if not quality.loc[method, "all_cases_complete"]:
                continue
            ax.scatter(index + np.linspace(-0.12, 0.12, len(values)), values)
            ax.plot([index - 0.25, index + 0.25], [values.mean()] * 2, color="black")
        ax.set(ylabel=label)
    for ax in axes:
        ax.set(xticks=range(len(order)), xticklabels=labels)
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen 150k models + shared 180s service guard; incomplete performance withheld")
    fig.tight_layout()
    fig.savefig(figures / "guard_comparison.png", dpi=170)
    plt.close(fig)
    lines = [
        "# 共享服务护栏的开发验证结果",
        "",
        "固定 15 万步模型，统一 180 秒请求年龄护栏，无新增训练。",
        "原策略存在未完成回合时，性能均值留空；三种子标准差见 global_summary.csv。",
        "",
        "| 方法 | 完整回合 | 平均等待 s | P95 s | 超 90 秒比例 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        values = (
            f"{row.mean_wait:.2f} | {row.p95_wait:.2f} | {100 * row.over_threshold_fraction:.3f}%"
            if row.all_cases_complete
            else "— | — | —"
        )
        lines.append(f"| {row.method} | {row.complete_cases}/{row.cases} | {values} |")
    lines += [
        "",
        "混合控制与纯 DQN 分开命名。阈值未扫描，未使用最终测试集。",
        "接管次数及阶段分布见 interventions.csv，逐种子对照见 fair_guard_minus_reference.csv。",
    ]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        summary[
            [
                "method",
                "complete_cases",
                "cases",
                "mean_wait",
                "p95_wait",
                "over_threshold_fraction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
