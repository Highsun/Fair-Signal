"""Combine two frozen validation studies with explicit source and replay evidence."""

from __future__ import annotations

import json
import shutil

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
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    root = project_root()
    destination = root / "docs/results/factorial-50k"
    destination.mkdir(parents=True, exist_ok=True)
    proof_path = root / "outputs/legacy-fair-state-check/check.json"
    proof = json.loads(proof_path.read_text())
    if not (
        proof["all_metrics_identical"]
        and proof["cases"] == 516
        and proof["viewer_sha256"] == source_digest()
    ):
        raise ValueError("Exact historical replay proof is missing")
    frames, sources, model_checks = [], [], []
    reference_config = None
    for name, expected_models, expected_cases in (
        ("improvement-50k", 15, 270),
        ("fair-state-50k", 3, 45),
    ):
        run = root / "outputs" / name
        archived = root / "docs/results" / name
        saved = json.loads((run / "config.json").read_text())
        config = saved["config"]
        if reference_config is None:
            reference_config = config
        shared_keys = set(config) - {"label", "methods"}
        for key in sorted(shared_keys):
            if config[key] != reference_config[key]:
                raise ValueError(f"Unmatched experiment setting: {key}")
        if (run / "test.csv").exists() or (run / "cases/test").exists():
            raise ValueError("Reserved final test was used")
        if (run / "validation.csv").read_text() != (archived / "validation.csv").read_text():
            raise ValueError("Frozen exported validation changed")
        if saved != json.loads((archived / "config.json").read_text()):
            raise ValueError("Frozen exported configuration changed")
        frame = pd.read_csv(run / "validation.csv")
        quality = json.loads((archived / "quality.json").read_text())
        if quality["trained_models"] != expected_models or len(frame) != expected_cases:
            raise ValueError("Unexpected study size")
        if not frame.valid_complete.all() or not frame.split.eq("validation").all():
            raise ValueError("Incomplete or non-validation cases")
        if (
            frame[["running", "pending_insertion", "collision_events", "teleport_events"]]
            .sum()
            .sum()
        ):
            raise ValueError("Validation failures")
        if frame.duplicated(
            ["method", "candidate", "training_seed", "scenario", "traffic_seed"]
        ).any():
            raise ValueError("Duplicate validation records")
        records = sorted((run / "models").glob("*/*/*/training.json"))
        if len(records) != expected_models:
            raise ValueError("Missing models")
        for path in records:
            record = json.loads(path.read_text())
            model_path = path.with_name("model.zip")
            if (
                record["source_sha256"] != saved["source_sha256"]
                or record["model_sha256"] != sha256(model_path)
                or record["decision_steps"] != 50000
                or record["gradient_updates"] != 12250
                or not record["save_reload_action_matches"]
            ):
                raise ValueError("Incomplete training or identity mismatch")
            model = DQN.load(model_path, device="cpu")
            if not all(
                torch.isfinite(value).all().item() for value in model.policy.state_dict().values()
            ):
                raise ValueError("Nonfinite online or target weights")
            events = list(path.parent.glob("tensorboard/**/events.out.tfevents.*"))
            # SB3's tensorboard_log path is recorded by the directory layout.
            if not events:
                events = list(path.parent.glob("**/events.out.tfevents.*"))
            if not events:
                raise ValueError("Missing TensorBoard events")
            model_checks.append(
                {"run": name, **record, "weights_finite": True, "tensorboard_events": len(events)}
            )
        frame["source_run"] = name
        frames.append(frame)
        sources.append(
            {
                "run": name,
                "source_sha256": saved["source_sha256"],
                "validation_sha256": sha256(run / "validation.csv"),
                "config_sha256": sha256(run / "config.json"),
                "cases": len(frame),
                "models": len(records),
                "execution": json.loads((run / "execution.json").read_text()),
                "provenance": json.loads((run / "provenance.json").read_text()),
            }
        )
    frame = pd.concat(frames, ignore_index=True)
    if not frame.groupby(["scenario", "traffic_seed"]).route_sha256.nunique().eq(1).all():
        raise ValueError("Route mismatch between studies")
    frame.to_csv(destination / "validation.csv", index=False)
    per_scenario, summary = hierarchical_summary(frame)
    per_scenario.rename(columns={"test_cases": "validation_cases"}).to_csv(
        destination / "per_scenario_model.csv", index=False
    )
    summary.rename(columns={"test_cases": "validation_cases"}).to_csv(
        destination / "scenario_summary.csv", index=False
    )
    normalized = frame.assign(training_seed=frame.training_seed.fillna(-1).astype(int))
    per_model = normalized.groupby(["method", "training_seed"])[METRICS].mean().reset_index()
    if not normalized.groupby(["method", "training_seed"]).size().eq(15).all():
        raise ValueError("Each model must use the same 15 validation cases")
    per_model.to_csv(destination / "per_model.csv", index=False)
    global_summary = per_model.groupby("method")[METRICS].agg(["mean", "std"])
    global_summary.columns = [f"{metric}_{stat}" for metric, stat in global_summary.columns]
    global_summary.to_csv(destination / "global_summary.csv")
    pairs = []
    for method, reference in (
        ("dqn_fair", "dqn_queue"),
        ("dqn_dense", "dqn_fair"),
        ("dqn_dense_state", "dqn_dense"),
        ("dqn_fair_state", "dqn_fair"),
        ("dqn_fair_state", "dqn_dense_state"),
        ("dqn_fair_state", "dqn_queue"),
        ("dqn_fair_state", "aging"),
    ):
        for seed in (11, 22, 33):
            a = frame[(frame.method == method) & (frame.training_seed == seed)]
            b = frame[frame.method == reference]
            if reference.startswith("dqn_"):
                b = b[b.training_seed == seed]
            merged = a.merge(
                b, on=["scenario", "traffic_seed"], suffixes=("_a", "_b"), validate="one_to_one"
            )
            if len(merged) != 15 or not merged.route_sha256_a.eq(merged.route_sha256_b).all():
                raise ValueError("Unmatched paired cases")
            pairs.append(
                {
                    "method": method,
                    "reference": reference,
                    "training_seed": seed,
                    "paired_cases": 15,
                    **{
                        metric: float((merged[f"{metric}_a"] - merged[f"{metric}_b"]).mean())
                        for metric in METRICS
                    },
                }
            )
    pd.DataFrame(pairs).to_csv(destination / "paired_by_seed.csv", index=False)
    save_json(
        destination / "manifest.json",
        {
            "purpose": "development_validation_only",
            "sources": sources,
            "current_source_sha256": source_digest(),
            "historical_replay_cases": proof["cases"],
            "validation_cases": len(frame),
            "complete_cases": int(frame.valid_complete.sum()),
            "models": len(model_checks),
            "test_performance_evaluated": False,
            "note": (
                "The fair-state cell follows observed development results; "
                "no final test or additional weight search."
            ),
        },
    )
    save_json(destination / "model_checks.json", model_checks)
    shutil.copy2(proof_path, destination / "legacy_check.json")
    shutil.copy2(proof_path.with_name("cases.csv"), destination / "legacy_cases.csv")
    methods = [
        "dqn_queue",
        "dqn_fair",
        "dqn_fair_state",
        "dqn_dense",
        "dqn_dense_state",
        "actuated",
        "aging",
    ]
    labels = [
        "Queue DQN",
        "Fair / legacy",
        "Fair / service",
        "Dense / legacy",
        "Dense / service",
        "Actuated",
        "Aging",
    ]
    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, label, scale in zip(
        axes,
        ["mean_wait", "p95_wait", "over_threshold_fraction"],
        ["Mean wait (s)", "Mean case P95 (s)", "Wait >90s (%)"],
        [1, 1, 100],
        strict=True,
    ):
        for index, method in enumerate(methods):
            values = per_model[per_model.method == method][metric].to_numpy() * scale
            ax.scatter(
                index + np.linspace(-0.12, 0.12, len(values)), values, color="tab:blue", s=30
            )
            ax.plot([index - 0.24, index + 0.24], [values.mean()] * 2, color="black", linewidth=2)
        ax.set(xticks=range(len(methods)), xticklabels=labels, ylabel=label)
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Development validation: dots = training seeds, bars = means; 15 matched cases/model"
    )
    fig.tight_layout()
    fig.savefig(figures / "factorial_comparison.png", dpi=170)
    plt.close(fig)
    print(
        global_summary[
            [
                f"{metric}_{stat}"
                for metric in ("mean_wait", "p95_wait", "over_threshold_fraction")
                for stat in ("mean", "std")
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()
