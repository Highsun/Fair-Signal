"""Bounded parallel seed study; all performance evaluations use validation routes."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

from fair_signal.analysis import hierarchical_summary
from fair_signal.config import load_config
from fair_signal.demand import build_network, save_json
from fair_signal.experiment import candidates, case_path, evaluate_case, initialize, train, tune
from fair_signal.runtime import output_directory, project_root


def seed_worker(config_path: str, output: str, seed: int) -> int:
    config = replace(load_config(Path(config_path)), training_seeds=[seed])
    root = Path(output)
    network = root / "network/intersection.net.xml"
    for method in config.methods:
        if not method.startswith("dqn_"):
            continue
        # Validate each finished model while other seed workers continue training.
        one = replace(config, methods=[method])
        train(one, root, network)
        for index, parameters in enumerate(candidates(config, method)):
            for scenario in config.scenarios:
                for traffic_seed in config.validation_seeds:
                    evaluate_case(
                        config,
                        root,
                        network,
                        method,
                        index,
                        parameters,
                        seed,
                        "validation",
                        scenario,
                        traffic_seed,
                    )
        print(f"Model trained and validated: {method} seed={seed}", flush=True)
    return seed


def run(config_path: Path, output: Path, workers: int) -> None:
    config = load_config(config_path)
    initialize(config, output)
    network = build_network(output / "network")
    # Do not even generate reserved final-test routes in this development runner.
    for scenario in config.scenarios:
        for seed in config.validation_seeds:
            case_path(output, config, "validation", scenario, seed)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(seed_worker, str(config_path), str(output), seed)
            for seed in config.training_seeds
        ]
        for method in config.methods:
            if method.startswith("dqn_"):
                continue
            for index, parameters in enumerate(candidates(config, method)):
                for scenario in config.scenarios:
                    for seed in config.validation_seeds:
                        evaluate_case(
                            config,
                            output,
                            network,
                            method,
                            index,
                            parameters,
                            None,
                            "validation",
                            scenario,
                            seed,
                        )
            print(f"Traditional validation finished: {method}", flush=True)
        for future in as_completed(futures):
            print(f"Training seed finished: {future.result()}", flush=True)
    tune(config, output, network)
    save_json(
        output / "execution.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "training_workers": workers,
            "test_performance_evaluated": False,
        },
    )


def summarize_pairs(output: Path, destination: Path) -> None:
    frame = pd.read_csv(output / "validation.csv")
    selected = json.loads((output / "selection.json").read_text())
    chosen = frame[
        frame.apply(lambda row: row.candidate == selected[row.method]["candidate"], axis=1)
    ]
    per_model, summary = hierarchical_summary(chosen)
    for table, name in ((per_model, "per_model_summary.csv"), (summary, "method_summary.csv")):
        table.rename(columns={"test_cases": "validation_cases"}).to_csv(
            destination / name, index=False
        )
    rows = []
    for method, reference in [
        ("dqn_dense", "dqn_fair"),
        ("dqn_dense", "dqn_queue"),
        ("dqn_dense_state", "dqn_dense"),
        ("dqn_dense_state", "dqn_queue"),
        ("dqn_dense_state", "aging"),
        ("dqn_dense", "aging"),
    ]:
        for seed in sorted(chosen[chosen.method == method].training_seed.unique()):
            a = chosen[(chosen.method == method) & (chosen.training_seed == seed)]
            b = chosen[chosen.method == reference]
            if reference.startswith("dqn_"):
                b = b[b.training_seed == seed]
            matched = a.merge(
                b, on=["scenario", "traffic_seed"], suffixes=("_a", "_b"), validate="one_to_one"
            )
            if len(matched) != len(a):
                raise ValueError("Missing matched reference cases")
            complete = bool(matched.valid_complete_a.all() and matched.valid_complete_b.all())
            record = {
                "method": method,
                "reference": reference,
                "training_seed": int(seed),
                "paired_cases": len(matched),
                "all_cases_complete": complete,
            }
            for metric in ("mean_wait", "p95_wait", "over_threshold_fraction"):
                record[f"delta_{metric}"] = (
                    float((matched[f"{metric}_a"] - matched[f"{metric}_b"]).mean())
                    if complete
                    else float("nan")
                )
            rows.append(record)
    pd.DataFrame(rows).to_csv(destination / "paired_by_seed.csv", index=False)
    # Compare model means across seeds; never treat repeated vehicles as training replicas.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, name in (
        ("mean_wait", "mean_wait_by_seed.png"),
        ("p95_wait", "p95_wait_by_seed.png"),
    ):
        fig, ax = plt.subplots(figsize=(11, 4))
        order = list(chosen.method.unique())
        for index, method in enumerate(order):
            group = per_model[per_model.method == method]
            # Average matched scenarios within each trained model.
            values = group.groupby("training_seed")[metric].mean()
            if not group.all_cases_complete.all():
                continue
            ax.scatter([index] * len(values), values, s=30)
            ax.plot([index - 0.2, index + 0.2], [values.mean()] * 2, color="black")
        ax.set(
            xticks=range(len(order)),
            xticklabels=order,
            ylabel=f"{metric} (s)",
            title="Validation: each dot is one trained model mean",
        )
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(destination / "figures" / name, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--config", default="configs/improvement.yaml")
    parser.add_argument("--output", default="outputs/improvement-50k")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    root = output_directory(args.output)
    if args.command == "run":
        path = (project_root() / args.config).resolve()
        if not path.is_relative_to(project_root()):
            parser.error("Configuration must stay inside the repository")
        run(path, root, args.workers)
    else:
        destination = project_root() / "docs/results/improvement-50k"
        summarize_pairs(root, destination)
