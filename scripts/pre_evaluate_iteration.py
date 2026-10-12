"""Overlap frozen-controller development replay with independent new training."""

import argparse
import json
import time
from pathlib import Path

from arterial_study import pool_map
from iteration_study import (
    BASELINES,
    METHODS,
    NEW,
    OLD,
    evaluate_one,
    initialize,
    prepare_cases,
    read_config,
    source_digest,
)

from fair_signal.demand import save_json
from fair_signal.runtime import output_directory


def run(config_path, output, workers, methods):
    config = read_config(config_path)
    if (output / "study.json").exists():
        study = json.loads((output / "study.json").read_text())
        if study["config"] != config or study["source_sha256"] != source_digest():
            raise ValueError("Existing study identity differs")
    else:
        initialize(config, output)
    for method in methods:
        if method in NEW and any(
            not (output / f"models/{method}/seed_{seed}/training.json").exists()
            for seed in config["training_seeds"]
        ):
            raise ValueError("Evaluate only completed model families")
    started = time.perf_counter()
    records = prepare_cases(config, output, "development")
    jobs = [
        dict(output=str(output), split="development", method=method, seed=seed, case=case)
        for method in methods
        for seed in ([None] if method in BASELINES else config["training_seeds"])
        for case in records
    ]
    rows = pool_map(evaluate_one, jobs, workers)
    save_json(
        output / ("execution_pre_evaluate_" + "_".join(methods) + ".json"),
        {
            "workers": workers,
            "wall_seconds": time.perf_counter() - started,
            "cases": len(rows),
            "complete_cases": sum(r["valid_complete"] for r in rows),
            "scope": "frozen methods; development only; reused by identity-checked full evaluation",
            "methods": methods,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/iteration.yaml")
    parser.add_argument("--output", default="outputs/iteration-50k")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=[*BASELINES, *OLD])
    args = parser.parse_args()
    run(Path(args.config), output_directory(args.output), args.workers, args.methods)
