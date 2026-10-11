"""Re-evaluate historical frozen models without relaxing their source identity checks."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from fair_signal.config import EnvConfig
from fair_signal.controllers import TraditionalController
from fair_signal.demand import save_json, sha256
from fair_signal.environment import TrafficEnv
from fair_signal.experiment import source_digest
from fair_signal.objectives import policy_options
from fair_signal.runtime import output_directory


def check_case(job: dict) -> dict:
    import torch
    from stable_baselines3 import DQN

    torch.set_num_threads(1)
    run = Path(job["run"])
    row = job["row"]
    seed = int(row["training_seed"]) if pd.notna(row["training_seed"]) else None
    candidate = int(row["candidate"])
    method, scenario, traffic = row["method"], row["scenario"], int(row["traffic_seed"])
    source = (
        run
        / "evaluation"
        / job["split"]
        / method
        / f"candidate_{candidate}"
        / f"model_{seed}"
        / scenario
        / str(traffic)
    )
    identity = json.loads((source / "identity.json").read_text())
    route = run / "cases" / job["split"] / scenario / str(traffic) / "vehicles.rou.xml"
    network = run / "network/intersection.net.xml"
    if sha256(route) != identity["route_sha256"] or sha256(network) != identity["network_sha256"]:
        raise ValueError("Historical simulator inputs changed")
    policy = None
    if seed is not None:
        model = run / "models" / method / f"candidate_{candidate}" / f"seed_{seed}/model.zip"
        if sha256(model) != identity["model_sha256"]:
            raise ValueError("Historical model changed")
        policy = DQN.load(model, device="cpu")
    params = identity["parameters"]
    controller = TraditionalController(method, params)
    destination = Path(job["output"]) / run.name / str(job["index"])
    audit = job["split"] == "validation" and (
        method == "dqn_queue" or (method == "dqn_fair" and candidate == 2)
    )
    env = TrafficEnv(
        EnvConfig(**identity["environment"]),
        network,
        destination,
        **policy_options(method, params),
        route_file=route,
        sumo_seed=traffic,
        trace=audit,
    )
    try:
        observation, _ = env.reset(seed=traffic)
        while True:
            action = (
                int(policy.predict(observation, deterministic=True)[0])
                if policy
                else controller.predict(env)
            )
            observation, _, terminal, truncated, info = env.step(action)
            if terminal or truncated:
                actual = info["metrics"]
                break
    finally:
        env.close()
    expected = json.loads((source / "episode_00000/metrics.json").read_text())
    differences = [key for key in expected if expected[key] != actual[key]]
    if differences:
        raise ValueError(f"Legacy behavior changed: {source}, metrics={differences}")
    result = {
        "run": run.name,
        "method": method,
        "candidate": candidate,
        "scenario": scenario,
        "traffic_seed": traffic,
        "all_metrics_identical": True,
    }
    if audit:
        trace = pd.DataFrame(env.trace_rows)
        trace = trace[trace.second <= env.config.demand_seconds]
        result.update(
            {
                "audit_queue_cost": float(trace.queue_cost.mean()),
                "audit_fair_cost": float(trace.fair_cost.mean()),
                "audit_head_blind_fraction": float((trace.head_blind_spot > 0).mean()),
                "audit_blind_spot_max_seconds": float(trace.head_blind_spot.max()),
                "audit_active_fraction": float((trace.fair_cost > 0).mean()),
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/legacy-improvement-check")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--include-improvement", action="store_true")
    args = parser.parse_args()
    output = output_directory(args.output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    runs = [("pilot", "test"), ("stability-50k", "validation")]
    if args.include_improvement:
        runs.append(("improvement-50k", "validation"))
    for name, split in runs:
        run = output_directory(f"outputs/{name}")
        frame = pd.read_csv(run / f"{split}.csv")
        jobs.extend(
            {"run": str(run), "split": split, "row": row, "output": str(output), "index": index}
            for index, row in enumerate(frame.to_dict(orient="records"))
        )
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for index, result in enumerate(pool.map(check_case, jobs)):
            results.append(result)
            if (index + 1) % 30 == 0:
                print(f"Identical historical cases: {index + 1}/{len(jobs)}", flush=True)
    pd.DataFrame(results).to_csv(output / "cases.csv", index=False)
    save_json(
        output / "check.json",
        {
            "viewer_sha256": source_digest(),
            "cases": len(results),
            "all_metrics_identical": True,
            "pilot_cases": 36,
            "stability_cases": 210,
            "improvement_cases": 270 if args.include_improvement else 0,
        },
    )
