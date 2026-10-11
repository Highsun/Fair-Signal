"""Environment checks, reproducible experiment pipeline and policy replay."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys

from fair_signal.runtime import configure_sumo, output_directory, project_root


def doctor() -> int:
    """Print actual package and SUMO versions, with a failing status on mismatch."""
    report: dict = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": {},
        "errors": [],
    }
    for name in (
        "torch",
        "stable-baselines3",
        "sumo-rl",
        "gymnasium",
        "traci",
        "sumolib",
        "numpy",
        "pandas",
        "matplotlib",
        "tensorboard",
        "pytest",
        "ruff",
    ):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Missing package: {name}")
    try:
        home = configure_sumo()
        binary = shutil.which("sumo")
        if binary is None:
            raise FileNotFoundError("sumo binary not found")
        completed = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, check=True, timeout=20
        )
        version_line = completed.stdout.splitlines()[0]
        report.update(sumo_home=str(home), sumo_binary=binary, sumo_version=version_line)
        version = re.search(r"\b\d+\.\d+\.\d+\b", version_line)
        if version is None or version.group() != "1.28.0":
            report["errors"].append("Expected SUMO 1.28.0 to match the pinned TraCI client.")
        for name in ("traci", "sumolib"):
            if report["packages"].get(name) != "1.28.0":
                report["errors"].append(f"Expected {name} 1.28.0")
        report["sumo_gui_binary"] = shutil.which("sumo-gui")
    except (OSError, subprocess.SubprocessError) as exc:
        report["errors"].append(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fair Signal traffic control experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check package versions and global SUMO")
    smoke_parser = subparsers.add_parser("smoke", help="Run a bounded SUMO/DQN integration check")
    smoke_parser.add_argument("--steps", type=int, default=128)
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.add_argument("--output", default="outputs/smoke")
    run_parser = subparsers.add_parser("run", help="Reproducible train/validate/test pipeline")
    run_parser.add_argument("--config", default="configs/pilot.yaml")
    run_parser.add_argument("--output", default="outputs/pilot")
    run_parser.add_argument(
        "--stage", choices=["prepare", "train", "tune", "test", "all"], default="all"
    )
    analyze_parser = subparsers.add_parser("analyze", help="Regenerate summaries and figures")
    analyze_parser.add_argument("--run", default="outputs/pilot")
    replay_parser = subparsers.add_parser(
        "replay", help="Load frozen policy and replay a test case"
    )
    replay_parser.add_argument("--run", default="outputs/pilot")
    replay_parser.add_argument("--method", default="dqn_fair")
    replay_parser.add_argument("--seed", type=int, default=11)
    replay_parser.add_argument(
        "--scenario", choices=["balanced", "imbalanced", "changing_peak"], default="imbalanced"
    )
    replay_parser.add_argument("--traffic-seed", type=int, default=2001)
    replay_parser.add_argument("--output", default="outputs/replay")
    replay_parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command != "smoke":
        try:
            if args.command == "run":
                from fair_signal.experiment import run_experiment

                config_path = (project_root() / args.config).resolve()
                if not config_path.is_relative_to(project_root()):
                    raise ValueError("Configuration must be inside this repository")
                result = {"output": str(run_experiment(config_path, args.output, args.stage))}
            elif args.command == "analyze":
                from fair_signal.analysis import analyze

                result = analyze(output_directory(args.run))
            else:
                from fair_signal.replay import replay

                run = output_directory(args.run)
                destination = output_directory(args.output)
                if destination == run or destination.is_relative_to(run):
                    raise ValueError("Replay output must be outside the original run directory")
                result = replay(
                    run,
                    args.method,
                    args.seed,
                    args.scenario,
                    args.traffic_seed,
                    destination,
                    args.gui,
                )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            print(f"{args.command} failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not 64 <= args.steps <= 2000:
        parser.error("smoke --steps must be between 64 and 2000; this is not formal training")
    try:
        from fair_signal.smoke import run_smoke

        report = run_smoke(args.steps, args.seed, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Smoke check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
