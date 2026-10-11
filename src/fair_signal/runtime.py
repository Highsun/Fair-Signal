"""Locate SUMO without changing the user's global shell configuration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def configure_sumo() -> Path:
    """Resolve SUMO_HOME and put a matching binary directory on the process PATH."""
    configured = os.environ.get("SUMO_HOME")
    candidates = (
        [Path(configured)]
        if configured
        else [
            Path("/Library/Frameworks/EclipseSUMO.framework/EclipseSUMO/share/sumo"),
            Path("/Library/SUMO"),
            Path("/opt/homebrew/opt/sumo/share/sumo"),
            Path("/usr/share/sumo"),
        ]
    )
    for home in candidates:
        if not home.is_dir():
            continue
        os.environ["SUMO_HOME"] = str(home)
        for binary_dir in (home / "bin", home / "../../bin"):
            if (binary_dir / "sumo").is_file() or (binary_dir / "sumo.exe").is_file():
                os.environ["PATH"] = str(binary_dir.resolve()) + os.pathsep + os.environ["PATH"]
                return home
        if shutil.which("sumo"):
            return home
    raise FileNotFoundError(
        "SUMO installation not found. Install SUMO 1.28.0 and set SUMO_HOME; see docs/setup.md."
    )


def project_root() -> Path:
    """Locate this source checkout. Experiment commands require a checkout."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError("Run experiment commands from a fair-signal source checkout.")
    return root


def output_directory(value: str) -> Path:
    """Keep all generated experiment artifacts inside this repository's outputs/."""
    root = project_root()
    output = (root / value).resolve()
    if not output.is_relative_to((root / "outputs").resolve()):
        raise ValueError("Output directory must be inside this repository's outputs/.")
    return output
