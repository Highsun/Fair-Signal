"""Check the boundary that keeps experiment artifacts in the code repository."""

import pytest

from fair_signal.cli import main
from fair_signal.runtime import output_directory, project_root


def test_experiment_artifacts_cannot_escape_outputs():
    for value in ("../outside", "outputs/../../outside", "/private/tmp/fair-signal-outside"):
        with pytest.raises(ValueError, match="inside"):
            output_directory(value)


def test_output_symlink_cannot_escape_repository(tmp_path):
    link = project_root() / "outputs" / "test-output-link"
    link.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="inside"):
            output_directory("outputs/test-output-link/experiment")
    finally:
        link.unlink()


def test_smoke_command_rejects_accidental_large_training_run():
    with pytest.raises(SystemExit) as exc:
        main(["smoke", "--steps", "300000"])
    assert exc.value.code == 2
