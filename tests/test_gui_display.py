import runpy
import subprocess
from pathlib import Path

import pytest

module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/gui_display.py"))
resolve_display = module["resolve_display"]
command_output = module["command_output"]


@pytest.mark.parametrize("inherited", ["", "stale-display"])
def test_stale_launchctl_display_falls_back_to_reachable_local_socket(monkeypatch, inherited):
    monkeypatch.setenv("DISPLAY", inherited)
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    monkeypatch.setattr(Path, "is_socket", lambda _: True)
    calls = []

    def run(args):
        calls.append(args)
        if args[0] == "launchctl":
            return "stale-display"
        if args[-1] == ":0":
            return "display information"
        return None

    monkeypatch.setitem(resolve_display.__globals__, "command_output", run)
    assert resolve_display() == ":0"
    assert ["open", "-a", "XQuartz"] not in calls
    assert calls.count([module["XDPYINFO"], "-display", "stale-display"]) == 1
    assert module["os"].environ["DISPLAY"] == inherited


def test_reachable_explicit_display_is_preserved(monkeypatch):
    monkeypatch.setenv("DISPLAY", "working-display")
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    monkeypatch.setattr(Path, "is_socket", lambda _: True)
    monkeypatch.setitem(resolve_display.__globals__, "command_output", lambda args: "")
    assert resolve_display() == "working-display"


def test_unavailable_display_fails_before_sumo_launch(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    monkeypatch.setattr(Path, "is_socket", lambda _: False)
    calls = []

    def run(args):
        calls.append(args)
        return "" if args[0] == "open" else None

    monkeypatch.setitem(resolve_display.__globals__, "command_output", run)
    with pytest.raises(RuntimeError, match="显示服务仍不可连接"):
        resolve_display(wait_seconds=0)
    assert calls.count(["open", "-a", "XQuartz"]) == 1


def test_display_probe_has_bounded_wait(monkeypatch):
    def run(args, **kwargs):
        assert kwargs["timeout"] == 2
        raise subprocess.TimeoutExpired(args, 2)

    monkeypatch.setattr(subprocess, "run", run)
    assert command_output([module["XDPYINFO"], "-display", "unreachable"]) is None
