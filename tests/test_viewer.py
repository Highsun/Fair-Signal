import json

import pytest

from fair_signal.environment import TrafficEnv, gui_arguments
from fair_signal.replay import compatible_source


def test_gui_waits_for_play_and_can_opt_into_autoplay():
    args = gui_arguments(200, True)
    assert "--start" not in args
    assert args[args.index("--delay") + 1] == "200"
    assert "--start" in gui_arguments(500, False)


@pytest.mark.parametrize("delay", [-1, 10001, float("nan"), 0.5])
def test_invalid_gui_delay_rejected(delay):
    with pytest.raises(ValueError, match="GUI delay"):
        gui_arguments(delay, True)


def test_compatibility_accepts_only_exact_verified_hash_pair(tmp_path, monkeypatch):
    import fair_signal.replay as module

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/replay-compatibility.json").write_text(
        json.dumps({"verified_pairs": [{"source_sha256": "old", "viewer_sha256": "new"}]})
    )
    monkeypatch.setattr(module, "project_root", lambda: tmp_path)
    monkeypatch.setattr(module, "source_digest", lambda: "new")
    assert compatible_source("new") and compatible_source("old")
    assert not compatible_source("unknown")
    monkeypatch.setattr(module, "source_digest", lambda: "another-edit")
    assert not compatible_source("old")


def test_interrupted_paused_gui_terminates_process_before_closing_socket():
    events = []

    class Process:
        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append("wait")

    class Connection:
        def close(self):
            events.append("close")

    env = TrafficEnv.__new__(TrafficEnv)
    env.gui_process, env.connection, env.gui = Process(), Connection(), True
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        env.close()
    assert events == ["terminate", "wait", "close"]
    assert env.gui_process is env.connection is None
