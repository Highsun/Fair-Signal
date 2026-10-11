import random

from fair_signal.config import EnvConfig
from fair_signal.signals import Signal


def test_green_clearance_and_early_request():
    signal = Signal(EnvConfig())
    signal.request(2)
    assert signal.mode == "green" and signal.blocked_requests == 1
    for _ in range(10):
        signal.before_second()
        signal.after_second()
    signal.request(2)
    states = []
    for _ in range(4):
        states.append(signal.before_second())
        signal.after_second()
    assert states == ["yyrrrryyrrrr"] * 3 + ["r" * 12]
    assert signal.phase == 2 and signal.mode == "green" and signal.elapsed == 0


def test_max_green_forces_switch_even_if_policy_holds():
    signal = Signal(EnvConfig(max_green=17))
    states = []
    for _ in range(18):
        signal.request(0)
        states.append(signal.before_second())
        signal.after_second()
    assert states[:17] == ["GGrrrrGGrrrr"] * 17
    assert states[17] == "yyrrrryyrrrr"
    assert signal.forced_switches == 1 and signal.target == 1


def test_arbitrary_requests_never_bypass_clearance_or_green_limits():
    rng = random.Random(5)
    config = EnvConfig()
    signal = Signal(config)
    intervals, active_mode, duration = [], signal.mode, 0
    for second in range(10000):
        if second % config.decision_seconds == 0:
            signal.request(rng.randrange(4))
        signal.before_second()
        if signal.mode != active_mode:
            intervals.append((active_mode, duration))
            active_mode, duration = signal.mode, 0
        duration += 1
        signal.after_second()
    for mode, duration in intervals:
        if mode == "green":
            assert config.min_green <= duration <= config.max_green
        elif mode == "yellow":
            assert duration == config.yellow_seconds
        else:
            assert duration == config.all_red_seconds
