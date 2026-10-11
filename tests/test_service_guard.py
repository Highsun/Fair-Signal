"""Service opportunity and physical-transition checks for the hybrid guard."""

import runpy
from pathlib import Path

import numpy as np

from fair_signal.config import EnvConfig
from fair_signal.demand import PHASE_LANES
from fair_signal.signals import Signal

guard_action = runpy.run_path(str(Path(__file__).parents[1] / "scripts/service_guard.py"))[
    "guard_action"
]


def test_guard_does_not_override_clearance_minimum_or_empty_demand():
    signal = Signal(EnvConfig(), phase=1, target=1)
    counts, ages = np.ones(8), np.zeros(8)
    ages[0] = 300
    assert not guard_action(signal, counts, ages, 1).overridden
    signal.elapsed = 10
    assert guard_action(signal, counts, ages, 1).action == 0
    signal.mode = "yellow"
    assert not guard_action(signal, counts, ages, 1).overridden
    signal.mode = "all_red"
    assert not guard_action(signal, counts, ages, 1).overridden
    signal.mode = "green"
    assert not guard_action(signal, np.zeros(8), ages, 1).overridden
    assert guard_action(signal, counts, np.zeros(8), 2).action == 2


def simulate(guarded):
    config = EnvConfig()
    signal = Signal(config, phase=1, target=1)
    counts, ages = np.ones(8), np.zeros(8)
    served_phases = set()
    maximum = 0
    for second in range(2000):
        if second % config.decision_seconds == 0:
            action = guard_action(signal, counts, ages, 1).action if guarded else 1
            signal.request(action)
        signal.before_second()
        served = PHASE_LANES[signal.phase] if signal.mode == "green" else ()
        if served:
            served_phases.add(signal.phase)
        ages = np.array([0 if lane in served else ages[lane] + 1 for lane in range(8)])
        maximum = max(maximum, ages.max())
        signal.after_second()
    return served_phases, maximum


def test_shared_guard_breaks_repeated_two_phase_starvation_loop():
    phases, age = simulate(False)
    assert phases == {1, 2} and age == 2000
    phases, age = simulate(True)
    assert phases == {0, 1, 2, 3}
    # Conservative bound for this persistent-demand fixture, not a trip-wait guarantee.
    assert age <= 180 + 4 * (10 + 3 + 1 + 5 + 2)
