import numpy as np
import pytest

from fair_signal.config import EnvConfig
from fair_signal.residual.environment import corrected_phase
from fair_signal.signals import Signal


def inputs():
    signal = Signal(EnvConfig(), elapsed=10)
    counts, waits = np.zeros(8), np.zeros(8)
    counts[0], counts[4], waits[4] = 1, 5, 120
    return signal, counts, waits


def test_accept_action_preserves_baseline_even_for_urgent_traffic():
    signal, counts, waits = inputs()
    assert corrected_phase(signal, counts, waits, 2, 0, 0, 10, 90) == (2, 0)


def test_extensions_are_bounded_and_require_actual_current_demand():
    signal, counts, waits = inputs()
    assert corrected_phase(signal, counts, waits, 2, 1, 0, 10, 90) == (0, 5)
    assert corrected_phase(signal, counts, waits, 2, 1, 5, 10, 90) == (0, 5)
    assert corrected_phase(signal, counts, waits, 2, 1, 10, 10, 90) == (2, 0)
    counts[0] = 0
    assert corrected_phase(signal, counts, waits, 2, 1, 0, 10, 90) == (2, 0)
    counts[0], signal.elapsed = 1, 56
    assert corrected_phase(signal, counts, waits, 2, 1, 0, 10, 90) == (2, 0)


@pytest.mark.parametrize("mode,elapsed", [("green", 9), ("yellow", 0), ("all_red", 0)])
def test_corrections_cannot_override_minimum_or_committed_clearance(mode, elapsed):
    signal, counts, waits = inputs()
    signal.mode, signal.elapsed = mode, elapsed
    for action in (1, 2):
        assert corrected_phase(signal, counts, waits, 1, action, 0, 10, 90) == (1, 0)


def test_urgent_action_requires_wait_threshold_and_existing_vehicle():
    signal, counts, waits = inputs()
    assert corrected_phase(signal, counts, waits, 0, 2, 0, 10, 90) == (2, 0)
    waits[4] = 89
    assert corrected_phase(signal, counts, waits, 0, 2, 0, 10, 90) == (0, 0)
    counts[4], waits[4] = 0, 120
    assert corrected_phase(signal, counts, waits, 0, 2, 0, 10, 90) == (0, 0)
