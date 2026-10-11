"""Shared inference-time service guard; this is a distinct hybrid controller."""

from __future__ import annotations

from dataclasses import dataclass

from fair_signal.demand import PHASE_LANES


@dataclass(frozen=True)
class GuardDecision:
    action: int
    overridden: bool
    oldest_age: float
    urgent_phase: int | None


def guard_action(
    signal, counts, request_ages, proposed: int, threshold: float = 180
) -> GuardDecision:
    """Request the oldest overdue demanded phase only when switching is permitted.

    This offers a service opportunity, not a bound on trip wait under spillback.
    It is shared by every compared controller and does not alter DQN training.
    """
    ages = [
        max((float(request_ages[lane]) for lane in lanes if counts[lane] > 0), default=0.0)
        for lanes in PHASE_LANES
    ]
    phase = max(range(4), key=lambda index: ages[index])
    oldest = ages[phase]
    urgent = phase if oldest >= threshold else None
    if urgent is not None and signal.mode == "green" and signal.elapsed >= signal.config.min_green:
        return GuardDecision(urgent, urgent != proposed, oldest, urgent)
    return GuardDecision(proposed, False, oldest, urgent)
