"""Explicit algorithm variants and a dense, lifetime-wait fairness objective."""

from __future__ import annotations

from collections.abc import Iterable


def quadratic_wait_increment(waits: Iterable[float], threshold: float, queue_scale: float) -> float:
    """Exact increase of sum(w_i**2)/(2*threshold*queue_scale) this second.

    waits contains CURRENT cumulative stop times of vehicles stopped this second,
    after increasing their clocks by one. Moving and arrived vehicles contribute
    zero; arrival never produces a positive potential-drop reward.
    """
    return sum(2 * wait - 1 for wait in waits) / (2 * threshold * queue_scale)


def policy_options(method: str, parameters: dict) -> dict:
    return {
        "enhanced": method != "dqn_queue",
        "fair_weight": parameters.get("fair_weight", 0.0),
        "reward_mode": "vehicle_quadratic"
        if method in ("dqn_dense", "dqn_dense_state")
        else "head_threshold",
        "observation_mode": "service"
        if method in ("dqn_dense_state", "dqn_fair_state")
        else "legacy",
    }
