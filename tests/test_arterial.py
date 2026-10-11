import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from gymnasium import spaces

from fair_signal.arterial.demand import (
    ALL_LANES,
    INCOMING,
    JUNCTIONS,
    NEIGHBORS,
    ORIGINS,
    generate_routes,
    route_edges,
)
from fair_signal.arterial.environment import LOCAL_DIM, decode_action, encode_action, service_action
from fair_signal.arterial.policies import LocalDQNPolicy
from fair_signal.arterial.validation import audit_signal_rows
from fair_signal.config import EnvConfig
from fair_signal.demand import DIRECTIONS, EXITS
from fair_signal.signals import Signal


def test_routes_really_cross_junctions_and_are_seeded(tmp_path):
    paths = [
        generate_routes(tmp_path / str(i), "tidal", "validation", 2001, 1800) for i in range(2)
    ]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert set(INCOMING["a"]).isdisjoint(INCOMING["b"])
    assert len(ALL_LANES) == 28
    legal = {
        (f"{NEIGHBORS[j][d]}_{j}", f"{j}_{NEIGHBORS[j][dest]}")
        for j in JUNCTIONS
        for d in DIRECTIONS
        for dest in EXITS[d]
    }
    routes = [
        v.find("route").get("edges").split()
        for v in ET.parse(paths[0]).getroot().findall("vehicle")
    ]
    assert sum(len(r) == 3 for r in routes) > len(routes) / 2
    for route in routes:
        assert len(route) in (2, 3)
        assert all(pair in legal for pair in zip(route, route[1:], strict=False))
    for origin in ORIGINS:
        for first in range(3):
            for second in range(3):
                assert len(route_edges(origin, (first, second))) in (2, 3)


def test_joint_actions_and_shared_guard():
    for a in range(16):
        assert encode_action(decode_action(a)) == a
    signal = Signal(EnvConfig())
    counts, ages = np.zeros(8), np.zeros(8)
    counts[4], ages[4] = 1, 190
    assert service_action(signal, counts, ages, 0, 180) == 0
    signal.elapsed = 10
    assert service_action(signal, counts, ages, 0, 180) == 2
    counts[4] = 0
    assert service_action(signal, counts, ages, 0, 180) == 0


def test_local_q_actions_cannot_read_the_neighbor_state():
    policy = LocalDQNPolicy(
        spaces.Box(0, 1, (LOCAL_DIM * 2,), np.float32),
        spaces.Discrete(16),
        lambda _: 0.001,
        net_arch=[16],
    )
    observations = torch.rand(2, LOCAL_DIM * 2)
    observations[1, :LOCAL_DIM] = observations[0, :LOCAL_DIM]
    with torch.no_grad():
        q = policy.q_net(observations).reshape(2, 4, 4)
    # Changing B may add a constant to A's values, never change A's preferences.
    assert torch.allclose(q[0, :, 0] - q[0, 0, 0], q[1, :, 0] - q[1, 0, 0], atol=1e-6)
    assert q[0, :, 0].argmax() == q[1, :, 0].argmax()


def test_audit_rejects_phase_change_without_clearance():
    rows = [
        {
            "a_mode": "green",
            "a_phase": int(t >= 10),
            "a_target": 0,
            "a_elapsed": t % 10,
            "b_mode": "green",
            "b_phase": 0,
            "b_target": 0,
            "b_elapsed": t,
        }
        for t in range(20)
    ]
    with pytest.raises(ValueError, match="without yellow"):
        audit_signal_rows(rows, EnvConfig())
