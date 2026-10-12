import ast
import inspect
import textwrap

import gymnasium as gym
import numpy as np
import pytest
import torch

from fair_signal.arterial.demand import generate_routes
from fair_signal.arterial.environment import ArterialEnv
from fair_signal.demand import sha256
from fair_signal.iteration.demand import generate_scaled_routes
from fair_signal.iteration.environment import IterationEnv
from fair_signal.iteration.physics import ScheduledArterialEnv
from fair_signal.iteration.policies import MatchedDQNPolicy
from fair_signal.residual.environment import ResidualEnv


def test_scheduled_physics_only_changes_window_expression():
    old = textwrap.dedent(inspect.getsource(ArterialEnv.step))
    new = textwrap.dedent(inspect.getsource(ScheduledArterialEnv.step))
    old = old.replace(
        "self.config.demand_seconds / 3 <= self.time < 2 * self.config.demand_seconds / 3",
        "self.slowdown_window[0] <= self.time < self.slowdown_window[1]",
    )
    assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))
    assert IterationEnv.__mro__[1:4] == (
        ResidualEnv,
        ScheduledArterialEnv,
        ArterialEnv,
    )


@pytest.mark.parametrize("split", ["validation", "test"])
def test_unscaled_routes_exactly_match_frozen_generator(tmp_path, split):
    old = generate_routes(tmp_path / "old", "tidal", split, 3001, 120)
    new = generate_scaled_routes(tmp_path / "new", "tidal", split, 3001, 120)
    assert sha256(old) == sha256(new)


def test_matched_local_heads_cannot_observe_other_node():
    space = gym.spaces.Box(0, 1, (162,), np.float32)
    policies = [
        MatchedDQNPolicy(
            space,
            gym.spaces.Discrete(9),
            lambda _: 3e-4,
            net_arch=[128, 128],
            local_information=local,
        )
        for local in (False, True)
    ]
    full, local = [p.q_net for p in policies]
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in local.parameters())
    obs = torch.rand(5, 162)
    for node in (0, 1):
        changed = obs.clone()
        foreign = local.information_masks[node] == 0
        changed[:, foreign] = torch.rand_like(changed[:, foreign])
        assert torch.equal(local.node_values(obs)[node], local.node_values(changed)[node])
        assert not torch.equal(full.node_values(obs)[node], full.node_values(changed)[node])
    a, b = local.node_values(obs)
    joint = local(obs).argmax(dim=1)
    assert torch.equal(joint, 3 * a.argmax(dim=1) + b.argmax(dim=1))
