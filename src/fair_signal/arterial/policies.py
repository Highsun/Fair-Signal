"""Local additive Q ablation: neither action head can inspect the other local state."""

import torch
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn

from fair_signal.arterial.environment import LOCAL_DIM


class LocalQNetwork(QNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_net = nn.Identity()
        layers = self.net_arch or [128, 128]
        self.local_heads = nn.ModuleList()
        for _ in range(2):
            modules = []
            previous = LOCAL_DIM
            for width in layers:
                modules.extend((nn.Linear(previous, width), self.activation_fn()))
                previous = width
            modules.append(nn.Linear(previous, 4))
            self.local_heads.append(nn.Sequential(*modules))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        a = self.local_heads[0](features[:, :LOCAL_DIM])
        b = self.local_heads[1](features[:, LOCAL_DIM:])
        return (a[:, :, None] + b[:, None, :]).reshape(-1, 16)


class LocalDQNPolicy(DQNPolicy):
    def make_q_net(self):
        args = self._update_features_extractor(self.net_args, features_extractor=None)
        return LocalQNetwork(**args).to(self.device)
