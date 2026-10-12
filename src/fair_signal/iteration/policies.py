"""Matched additive correction heads; local masks prevent cross-node leakage."""

import torch
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn


class MatchedQNetwork(QNetwork):
    def __init__(self, *args, local_information=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_net = nn.Identity()
        masks = torch.ones(2, 162)
        if local_information:
            masks.zero_()
            masks[0, :76] = masks[0, 152:157] = 1
            masks[1, 76:152] = masks[1, 157:162] = 1
        self.register_buffer("information_masks", masks)
        self.heads = nn.ModuleList()
        for _ in range(2):
            modules = []
            width = 162
            for hidden in self.net_arch:
                modules.extend((nn.Linear(width, hidden), self.activation_fn()))
                width = hidden
            modules.append(nn.Linear(width, 3))
            self.heads.append(nn.Sequential(*modules))

    def node_values(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        return [
            head(features * mask)
            for head, mask in zip(self.heads, self.information_masks, strict=True)
        ]

    def forward(self, obs):
        a, b = self.node_values(obs)
        return (a[:, :, None] + b[:, None, :]).reshape(-1, 9)


class MatchedDQNPolicy(DQNPolicy):
    def __init__(self, *args, local_information=False, **kwargs):
        self.local_information = local_information
        super().__init__(*args, **kwargs)

    def make_q_net(self):
        args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MatchedQNetwork(**args, local_information=self.local_information).to(self.device)
