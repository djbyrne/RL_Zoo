import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, obs_size, act_size, hidden_layer_size=64):
        super(ActorNetwork, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)


class CriticNetwork(nn.Module):
    def __init__(self, obs_size, hidden_layer_size=64):
        super(CriticNetwork, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        )

    def forward(self, x):
        return self.value(x)



# class Network(nn.Module):
#     def __init__(self, obs_size, act_size, hidden_layer_size=128):
#         super(Network, self).__init__()
#
#         self.base = nn.Sequential(
#             nn.Linear(obs_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#         )
#         self.mu = nn.Sequential(
#             nn.Linear(64, act_size),
#             nn.Tanh(),
#         )
#         self.var = nn.Sequential(
#             nn.Linear(64, act_size),
#             nn.Softplus(),
#         )
#         self.value = nn.Linear(64, 1)
#
#     def forward(self, x):
#         base_out = self.base(x)
#         return self.mu(base_out), self.var(base_out), self.value(base_out)
