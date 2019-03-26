import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Network(nn.Module):
    def __init__(self, obs_size, act_size, hidden_layer_size=128):
        super(Network, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(300, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(300, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(300, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)
