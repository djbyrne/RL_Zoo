import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Network(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Network, self).__init__()

        self.body = nn.Sequential(nn.Linear(input_shape[0], 64), nn.ReLU())

        # conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

        self.value = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        fx = x.float()
        out = self.body(fx)
        return self.policy(out), self.value(out)
