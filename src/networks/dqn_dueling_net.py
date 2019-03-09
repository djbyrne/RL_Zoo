import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .ops import NoisyLinear, get_conv_out

class Network(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Network, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = get_conv_out(input_shape, self.conv)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)

        return value + advantage - advantage.mean()