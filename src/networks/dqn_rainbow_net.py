import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .ops import NoisyLinear, get_conv_out

class Network(nn.Module):
    """
    implementation of a rainbow dqn network combing dueling network heads for value and advantage,
    Noisy layers and distributional representation of the outputs.
    """

    def __init__(self, input_shape, n_actions, n_atoms=51, v_min=-10, v_max=10):
        super(Network, self).__init__()

        self.n_atoms = n_atoms
        self.n_actions = n_actions
        self.Vmin = -10
        self.Vmax = 10
        self.delta_z = (self.Vmax - self.Vmin) / (self.n_atoms - 1)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = get_conv_out(input_shape, self.conv)

        self.fc_val = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, self.n_atoms)
        )

        self.fc_adv = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, self.n_actions * self.n_atoms)
        )

        self.register_buffer("supports", torch.arange(self.Vmin,self.Vmax + self.delta_z, self.delta_z))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, self.n_atoms)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.n_atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    def both(self, x):
        category_output = self(x)
        probabilities = self.apply_softmax(category_output)
        weights = probabilities * self.supports
        values = weights.sum(dim=2)

        return category_output, values

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.n_atoms)).view(t.size())
