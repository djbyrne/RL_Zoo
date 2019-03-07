import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .ops import NoisyLinear, get_conv_out


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class NoisyDQN(nn.Module):
    """
    Conv net with noisy linear layers at the end
    """
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def forward(self, x):
        # reshape input to be normalised
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return[
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = get_conv_out(input_shape)

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

class DistributionalDQN(nn.Module):
    """
    implementation of a DQN conv network with a categorical distribution at the head of the network instead of the
    discrete actions. Also known as C51
    """

    def __init__(self, input_shape, n_actions, n_atoms=51, v_min=-10, v_max=10):
        super(DistributedDQN, self).__init__()

        self.n_atoms = n_atoms
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

        conv_out_size = get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * self.n_atoms)
        )

        self.register_buffer("supports", torch.arange(self.Vmin,self.Vmax + self.delta_z, self.delta_z))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1 , self.n_atoms)

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
