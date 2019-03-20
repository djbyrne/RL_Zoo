import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .ops import NoisyLinear, get_conv_out


class Network(nn.Module):
    """
    Conv net with noisy linear layers at the end
    """

    def __init__(self, input_shape, n_actions):
        super(Network, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = get_conv_out(input_shape, self.conv)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions),
        ]
        self.fc = nn.Sequential(self.noisy_layers[0], nn.ReLU(), self.noisy_layers[1])

    def forward(self, x):
        # reshape input to be normalised
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return [
            (
                (layer.weight ** 2).mean().sqrt()
                / (layer.sigma_weight ** 2).mean().sqrt()
            ).item()
            for layer in self.noisy_layers
        ]
