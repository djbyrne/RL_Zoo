import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .ops import NoisyLinear, get_conv_out


class Network(nn.Module):
    """
    Standard MLP for the dqn network
    """
    def __init__(self, input_shape, n_actions):
        """init method generate sequential model

        Args:
            input_shape(numpy array): the shape of the environments state space
            n_actions(int): the number of outputs the network must return

        Returns:
            tensor of (batch_size, n_actions)
        """
        super(Network, self).__init__()

        self.fc1 = nn.Linear(input_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)


    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


