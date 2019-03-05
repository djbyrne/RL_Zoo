import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class NoisyLinear(nn.Linear):
    """
    Layer that uses Gaussian noise for the purpose of exploration. Random values drawn from a
    normal distribution are added to each weight. The parameters to decide these noisey weights are stored
    and trained during back propagation.
    """
    def __init__(self,in_features,out_features, sigma_init=0.017,bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        # Make sigma trainable
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # creates a tensor that wont be update during back propagation
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialises the layer
        """
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, layer_input):
        """
        Sample random noise in both weights and bias buffers, performs linear transform of the input data.

        :param input: input from previous layer
        :return: linear activation of input with noise (weights and bias)
        """
        self.epsilon_weight.normal_()
        bias = self.sigma_bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        return F.linear(layer_input, self.weight + self.sigma_weight.data, bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, layer_input):
        self.epsison_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(layer_input, self.weight + self.sigma_weight * noise_v, bias)


