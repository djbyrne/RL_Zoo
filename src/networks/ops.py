import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def get_conv_out(shape, conv):
    o = conv(torch.zeros(1, *shape))
    return int(np.prod(o.size()))


class NoisyLinear(nn.Linear):
    """
    Layer that uses Gaussian noise for the purpose of exploration. Random values drawn from a
    normal distribution are added to each weight. The parameters to decide these noisey weights are stored
    and trained during back propagation.
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        # Make sigma trainable
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )
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
        std = math.sqrt(3 / self.in_features)
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
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias
        )
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )
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


def distributional_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    For each atom the network predicts that the discounted value will fall into this atom's range.
    This function performs the contraction of distribution of the next states best action and projects the results back
    into the original algorithm

    :param next_distr: the probability distribution of the next state given after the step function
    :param rewards: the reward given after the step function
    :param dones: is the state terminal
    :param Vmin: the minimum range of atom values
    :param Vmax: the maximum range of atom values
    :param n_atoms: the number of categories in the distribution
    :param gamma: the discount factor
    :return: the projected distribution
    """

    batch_size = len(rewards)
    projected_distribution = np.zeros((batch_size, n_atoms), dtype=np.float32)
    # width of each atom in our value range
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    for atom in range(n_atoms):
        # calculate where each atom will be projected
        target_atom = np.minimum(
            Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma)
        )
        # calculate the atom numbers the sample has projected
        projected_atom = (target_atom - Vmin) / delta_z

        # handles situation when projected atom lands exactly on target atom
        lower_bound = np.floor(projected_atom).astype(np.int64)
        upper_bound = np.ceil(projected_atom).astype(np.int64)

        # if the upper and lower points of the atom are equal, lands exactly on target
        eq_mask = upper_bound == lower_bound
        projected_distribution[eq_mask, lower_bound[eq_mask]] += next_distr[
            eq_mask, atom
        ]

        # if the upper and lower points of the atom are not equal, lands between 2 atoms
        ne_mask = upper_bound != lower_bound
        projected_distribution[ne_mask, lower_bound[ne_mask]] += (
            next_distr[ne_mask, atom] * (upper_bound - projected_atom)[ne_mask]
        )
        projected_distribution[ne_mask, upper_bound[ne_mask]] += (
            next_distr[ne_mask, atom] * (projected_atom - lower_bound)[ne_mask]
        )

        # handles situation of the next state is terminal, dont take into account next distr, just have prob of 1
        if dones.any():
            projected_distribution[dones] = 0.0
            target_atom = np.minimum(Vmax, rewards[dones])
            projected_atom = (target_atom - Vmin) / delta_z

            lower_bound = np.floor(projected_atom).astype(np.int64)
            upper_bound = np.ceil(projected_atom).astype(np.int64)

            eq_mask = upper_bound == lower_bound
            eq_dones = dones.copy()
            eq_dones[dones] = eq_mask

            if eq_dones.any():
                projected_distribution[eq_dones, 1] = 1.0

            ne_mask = upper_bound != lower_bound
            ne_dones = dones.copy()
            ne_dones[dones] = ne_mask

            if ne_dones.any():
                projected_distribution[ne_dones, lower_bound] = (
                    upper_bound - projected_atom
                )[ne_mask]
                projected_distribution[ne_dones, upper_bound] = (
                    projected_atom - lower_bound
                )[ne_mask]

        return projected_distribution
