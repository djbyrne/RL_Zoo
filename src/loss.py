#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common import utils
from networks import ops


def calc_loss_dqn(batch, net, tgt_net, gamma=0.99, device="cpu", double=True):
    """
    Calculate the mean squared error (MSE) of the sampled batch
    :param batch: sampled experiences
    :param net: main network
    :param tgt_net: tartget network
    :param gamma: discount factor
    :param device: what device to carry out matrix math
    :param double: is this model use double Q learning
    :return: the MSE of the samples in batch
    """

    # unpack batch of experience
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    if double:
        # calculate the next action to take using main network
        next_state_actions = net(next_states_v).max(1)[1]
        # calculate the values of this action using the target network
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    # discounted reward
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_weighted_loss_dqn(batch, batch_weights, net, tgt_net, gamma=0.99, device="cpu", double=True):
    """
    Calculate the mean squared error (MSE) of the sampled batch for weighted experiences such as when
    using Prioritized Experience Replay (PER)

    :param batch: sampled experiences
    :param batch_weights: the priority
    :param net: main network
    :param tgt_net: tartget network
    :param gamma: discount factor
    :param device: what device to carry out matrix math
    :param double: is this model use double Q learning

    :return: mse: the MSE of the samples in batch
    :return: losses_v: the individual losses of the sample batch with a small constant added
    """

    # unpack batch of experience
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    if double:
        # calculate the next action to take using main network
        next_state_actions = net(next_states_v).max(1)[1]
        # calculate the values of this action using the target network
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    # discounted reward
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v

    # explicitly calculate MSE. allows us to maintain the individual sample loss
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), losses_v + 1e-5


def calc_loss_distributional(batch, net, tgt_net, gamma=0.99, device="cpu", double=True, v_min=-10, v_max=10, n_atoms=51):
    # unpack batch of experience
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v).data.cpu().numpy()

    next_best_distr = next_distr[range(batch_size), next_actions]
    dones = dones.astype(np.bool)

    # project distribution
    projected_distribution = ops.distributional_projection(next_best_distr, rewards, dones, v_min, v_max, n_atoms, gamma)

    # calculate network output
    distr_v = net(states_v)
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_activation_v = F.log_softmax(state_action_values, dim=1)
    projected_distribution_v = torch.tensor(projected_distribution).to(device)

    loss_v = -state_activation_v * projected_distribution_v

    return loss_v.sum(dim=1).mean()

def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()

    dones = dones.astype(np.bool)
    projected_distribution = ops.distributional_projection(next_best_distr, rewards, dones, v_min, v_max, n_atoms, gamma)

    # calculate net output
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = batch_weights_v * loss_v.sum(dim=1)
    return loss_v.mean(), loss_v + 1e-5



