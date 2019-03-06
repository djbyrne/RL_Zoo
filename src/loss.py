import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import utils


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
    Calculate the mean squared error (MSE) of the sampled batch
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

