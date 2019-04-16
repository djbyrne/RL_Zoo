#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../../", "src")))
import gym
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from common import hyperparameters
from common import utils
import time

def unpack_batch_ddqn(batch, device="cpu"):
    """Takes in a batch of environment transitions

        Args:
            batch: batch of stored experiences/environment transitions
            device: cpu or cuda

        Returns:
            states, actions, rewards, dones and last state
    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = utils.float32_preprocessor(states).to(device)
    actions_v = utils.float32_preprocessor(actions).to(device)
    rewards_v = utils.float32_preprocessor(rewards).to(device)
    last_states_v = utils.float32_preprocessor(last_states).to(device)
    dones_t = torch.ByteTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


def test_net(net, env, clipping=[0, 1], count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = utils.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, clipping[0], clipping[1])
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count