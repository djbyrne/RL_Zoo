#!/usr/bin/env python3
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join("../../", "src")))
import gym
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import actions
import agents
import runner
from wrapper import build_multi_env
import wrapper
import loss
import ptan
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F

from networks import actor_critic_mlp
from common import hyperparameters, logger, utils
from memory import ExperienceReplayBuffer
from loss import calc_a2c_loss


if __name__ == "__main__":
    # CONFIG
    config = "cartpole"
    params = hyperparameters.PARAMS[config]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable Cuda"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # INIT ENV
    envs, observation_space, action_space = build_multi_env(
        params["env_name"], env_type=params["env_type"], num_envs=params["num_env"]
    )

    # LOGGING
    writer = SummaryWriter(comment=config)

    # NETWORK
    net = actor_critic_mlp.Network(observation_space, action_space).to(device)

    # AGENT
    agent = agents.PolicyGradientAgent(
        lambda x: net(x)[0], apply_softmax=True, device=device
    )

    # RUNNER
    exp_source = runner.RunnerSourceFirstLast(
        envs, agent, gamma=params["gamma"], steps_count=params["step_count"]
    )
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"], eps=1e-3)

    batch = []

    # TRAINING
    with logger.RewardTracker(writer, stop_reward=195) as tracker:
        for step_idx, exp in enumerate(exp_source):
            batch.append(exp)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break

            if len(batch) < params["batch_size"]:
                continue

            loss_policy, loss_v = calc_a2c_loss(batch, net, params)
            batch.clear()

            optimizer.zero_grad()

            loss_policy.backward(retain_graph=True)
            loss_v.backward()

            nn_utils.clip_grad_norm_(net.parameters(), params["grad_clip"])
            optimizer.step()
