#!/usr/bin/env python3
import os
import math
import ptan
import time
import gym
import roboschool
import argparse
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config
import ac_common
import sys

sys.path.append(os.path.abspath(os.path.join("../../", "src")))
import agents
from networks import actor_critic_continuous as network
import loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-c", "--config", default="half-cheetah", help="Environment id, default=half-cheetah")
    args = parser.parse_args()
    device = "cpu"

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    params = config.PARAMS[args.config]

    envs = [gym.make(params["env_name"]) for _ in range(params["num_env"])]
    test_env = gym.make(params["env_name"])

    net_act = network.ActorNetwork(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
    net_crt = network.CriticNetwork(envs[0].observation_space.shape[0]).to(device)

    writer = SummaryWriter(comment="-a2c_" + args.name)
    agent = agents.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params["gamma"], steps_count=params["step_count"])

    opt_act = optim.Adam(net_act.parameters(), lr=params["actor_learning_rate"])
    opt_crt = optim.Adam(net_crt.parameters(), lr=params["critic_learning_rate"])

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % params["test_iterations"] == 0:
                    rewards, steps = ac_common.run_test(net_act, test_env)

                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)

                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net_act.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)

                if len(batch) < params["batch_size"]:
                    continue

                opt_crt.zero_grad()
                opt_act.zero_grad()

                loss_value_v, loss_v = loss.calc_a2c_continuous_loss(batch, net_act, net_crt, params)
                loss_value_v.backward()
                loss_v.backward()

                batch.clear()

                opt_crt.step()
                opt_act.step()
