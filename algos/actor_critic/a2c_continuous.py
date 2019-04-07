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


class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)

HID_SIZE = 64

class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


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

    net_act = ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
    net_crt = ModelCritic(envs[0].observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

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
                    ts = time.time()
                    rewards, steps = ac_common.test_net(net_act, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
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

                states_v, actions_v, vals_ref_v = \
                    ac_common.unpack_batch_a2c(batch, net_crt,
                                               last_val_gamma=params["gamma"] ** params["step_count"], device=device)
                batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                opt_act.zero_grad()
                mu_v = net_act(states_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * ac_common.calc_logprob(mu_v, net_act.logstd, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = params["beta"] * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)

