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
import torch.multiprocessing as mp
import config

from networks import actor_critic_mlp
from common import logger, utils
from memory import ExperienceReplayBuffer
from loss import calc_a2c_loss
import collections


def gather_data(envs, net, device, train_queue, params):
    """
    carry out the actions of the agents in parallel while gathering the experience of each agent

    Args:
        envs: list of environments
        net: neural network
        device: cpu or cuda
        train_queue: queue to store the data gathered by all agents
        params: config parameters

    """

    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0], device=device, apply_softmax=True
    )
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=params["gamma"], steps_count=params["step_count"]
    )

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)


def init_procs(envs, params):
    """
    initialising the threads used to run several agents in parallel. 
    Each process will execute several agents which will have their data added to the master train_queue

    Args:
        envs: list of environments to be used in parallel training
        params: config parameters

    Returns:
        train_queue: master queue to hold all experience gathered by each process
        data_proc_list: list to store each gathering process
    """

    train_queue = mp.Queue(maxsize=params["num_procs"])
    data_proc_list = []
    for _ in range(params["num_procs"]):
        data_proc = mp.Process(
            target=gather_data, args=(envs, net, device, train_queue, params)
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    return train_queue, data_proc_list


TotalReward = collections.namedtuple("TotalReward", field_names="reward")


if __name__ == "__main__":
    # CONFIG
    params = config.PARAMS["cartpole"]
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
    writer = SummaryWriter(comment=params["env_name"])

    # NETWORK
    net = actor_critic_mlp.Network(observation_space, action_space).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"], eps=1e-3)

    # initialise processes
    train_queue, data_proc_list = init_procs(envs, params)

    batch = []
    step_idx = 0

    # TRAINING
    try:
        with logger.RewardTracker(
            net, writer, stop_reward=params["stop_reward"], tag="a3c"
        ) as tracker:
            while True:
                train_entry = train_queue.get()
                if isinstance(train_entry, TotalReward):
                    if tracker.reward(train_entry.reward, step_idx):
                        break
                    continue

                step_idx += 1
                batch.append(train_entry)

                if len(batch) < params["batch_size"]:
                    continue

                loss_policy, loss_v = calc_a2c_loss(batch, net, params)
                batch.clear()

                optimizer.zero_grad()

                loss_policy.backward(retain_graph=True)
                loss_v.backward()

                nn_utils.clip_grad_norm_(net.parameters(), params["grad_clip"])
                optimizer.step()

    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
