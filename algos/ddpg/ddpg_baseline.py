#!/usr/bin/env python3
import os
import time
import gym

import argparse
from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import ddpg_common
import sys, os
sys.path.append(os.path.abspath(os.path.join("../../", "src")))
import runner
from common import utils
from common import logger
import ptan
from networks import ddpg_mlp
import agents
import memory


ENV_ID = "Pendulum-v0"
GAMMA = 0.99
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
REPLAY_SIZE = 1000000
REPLAY_INITIAL = 10000
EXPLORE = 100000

TEST_ITERS = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cpu")

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    act_net = ddpg_mlp.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = ddpg_mlp.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = agents.AgentDDPG(act_net, device=device)
    exp_source = runner.RunnerSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1
    )
    buffer = memory.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=0.00001)
    crt_opt = optim.Adam(crt_net.parameters(), lr=0.0001)

    frame_idx = 0
    best_reward = None

    with logger.RewardTracker(act_net,writer, 200) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = ddpg_common.unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)


