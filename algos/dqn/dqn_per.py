#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import gym
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import actions
import agents
import runner
from wrapper import build_env_wrapper
from networks import dqn_cnn_net
from common import hyperparameters, logger
from memory import PrioritizedExperienceReplayBuffer


if __name__ == "__main__":
    # CONFIG
    params = hyperparameters.PARAMS["pong"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable Cuda"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # INIT ENV
    env, observation_space, action_space = build_env_wrapper(
        params["env_name"], env_type=params["env_type"]
    )

    # LOGGING
    writer = SummaryWriter(comment="-" + params["run_name"] + "-PER")

    # NETWORK
    net = dqn_cnn_net.Network(env.observation_space.shape, env.action_space.n).to(
        device
    )
    tgt_net = agents.TargetNetwork(net)

    # AGENT
    selector = actions.EpsilonGreedyActionSelector(epsilon=params["epsilon_start"])
    epsilon_tracker = logger.EpsilonTracker(selector, params)
    agent = agents.DQNAgent(net, selector, device=device)

    # RUNNER
    exp_source = runner.RunnerSourceFirstLast(
        env, agent, gamma=params["gamma"], steps_count=3
    )
    buffer = PrioritizedExperienceReplayBuffer(
        exp_source, buffer_size=params["replay_size"]
    )
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])

    # TRAIN
    PRIO_REPLAY_ALPHA = 0.6
    BETA_START = 0.4
    BETA_FRAMES = 100000

    frame_idx = 0
    beta = BETA_START

    with logger.RewardTracker(writer, params["stop_reward"]) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params["replay_initial"]:
                continue

            # LEARNING
            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(
                params["batch_size"], beta
            )
            loss_v, sample_priorities_v = loss.calc_weighted_loss_dqn(
                batch,
                batch_weights,
                net,
                tgt_net.target_model,
                params["gamma"],
                device=device,
            )

            loss_v.backward()
            optimizer.step()

            buffer.update_priorities(
                batch_indices, sample_priorities_v.data.cpu().numpy()
            )

            if frame_idx % params["target_net_sync"] == 0:
                tgt_net.sync()
