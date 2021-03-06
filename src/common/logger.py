import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque


def save_model(net, save_name="latest_model.pth"):
    torch.save(net.state_dict(), save_name)


class RewardTracker:
    def __init__(self, net, writer, stop_reward, tag="experiment"):
        self.writer = writer
        self.stop_reward = stop_reward
        self.best_avg_reward = -np.inf
        self.tag = tag
        self.net = net

        self.save_path = os.path.join("saves", tag)
        os.makedirs(self.save_path, exist_ok=True)

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        """
        add reward to tracker and check if early stopping should be activated
        """

        self.total_rewards.append(reward)
        time_difference = (time.time() - self.ts)
        frame_difference = (frame - self.ts_frame)
        speed = frame_difference if time_difference == 0 else frame_difference / time_difference
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-10:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print(
            "%d: done %d games, mean reward %.3f, speed %.2f f/s%s"
            % (frame, len(self.total_rewards), mean_reward, speed, epsilon_str)
        )
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)

        if mean_reward > self.best_avg_reward:
            self.best_avg_reward = mean_reward
            name = self.tag + "_best_avg_reward.dat"
            save_name = os.path.join(self.save_path, name)
            save_model(self.net, save_name)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params["epsilon_start"]
        self.epsilon_final = params["epsilon_final"]
        self.epsilon_frames = params["epsilon_frames"]
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = max(
            self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames
        )
