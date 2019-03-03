
#!/usr/bin/env python3
import gym
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from src import actions, agents, experiences, common, wrapper
from src.models import dqn_model
from src.common import hyperparameters, logger


# from lib import dqn_model, common

if __name__ == "__main__":
	params = hyperparameters.PARAMS['pong']
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default=False, action="store_true", help="Enable Cuda")
	args = parser.parse_args()
	device = torch.device("cuda" if args.cuda else "cpu")

	env = gym.make(params['env_name'])
	env = wrapper.wrap_dqn(env)

	writer = SummaryWriter(comment="-" + params['run_name'] + "-basic")
	net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

	tgt_net = agents.TargetNetwork(net)
	selector = actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
	epsilon_tracker = logger.EpsilonTracker(selector, params)
	agent = agents.DQNAgent(net, selector, device=device)

	exp_source = experiences.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'],steps_count=1)
	buffer = experiences.ExperienceReplayBuffer(exp_source,buffer_size=params['replay_size'])
	optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

	# frame_idx = 0

