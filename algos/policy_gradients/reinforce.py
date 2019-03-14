
#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join('../../', 'src')))
import gym
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import actions
import agents
import runner
from wrapper import build_env_wrapper
import wrapper
import loss
from networks import dqn_cnn_net, dqn_mlp_net
from common import hyperparameters, logger, utils
from memory import ExperienceReplayBuffer


if __name__ == "__main__":
	# CONFIG
	params = hyperparameters.PARAMS['cartpole']
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default=False, action="store_true", help="Enable Cuda")
	args = parser.parse_args()
	device = torch.device("cuda" if args.cuda else "cpu")

	EPISODES_TO_TRAIN = 4

	# INIT ENV
	env, observation_space, action_space = build_env_wrapper(params['env_name'], env_type=params['env_type'])

	# LOGGING
	writer = SummaryWriter(comment="-" + params['run_name'] + "-reinforce")

	# NETWORK
	net = dqn_mlp_net.Network(observation_space, action_space, hidden_layer_size=64).to(device)

	# AGENT
	agent = agents.PolicyGradientAgent(net, preprocessor=utils.float32_preprocessor, apply_softmax=True)

	# RUNNER
	exp_source = runner.RunnerSourceFirstLast(env, agent, gamma=params['gamma'])
	optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

	total_rewards = []
	step_idx = 0
	done_episodes = 0

	batch_episodes = 0
	batch_states, batch_actions, batch_qvals = [], [], []
	cur_rewards = []


	with logger.RewardTracker(writer, params['stop_reward']) as reward_tracker:
		for step_idx, exp in enumerate(exp_source):
			batch_states.append(exp.state)
			batch_actions.append(int(exp.action))
			cur_rewards.append(exp.reward)

			if exp.last_state is None:
				batch_qvals.extend(loss.calc_qvals(cur_rewards))
				cur_rewards.clear()
				batch_episodes += 1

			new_rewards = exp_source.pop_total_rewards()
			if new_rewards:
				done_episodes += 1
				reward = new_rewards[0]
				total_rewards.append(reward)
				if reward_tracker.reward(new_rewards[0], step_idx):
					break

			if batch_episodes < EPISODES_TO_TRAIN:
				continue

			optimizer.zero_grad()
			states_v = torch.FloatTensor(batch_states)
			batch_actions_t = torch.LongTensor(batch_actions)
			batch_qvals_v = torch.FloatTensor(batch_qvals)

			logits_v = net(states_v)
			log_prob_v = F.log_softmax(logits_v, dim=1)
			log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
			loss_v = -log_prob_actions_v.mean()

			loss_v.backward()
			optimizer.step()

			batch_episodes = 0
			batch_states.clear()
			batch_actions.clear()
			batch_qvals.clear()
