
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
from pg_common import calculate_entropy, calculate_kl_divergence
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
	GAMMA = 0.99
	LEARNING_RATE = 0.01
	ENTROPY_BETA = 0.01
	BATCH_SIZE = 64

	REWARD_STEPS = 10

	# INIT ENV
	env, observation_space, action_space = build_env_wrapper(params['env_name'], env_type=params['env_type'])

	# LOGGING
	writer = SummaryWriter(comment="-" + params['run_name'] + "-vpg")

	# NETWORK
	net = dqn_mlp_net.Network(observation_space, action_space, hidden_layer_size=32).to(device)

	# AGENT
	agent = agents.PolicyGradientAgent(net, preprocessor=utils.float32_preprocessor, apply_softmax=True)

	# RUNNER
	exp_source = runner.RunnerSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=params['step_count'])
	optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

	total_rewards = []
	step_idx = 0
	done_episodes = 0

	batch_episodes = 0
	batch_states, batch_actions, batch_scales = [], [], []
	cur_rewards = []
	reward_sum = 0


	with logger.RewardTracker(writer, params['stop_reward']) as reward_tracker:
		for step_idx, exp in enumerate(exp_source):
			reward_sum += exp.reward
			baseline = reward_sum / (step_idx + 1)
			writer.add_scalar("baseline", baseline, step_idx)

			batch_states.append(exp.state)
			batch_actions.append(int(exp.action))
			batch_scales.append(exp.reward - baseline)

			new_rewards = exp_source.pop_total_rewards()
			if new_rewards:
				done_episodes += 1
				reward = new_rewards[0]
				total_rewards.append(reward)
				if reward_tracker.reward(new_rewards[0], step_idx):
					break

			if len(batch_states) < params['batch_size']:
				continue

			states_v = torch.FloatTensor(batch_states)
			batch_actions_t = torch.LongTensor(batch_actions)
			batch_scale_v = torch.FloatTensor(batch_scales)

			# calculate loss
			optimizer.zero_grad()
			logits_v = net(states_v)
			log_prob_v = F.log_softmax(logits_v, dim=1)
			log_prob_actions_v = batch_scale_v * log_prob_v[range(params['batch_size']), batch_actions_t]
			loss_policy_v = -log_prob_actions_v.mean()

			# # calculate entropy
			prob_v = F.softmax(logits_v, dim=1)
			entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
			entropy_loss_v = -ENTROPY_BETA * entropy_v
			loss_v = loss_policy_v + entropy_loss_v
			# entropy_loss, prob_v = calculate_entropy(logits_v, log_prob_v)
			# loss_v = loss_policy_v + entropy_loss

			loss_v.backward()
			optimizer.step()

			# calc KL-div
			new_logits_v = net(states_v)
			new_prob_v = F.softmax(new_logits_v, dim=1)
			kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
			# kl_div_v = calculate_kl_divergence(net(states_v), prob_v)
			# writer.add_scalar("kl", kl_div_v.item(), step_idx)

			# calculate the stats
			grad_max = 0.0
			grad_means = 0.0
			grad_count = 0
			for p in net.parameters():
				grad_max = max(grad_max, p.grad.abs().max().item())
				grad_means += (p.grad ** 2).mean().sqrt().item()
				grad_count += 1

			batch_states.clear()
			batch_actions.clear()
			batch_scales.clear()
