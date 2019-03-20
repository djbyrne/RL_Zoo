
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join('../../', 'src')))
import gym
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import actions
import agents
import runner
from wrapper import build_env_wrapper
import wrapper
import loss
from networks import dqn_cnn_net, dqn_mlp_net
from common import hyperparameters, logger
from memory import ExperienceReplayBuffer



if __name__ == "__main__":
	# CONFIG
	params = hyperparameters.PARAMS['cartpole']
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default=False, action="store_true", help="Enable Cuda")
	args = parser.parse_args()
	device = torch.device("cuda" if args.cuda else "cpu")

	# INIT ENV
	env, observation_space, action_space = build_env_wrapper(params['env_name'], env_type=params['env_type'])

	# LOGGING
	writer = SummaryWriter(comment="-" + params['run_name'] + "-basic")

	# NETWORK
	net = dqn_mlp_net.Network(observation_space, action_space, hidden_layer_size=64).to(device)
	tgt_net = agents.TargetNetwork(net)

	# AGENT
	selector = actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
	epsilon_tracker = logger.EpsilonTracker(selector, params)
	agent = agents.DQNAgent(net, selector, device=device)

	# RUNNER
	exp_source = runner.RunnerSourceFirstLast(env, agent, gamma=params['gamma'],steps_count=1)
	buffer = ExperienceReplayBuffer(exp_source,buffer_size=params['replay_size'])
	optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

	frame_idx = 0
	done = False

	# TRAIN
	with logger.RewardTracker(writer, params['stop_reward']) as reward_tracker:
		while True:
			frame_idx += 1
			buffer.populate(1)
			epsilon_tracker.frame(frame_idx)

			new_rewards = exp_source.pop_total_rewards()
			if new_rewards:
				if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
					print("finished")
					break

			if len(buffer) < params['replay_initial']:
				continue

			# learning step
			optimizer.zero_grad()
			batch = buffer.sample(1)
			loss_v = loss.calc_loss_dqn(batch, net, tgt_net.target_model,params['gamma'],device)
			loss_v.backward()
			optimizer.step()

			if frame_idx % params['target_net_sync'] == 0:
				tgt_net.sync()
