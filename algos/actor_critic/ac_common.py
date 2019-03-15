#!/usr/bin/env python3
import gym
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common

def unpack_batch(batch, net, device='cpu'):
	 """
    Convert batch into training tensors

    Args:
    	batch: batch of stored experiences
    	net: neural network

    Returns:
    	states variable, actions tensor, reference values variable
    """

    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx, exp in enumerate(batch):
    	states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)

        if exp.last_state is not None:
        	not_done_idx.append(idx)
        	last_states.append(np.array(exp.last_state, copy=False))

       states_v = torch.FloatTensor(states).to(device)
       actions_t = torch.LongTensor(actions).to(device)

       rewards_np = np.array(rewars, dtype=np.float32)

       if not_done_idx:
       		last_states_v = torch.FloatTensor(last_states).to(device)
       		last_vals_v = net(last_states_v)[1]
       		last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
       		rewards_np[not_done_idx] += params['gamma'] ** params['step_count'] * last_vals_np

   		ref_vals_v = torch.FloatTensor(rewards_np).to(device)
   		return states_v, actions_t, ref_vals_v