"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import actions
from src.common import utils


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def defualt_states_preprocessor(states):
    """
        Convert list of states into the form suitable for model. By default we assume Variable
        :param states: list of numpy arrays with states
        :return: Variable
    """

    if len(states) == 1:
        np_states = np.expand_dims(states[0],0)    #add extra colum for batch size
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)

    return torch.tensor(np_states)

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=defualt_states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] *len(states)

        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        q_value = self.dqn_model(states)
        q = q_value.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states

    def calc_loss(self,batch, net, tgt_net, gamma=0.99, device="cpu"):

        #unpack batch of experience
        states, actions, rewards, dones, next_states = utils.unpack_batch(batch)

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

        #discounted reward
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)


class TargetNetwork:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.states_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v

        self.target_model.load_state_dict(tgt_state)