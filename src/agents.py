"""
Agent is something which converts states into actions and has state
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import actions
from common import utils



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


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    def __init__(
            self,
            dqn_model,
            action_selector,
            device="cpu",
            preprocessor=utils.default_states_preprocessor,
    ):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)

        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        q_value = self.dqn_model(states)
        q = q_value.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states


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


class PolicyGradientAgent(BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """

    def __init__(
            self,
            model,
            action_selector=actions.ProbabilityActionSelector(),
            device="cpu",
            apply_softmax=False,
            preprocessor=utils.default_states_preprocessor,
    ):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        """
        Return actions from a given list of states

        Args:
            states: batch of states
            agent_states: 

        Returns:
            list of actions
        """

        if agent_states is None:
            agent_states = [None] * len(states)

        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)

        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states


class ContinuousAgent(BaseAgent):
    def __init__(
            self,
            model,
            action_selector=actions.VarianceSampleSelector(),
            device="cpu",
            apply_softmax=False,
            preprocessor=utils.float32_preprocessor,
    ):

        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        """
        Return continuous action from a given list of states

        Args:
            states: batch of states
            agent_states: 

        Returns:
            list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)

        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        probs_v, var_v, _ = self.model(states)
        probs = probs_v.data.cpu().numpy()

        actions = self.action_selector(probs, var_v)

        return actions, agent_states


class AgentA2C(BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = utils.float32_preprocessor(states).to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class AgentDDPG(BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """

    def __init__(self, net, device="cpu", clipping=[0, 1], ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon
        self.clipping = clipping

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = utils.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)

                # apply OU noise
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)
                action += self.ou_epsilon * a_state

                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        # actions = np.clip(actions, self.clipping[0], self.clipping[1])
        return actions, new_a_states
