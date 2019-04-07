import numpy as np
import torch


class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """

    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    """
    Selects action based on the epsilon greedy policy
    """

    def __init__(self, epsilon=0.5, selector=ArgmaxActionSelector()):
        self.epsilon = epsilon
        self.selector = selector

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """

    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class VarianceSampleSelector(ActionSelector):
    """
    Samples action from the network output and network variance.
    """

    def __call__(self, probs, variance, lower_bound=-1, upper_bound=1):
        """
        Args:
            probs: probabilities from the network
            variance: variance from the network

        Returns:
            action sampled between probs and variance, clipped between 
        """

        sigma = torch.sqrt(variance).data.cpu().numpy()
        actions = np.random.normal(probs, sigma)
        actions = np.clip(actions, lower_bound, upper_bound)

        return actions
