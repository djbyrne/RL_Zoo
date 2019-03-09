import random
import collections
import numpy as np
from collections import namedtuple

from runner import Runner

# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class BaseBuffer:
    def __init__(self,runner, buffer_size):
        return None

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def populate(self):
        raise NotImplementedError

    def _add(self):
        raise NotImplementedError


class ExperienceReplayBuffer:
    def __init__(self, runner, buffer_size):
        assert isinstance(runner, (Runner, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if runner is None else iter(runner)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get a batch of randomly selected from experiences from the buffer
        TODO: implement sampling order policy
        :param batch_size: the amount of experiences to sample from the buffer
        :return: list of sampled experiences
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        """
        add an experience to the buffer
        :param sample: experience to be added
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)


class PrioritizedExperienceReplayBuffer:
    """
    Experience replay buffer that stores experiences based on a given priority according
    to the training loss
    """

    def __init__(self, runner, buffer_size, prob_alpha=0.6):
        self.runner_iterator = iter(runner)
        self.prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.position = 0
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        """
        Iterates through the runner object and adds experiences to the buffer

        :param count: the number of samples to pull from the runner
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.runner_iterator)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.position] = sample
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Get a batch of randomly selected from experiences from the buffer

        :param batch_size: the amount of experiences to retrieve
        :param beta: determines how much bias we give to the sampling priority

        :return: sample: list of sampled experiences
        :return: indices: indices of sampled experiences
        :return: weights: weight of sampled experiences
        """

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # convert priorities to probabilities
        probabilities = priorities ** self.prob_alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size,p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """
        Update the new priorities of the processed batch
        :param batch_indices: the indices of the sampled experiences of the batch
        :param batch_priorities: the new priorities of the sampled batch
        """

        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority