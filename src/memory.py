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
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
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
    def __init__(self, runner, buff_size, prob_alpha=0.6):
        self.runner = iter(runner)
        self.prob_alpha = prob_alpha
        self.capacity = buff_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buff_size,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        raise NotImplementedError