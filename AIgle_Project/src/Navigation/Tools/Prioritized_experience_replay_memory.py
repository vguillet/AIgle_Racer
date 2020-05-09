
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import sys
import random
import time
from collections import deque

# Libs
import numpy as np

# Own modules

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Prioritized_experience_replay_memory(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.memory = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.indexes = deque(maxlen=max_size)

    def remember(self, state, action, reward, next_state, done):
        # --> Save experience to memory
        self.memory.append([state, action, reward, next_state, done])
        self.priorities.append(1)

        ln = len(self.memory)
        if ln < self.memory_size:
            self.indexes.append(ln)
        return

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[index-1] = priority + 1
        return

    def sample_memory(self, batch_size):
        indices = random.choices(self.indexes, weights=self.priorities, k=batch_size)

        replay_memory = [self.memory[indx - 1] for indx in indices]
        arr = np.array(replay_memory)
        states_batch = np.vstack(arr[:, 0])
        actions_batch = arr[:, 1].astype('float32').reshape(-1, 1)
        rewards_batch = arr[:, 2].astype('float32').reshape(-1, 1)
        next_states_batch = np.vstack(arr[:, 3])
        done_batch = np.vstack(arr[:, 4]).astype(bool)
        return states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, indices
