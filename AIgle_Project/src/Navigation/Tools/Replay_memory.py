
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


class Replay_memory(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.memory = deque(maxlen=max_size)

    def remember(self, state, action, reward, next_state, done):
        # --> Save experience to memory
        self.memory.append([state, action, reward, next_state, done])
        return

    def sample_memory(self, batch_size):
        replay_memory = np.array(random.sample(self.memory, batch_size))
        arr = np.array(replay_memory)
        states_batch = np.vstack(arr[:, 0])
        actions_batch = arr[:, 1].astype('float32').reshape(-1, 1)
        rewards_batch = arr[:, 2].astype('float32').reshape(-1, 1)
        next_states_batch = np.vstack(arr[:, 3])
        dones_batch = np.vstack(arr[:, 4]).astype(bool)

        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
