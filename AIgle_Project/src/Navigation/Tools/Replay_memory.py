
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import random
from collections import deque
import pickle

# Libs
import numpy as np

# Own modules

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Replay_memory(object):
    def __init__(self, max_size, memory_ref=None):

        if memory_ref is not None:
            self.memory = self.load_replay_memory(memory_ref)
            self.memory_size = len(self.memory)

        else:
            self.memory_size = max_size
            self.memory = deque(maxlen=max_size)

    @property
    def length(self):
        return len(self.memory)

    def remember(self, state, action, reward, next_state, done):
        # --> Save experience to memory
        self.memory.append([state, action, reward, next_state, done])
        return

    def sample_memory(self, batch_size):
        minibatch = np.array(random.sample(self.memory, batch_size))
        return minibatch, None

    def save_replay_memory(self, ref):
        # --> Record replay memory
        with open('Data/ddpg/simple_replay_memory/RM_' + str(ref), 'wb') as file:
            pickle.dump({'memory': self.memory}, file)
        return

    def load_replay_memory(self, memory_ref):
        # TODO: Add load pickle
        return
