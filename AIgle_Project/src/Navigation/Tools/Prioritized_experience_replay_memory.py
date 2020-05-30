
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
from AIgle_Project.src.Tools.Python_tools import weighted_choice

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Prioritized_experience_replay_memory(object):
    def __init__(self, max_size, memory_ref=None):

        if memory_ref is not None:
            self.memory, self.priorities, self.indexes = self.load(memory_ref)
            self.memory_size = self.length

        else:
            self.memory_size = max_size
            self.memory = deque(maxlen=max_size)
            self.priorities = deque(maxlen=max_size)
            self.indexes = deque(maxlen=max_size)

    @property
    def length(self):
        return len(self.memory)

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

    def sample(self, batch_size):
        assert(len(self.memory) == len(self.priorities))
        # --> Sample memory according to priorities
        # indices = weighted_choice(list(self.indexes), list(self.priorities), batch_size=batch_size)
        indices = random.choices(self.indexes, weights=self.priorities, k=batch_size)
        minibatch = [self.memory[indx - 1] for indx in indices]

        return minibatch, indices

    def save(self, ref):
        # --> Record replay memory
        with open('Data/ddpg/PR_replay_memory/RM_' + ref, 'wb') as file:
            pickle.dump({'memory': self.memory}, file)
            pickle.dump({'priorities': self.priorities}, file)
            pickle.dump({'indexes': self.indexes}, file)
        return

    def load(self, memory_ref):
        # TODO: Add load pickle
        return
