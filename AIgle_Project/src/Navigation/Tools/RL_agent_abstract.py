
##################################################################################################################
"""

"""

# Built-in/Generic Imports
from abc import ABC, abstractmethod

# Libs

# Own modules

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class RL_agent_abc(ABC):
    @property
    @abstractmethod
    def observation(self):
        return

    @property
    @abstractmethod
    def action_lst(self):
        return

    @abstractmethod
    def get_qs(self):
        return

    @abstractmethod
    def step(self, action):
        return

    @abstractmethod
    def remember(self, current_state, action, reward, next_state, done):
        return

    @abstractmethod
    def train(self, discount, tau):
        return
