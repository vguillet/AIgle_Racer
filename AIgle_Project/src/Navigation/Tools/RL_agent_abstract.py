
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
    def step(self, action):
        return

    @abstractmethod
    def __str__(self):
        return

    @abstractmethod
    def __repr__(self):
        self.__repr__()
