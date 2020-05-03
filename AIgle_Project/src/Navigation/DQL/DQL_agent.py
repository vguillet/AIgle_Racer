
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys

# Libs
import airsim
import cv2
import numpy as np

# Own modules
from AIgle_Project.src.Tools.Agent import Agent
from AIgle_Project.src.Navigation.Tools.RL_agent_abstract import RL_agent_abc

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class DQL_agent(RL_agent_abc, Agent):
    def __init__(self, client, name, model):
        super().__init__(client, name)

        # --> Setup rl settings
        self.settings.rl_behavior_settings.gen_simple_ql_settings()

        # --> Setup model
        self.model = model

        # --> Setup reward dict
        # TODO: Setup reward function
        # self.reward_function = gen_reward_function(reward_dict)

        # --> Setup trackers
        # Step trackers
        self.pos_history = []

        self.reward_history = []
        self.action_history = []
        self.action_success_history = []

        # Episode trackers
        self.reward_timeline = []
        self.aggr_ep_reward_timeline = {'ep': [], 'avg': [], 'max': [], 'min': []}

        return

    def step(self):
        return

    def get_action_lst(self):
        return

    def get_observation(self):
        return

    def __str__(self):
        return self.name + " (Bot)"

    def __repr__(self):
        self.__repr__()
