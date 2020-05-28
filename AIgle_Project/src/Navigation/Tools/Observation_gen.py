
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import sys
import random
import time

# Libs
import numpy as np
from collections import deque
from itertools import combinations, permutations, product

# Own modules
from AIgle_Project.src.Navigation.Tools.RL_agent_abstract import RL_agent_abc
from AIgle_Project.src.Tools.Agent import Agent

from AIgle_Project.src.State_estimation.Camera import Camera
from AIgle_Project.src.Navigation.Models.Simple_image_models import Simple_image_model

from AIgle_Project.src.Navigation.Tools.Replay_memory import Replay_memory
from AIgle_Project.src.Navigation.Tools.Prioritized_experience_replay_memory import Prioritized_experience_replay_memory

from AIgle_Project.src.Navigation.Tools.Tensor_board_gen import ModifiedTensorBoard
from AIgle_Project.src.Navigation.Tools.Reward_function_gen import Reward_function

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Observation:
    def __init__(self, observation_type: "[Vector, Image]"):
        if observation_type == "Vector":
            self.__call__ = self.vector_observation

        else:
            self.__call__ = self.image_observation

    def __call__(self, *args, **kwargs):
        return

    def vector_observation(self, goal_dict, goal_tracker, state):
        # --> Determine vector to next goal
        x = goal_dict[str(goal_tracker)]["x"] - state.kinematics_estimated.position.x_val
        y = goal_dict[str(goal_tracker)]["y"] - state.kinematics_estimated.position.y_val
        z = goal_dict[str(goal_tracker)]["z"] - state.kinematics_estimated.position.z_val

        # --> Determine velocity vector magnitude ot next goal
        linear_velocity_magnitude = (abs(state.kinematics_estimated.linear_velocity.x_val)
                                     + abs(state.kinematics_estimated.linear_velocity.y_val)
                                     + abs(state.kinematics_estimated.linear_velocity.z_val)) / 3

        return [x, y, z, linear_velocity_magnitude]

    def image_observation(self):
        return
