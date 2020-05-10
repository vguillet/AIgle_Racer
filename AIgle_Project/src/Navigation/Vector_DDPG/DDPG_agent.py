
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
from AIgle_Project.src.Tools.Agent import Agent
from AIgle_Project.src.Navigation.Tools.RL_agent_abstract import RL_agent_abc
from AIgle_Project.src.Navigation.Tools.Tensor_board_gen import ModifiedTensorBoard

from AIgle_Project.src.Navigation.Vector_DDPG.Actor_model import Actor_model
from AIgle_Project.src.Navigation.Vector_DDPG.Critic_model import Critic_model

from AIgle_Project.src.Navigation.Tools.Replay_memory import Replay_memory
from AIgle_Project.src.Navigation.Tools.Prioritized_experience_replay_memory import Prioritized_experience_replay_memory

from AIgle_Project.src.Navigation.Tools.Reward_function_gen import Reward_function
from AIgle_Project.src.Navigation.Vector_DDPG.OUAction_noise import OUAction_noise

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class DDPG_agent(RL_agent_abc, Agent):
    def __init__(self, client, name, memory_type="simple",
                 memory_ref=None,
                 actor_ref=None,
                 critic_ref=None):

        super().__init__(client, name)

        # --> Setup rl settings
        self.settings.rl_behavior_settings.gen_ddql_settings()

        # --> Setup tools
        self.noise = OUAction_noise(mu=np.zeros(len(self.action_lst)))

        # --> Setup model
        self.actor_model = Actor_model("Actor",
                                       self.settings.rl_behavior_settings.actor_learning_rate,
                                       len(self.observation),
                                       len(self.action_lst),
                                       self.settings.agent_settings.max_step)

        self.critic_model = Critic_model("Critic",
                                         self.settings.rl_behavior_settings.critic_learning_rate,
                                         self.observation.shape,
                                         len(self.action_lst))

        # --> Setup rewards
        self.reward_function = Reward_function()
        self.goal_tracker = 0

        # --> Setup trackers
        if memory_type == "simple":
            self.memory = Replay_memory(self.settings.rl_behavior_settings.memory_size)

        elif memory_type == "prioritized":
            self.memory = Prioritized_experience_replay_memory(self.settings.rl_behavior_settings.memory_size)

        else:
            raise("!!!!! Invalid memory setting !!!!!")

        # Step trackers
        self.action_history = []

        # self.reward_history = []
        # self.action_history = []
        # self.action_success_history = []

        # Episode trackers
        # self.reward_timeline = []
        # self.aggr_ep_reward_timeline = {'ep': [], 'avg': [], 'max': [], 'min': []}

        return

    @property
    def observation(self):
        # --> Determine vector to next goal
        x = self.reward_function.goal_dict[str(self.goal_tracker)]["x"] - self.state.kinematics_estimated.position.x
        y = self.reward_function.goal_dict[str(self.goal_tracker)]["y"] - self.state.kinematics_estimated.position.y
        z = self.reward_function.goal_dict[str(self.goal_tracker)]["z"] - self.state.kinematics_estimated.position.z

        # --> Determine velocity vector magnitude ot next goal
        linear_velocity_magnitude = (abs(self.state.kinematics_estimated.linear_velocity.x_val)
                                     + abs(self.state.kinematics_estimated.linear_velocity.y_val)
                                     + abs(self.state.kinematics_estimated.linear_velocity.z_val)) / 3

        return [x, y, z, linear_velocity_magnitude]

    @property
    def action_lst(self):
        possible_moves = []
        possible_speeds = []

        action_lst = []

        # --> List all possible positions combinations
        for dimension in range(3):
            for i in range(self.settings.agent_settings.agent_min_move,
                           self.settings.agent_settings.agent_max_move + 1):
                possible_moves.append(i)
                possible_moves.append(-i)

        possible_moves = set(combinations(possible_moves, 3))

        # --> List all possible speeds
        for speed in range(self.settings.agent_settings.agent_min_speed,
                           self.settings.agent_settings.agent_max_speed + 1):
            possible_speeds.append(speed)

        # --> List all possible positions and speed combinations
        action_lst = list(product(possible_moves, possible_speeds))

        return action_lst

    def get_qs(self):
        # --> Queries actor main network for Q values given current observation
        action = self.actor_model.main_network.predict(self.observation).ravel()[0]
        self.action_history.append(action)

        return action

    def step(self, action):
        # --> Determine action requested
        action = self.action_lst[action]

        # --> Determine target new state
        current_state = self.observation
        next_state = [[round(current_state[0][0] + action[0][0], 1),
                      round(current_state[0][1] + action[0][1], 1),
                      round(current_state[0][2] + action[0][2], 1)],
                      action[1]]

        # --> Limiting top and low
        # TODO: Improve limits
        if next_state[0][2] < -6:
            next_state[0][2] = -6
        elif next_state[0][2] >= 3.5:
            next_state[0][2] = 3.5

        # --> Move to target
        self.move(next_state)

        # --> Evaluate collision
        collision = self.collision

        # --> Determine reward based on resulting state
        reward = self.reward_function.get_reward(self.observation, self.goal_tracker, collision, self.age)

        # --> Determine whether done or not
        done = self.reward_function.check_if_done(self.observation, self.goal_tracker, collision, self.age)

        if not done:
            self.age += 1

        # --> Record step results


        return self.observation, reward, done

    def remember(self, current_state, action, reward, next_state, done):
        self.memory.remember(current_state, action, reward, next_state, done)
        return

    def train(self, terminal_state, target_update_counter):


        return target_update_counter

    def __str__(self):
        return self.name + " (Bot)"

    def __repr__(self):
        self.__repr__()
