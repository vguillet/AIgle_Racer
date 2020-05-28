
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
from AIgle_Project.src.Navigation.Models.Vector_DDQL_model import Vector_DDQL_model

from AIgle_Project.src.Navigation.Tools.Replay_memory import Replay_memory
from AIgle_Project.src.Navigation.Tools.Prioritized_experience_replay_memory import Prioritized_experience_replay_memory

from AIgle_Project.src.Navigation.Tools.Tensor_board_gen import ModifiedTensorBoard
from AIgle_Project.src.Navigation.Tools.Reward_function_gen import Reward_function

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Vector_DDQL_agent(RL_agent_abc, Agent):
    def __init__(self, client, name, memory_type="simple",
                 memory_ref=None,
                 model_ref=None):
        super().__init__(client, name)

        # --> Setup rl settings
        self.settings.rl_behavior_settings.gen_dql_settings()

        # --> Create custom tensorboard object
        # TODO: Fix tensorboard
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format("", int(time.time())))

        # --> Setup rewards
        self.reward_function = Reward_function()
        self.goal_tracker = 0

        # ---- Setup agent properties
        # --> Setup model
        self.model = Vector_DDQL_model("Actor",
                                       len(self.observation),
                                       len(self.action_lst),
                                       model_ref=model_ref)

        # --> Setup memory
        if memory_type == "simple":
            self.memory = Replay_memory(self.settings.rl_behavior_settings.memory_size, memory_ref)

        elif memory_type == "prioritized":
            self.memory = Prioritized_experience_replay_memory(self.settings.rl_behavior_settings.memory_size, memory_ref)

        else:
            print("!!!!! Invalid memory setting !!!!!")
            sys.exit()

        # ---- Setup trackers
        # --> Used to count when to update target network with main network's weights
        # TODO: Implemented blended hard/soft update
        self.target_update_counter = 0

        # --> Step trackers
        self.observation_history = [self.observation]
        self.action_history = []
        self.reward_history = []

        # --> Episode trackers
        self.observation_timeline = []
        self.action_timeline = []
        self.reward_timeline = []

        # self.aggr_ep_reward_timeline = {'ep': [], 'avg': [], 'max': [], 'min': []}

        return

    @property
    def observation(self):
        # --> Determine vector to next goal
        x = self.reward_function.goal_dict[str(self.goal_tracker)]["x"] - self.state.kinematics_estimated.position.x_val
        y = self.reward_function.goal_dict[str(self.goal_tracker)]["y"] - self.state.kinematics_estimated.position.y_val
        z = self.reward_function.goal_dict[str(self.goal_tracker)]["z"] - self.state.kinematics_estimated.position.z_val

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

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self):
        return self.model.main_network.predict(np.array(self.observation).reshape(-1, *self.observation.shape) / 255)[0]

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

        collision = self.check_final_state

        # --> Determine reward based on resulting state
        reward = self.reward_function.get_reward(self.observation, self.goal_tracker, collision, self.age)

        # --> Determine whether done or not
        done = self.reward_function.check_if_done(self.observation, self.goal_tracker, collision, self.age, self.settings.agent_settings.max_step)

        if not done:
            self.age += 1

        # --> Record step results
        self.observation_history.append(self.observation)
        self.action_history.append(action)
        self.reward_history.append(reward)

        return self.observation, reward, done

    def remember(self, current_state, action, reward, next_state, done):
        self.memory.remember(current_state, action, reward, next_state, done)
        return

    def train(self):
        # TODO: Connect settings to epoque
        # --> Check whether memory contains enough experience
        if self.memory.length < self.settings.rl_behavior_settings.min_replay_memory_size:
            return

        # --> Randomly sample minibatch from the memory
        minibatch, indices = self.memory.sample(self.settings.rl_behavior_settings.minibatch_size)
        # minibatch = random.sample(self.memory.memory, self.settings.rl_behavior_settings.minibatch_size)

        # --> Get current states from minibatch (rgb normalised)
        current_states = np.array([transition[0] for transition in minibatch])
        
        # --> Query main model for Q values
        current_qs_list = self.model.main_network.predict(current_states)
        
        # --> Get next states from minibatch
        next_states = np.array([transition[3] for transition in minibatch])
        
        # --> Query target model for Q values
        future_qs_list = self.model.target_network.predict(next_states)
        
        # --> Creating feature set and target list
        x = []      # Images
        y = []      # Resulting Q values
        
        # --> Enumerating the batches (tuple is content of minibatch, see remember)
        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                # --> If not done, get new q from future states
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.settings.rl_behavior_settings.discount * max_future_q
            else:
                # --> If done, set new q equal reward
                new_q = reward

            # --> Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # --> Append to training data
            x.append(current_state)
            y.append(current_qs)

        # --> Fit main model on all samples as one batch, log only on terminal state
        # TODO: Fix tensorboard
        # self.main_model.fit(np.array(x), np.array(y),
        #                     batch_size=self.settings.rl_behavior_settings.minibatch_size,
        #                     verbose=0,
        #                     shuffle=False,
        #                     callbacks=[self.tensorboard] if terminal_state else None)

        self.model.main_network.fit(np.array(x), np.array(y),
                                    batch_size=self.settings.rl_behavior_settings.minibatch_size,
                                    verbose=0,
                                    shuffle=False)

        if self.target_update_counter > self.settings.rl_behavior_settings.update_target_every:
            # --> Update target network with weights of main network
            self.model.target_network.set_weights(self.model.main_network.get_weights())

            # --> Reset target_update_counter
            self.target_update_counter = 0

        return

    def reset(self, random_starting_pos=False):
        # TODO: Implement random offset starting point
        # --> Reset Drone to starting position
        self.client.reset()

        # --> Restart simulation
        self.client.simPause(False)

        # --> Enable API control and take off
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(0, 0, -2, 3).join()

        # --> Reset agent properties
        self.age = 0

        # --> Record episode trackers to timeline trackers
        self.observation_timeline += self.observation_history
        self.action_timeline += self.action_history
        self.reward_timeline += self.reward_history

        # --> Reset step trackers
        self.observation_history = [self.observation]
        self.action_history = []
        self.reward_history = []

        # --> Update target network counter
        self.target_update_counter += 1
