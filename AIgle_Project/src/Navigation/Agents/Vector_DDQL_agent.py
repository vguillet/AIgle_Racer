
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
    def __init__(self, client, name):

        super().__init__(client, name)

        # --> Setup rl settings
        self.settings.rl_behavior_settings.gen_ddql_settings()

        # --> Create custom tensorboard object
        # TODO: Fix tensorboard
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format("", int(time.time())))

        # --> Setup rewards
        self.reward_function = Reward_function()
        self.goal_tracker = 0

        # ---- Setup agent properties
        # --> Setup model
        checkpoint_path = "AIgle_Project/src/Navigation/Saved_models/Vector_ddql" \
                          + '/' + self.settings.rl_behavior_settings.training_type \
                          + "/" + self.settings.rl_behavior_settings.run_name

        self.model = Vector_DDQL_model("Vector",
                                       self.observation.shape,
                                       len(self.action_lst),
                                       model_ref=self.settings.rl_behavior_settings.model_ref,
                                       checkpoint_directory=checkpoint_path)

        # --> Setup memory
        if self.settings.rl_behavior_settings.memory_type == "simple":
            self.memory = Replay_memory(self.settings.rl_behavior_settings.memory_size,
                                        self.settings.rl_behavior_settings.memory_ref)

        elif self.settings.rl_behavior_settings.memory_type == "prioritized":
            self.memory = Prioritized_experience_replay_memory(self.settings.rl_behavior_settings.memory_size,
                                                               self.settings.rl_behavior_settings.memory_ref)

        else:
            print("!!!!! Invalid memory setting !!!!!")
            sys.exit()

        # ---- Setup trackers
        # --> Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # --> Step trackers
        self.observation_history = [self.observation]
        self.action_history = []
        self.reward_history = []

        # --> Epoque trackers
        self.observation_timeline = []
        self.action_timeline = []
        self.reward_timeline = []

        # self.aggr_ep_reward_timeline = {'ep': [], 'avg': [], 'max': [], 'min': []}

        return

    @property
    def observation(self):
        # --> Determine vector to next goal
        x = round(self.reward_function.goal_dict[str(self.goal_tracker)]["x"] - self.state.kinematics_estimated.position.x_val, 1)
        y = round(self.reward_function.goal_dict[str(self.goal_tracker)]["y"] - self.state.kinematics_estimated.position.y_val, 1)
        z = round(self.reward_function.goal_dict[str(self.goal_tracker)]["z"] - self.state.kinematics_estimated.position.z_val, 1)

        # --> Determine velocity vector
        u = round(self.state.kinematics_estimated.linear_velocity.x_val, 1)
        v = round(self.state.kinematics_estimated.linear_velocity.y_val, 1)
        w = round(self.state.kinematics_estimated.linear_velocity.z_val, 1)

        # return np.array([x, y, z, u, v, w])
        return np.array([x, y, z])

    @property
    def action_lst(self):
        possible_moves = []
        possible_speeds = []

        # --> List all possible positions combinations
        for dimension in range(3):
            for i in range(self.settings.agent_settings.agent_min_move,
                           self.settings.agent_settings.agent_max_move + 1):
                possible_moves.append(i)
                possible_moves.append(-i)

        possible_moves = set(combinations(possible_moves, 3))

        # --> Convert to lst of lst
        possible_moves_lst = []
        for moves in possible_moves:
            possible_moves_lst.append(list(moves))

        # --> List all possible speeds
        for speed in range(self.settings.agent_settings.agent_min_speed,
                           self.settings.agent_settings.agent_max_speed + 1):
            possible_speeds.append(speed)

        # --> List all possible positions and speed combinations
        actions = list(product(possible_moves_lst, possible_speeds))

        # TODO: Clean up
        # --> Convert to lst of lst
        action_lst = []
        for action in actions:
            action_lst.append(list(action))

        flat_action_lst = []
        # --> Flatten list
        for action in action_lst:
            item_lst = []
            for item in action:
                if type(item) is list:
                    for subitem in item:
                        item_lst.append(subitem)
                else:
                    item_lst.append(item)
            flat_action_lst.append(item_lst)

        return flat_action_lst

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self):
        return self.model.main_network.predict(np.array(self.observation).reshape(-1, *self.observation.shape))[0]

    def step(self, action):
        # --> Increase agent's age
        self.age += 1

        # --> Determine action requested
        action = self.action_lst[action]

        # --> Determine target new state
        current_state = self.state

        waypoint = [round(current_state.kinematics_estimated.position.x_val + action[0], 1),
                    round(current_state.kinematics_estimated.position.y_val + action[1], 1),
                    round(current_state.kinematics_estimated.position.z_val + action[2], 1),
                    action[3]]

        # print([round(current_state.kinematics_estimated.position.x_val, 1),
        #        round(current_state.kinematics_estimated.position.y_val, 1),
        #        round(current_state.kinematics_estimated.position.z_val, 1)])

        # --> Move to target
        self.move(waypoint)

        collision = self.check_final_state

        # --> Limiting top and low
        # # TODO: Improve limits
        if waypoint[2] < -6:
            collision = True
        elif waypoint[2] >= 3:
            collision = True

        # --> Determine reward based on resulting state
        reward = self.reward_function.get_reward(self.observation, self.goal_tracker, collision, self.age, self.settings.agent_settings.max_step)

        # --> Determine whether done or not
        done = self.reward_function.check_if_done(self.observation, self.goal_tracker, collision, self.age, self.settings.agent_settings.max_step)

        # --> Record step results
        self.observation_history.append(self.observation)
        self.action_history.append(action)
        self.reward_history.append(reward)

        return self.observation, reward, done

    def remember(self, current_state, action, reward, next_state, done):
        # print(current_state, action, reward, next_state, done)
        self.memory.remember(current_state, action, reward, next_state, done)
        return

    def train(self, discount, tau):
        # --> Check whether memory contains enough experience
        if self.memory.length < self.settings.rl_behavior_settings.min_replay_memory_size:
            return

        # --> Randomly sample minibatch from the memory
        minibatch, indices = self.memory.sample(self.settings.rl_behavior_settings.minibatch_size)

        # --> Get current states, action and next states from minibatch
        batch_current_states = np.array([transition[0] for transition in minibatch])
        batch_next_states = np.array([transition[3] for transition in minibatch])

        # --> Query main model for current states Q values
        batch_current_qs_list = self.model.main_network.predict(batch_current_states)

        # --> Query target model for next states Q values
        batch_next_qs_list = self.model.target_network.predict(batch_next_states)

        # --> Creating new qs list
        new_qs_lst = []

        # --> Creating feature set and target list
        x = []      # States
        y = []      # Resulting Q values

        # --> Enumerating the batches (tuple is content of minibatch)
        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                # --> If not done, get new q from future states
                max_future_q = np.max(batch_next_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                # --> If done, set new q equal reward
                new_q = reward

            # --> Update Q value for given state
            current_qs = batch_current_qs_list[index]
            current_qs[action] = new_q

            # --> Append to new qs lst
            new_qs_lst.append(new_q)

            # --> Append to training data
            x.append(current_state)
            y.append(current_qs)

        # --> Turns lists to arrays
        x = np.array(x)
        y = np.array(y)

        # --> Updating priorities if using Prioritized experience replay
        if isinstance(self.memory, Prioritized_experience_replay_memory):
            td_error = np.abs(np.transpose(np.array([new_qs_lst])) -
                              np.transpose(batch_current_qs_list.max(axis=1)[np.newaxis]))

            self.memory.update_priorities(indices, td_error)

        # --> Fit main model on all samples as one batch, log only on terminal state
        # TODO: Fix tensorboard
        # self.main_model.fit(np.array(x), np.array(y),
        #                     batch_size=self.settings.rl_behavior_settings.minibatch_size,
        #                     verbose=0,
        #                     shuffle=False,
        #                     callbacks=[self.tensorboard] if terminal_state else None)

        self.model.main_network.fit(x, y,
                                    batch_size=self.settings.rl_behavior_settings.minibatch_size,
                                    verbose=0,
                                    shuffle=False)

        self.model.soft_update_target(tau)

        if self.settings.rl_behavior_settings.hard_update_target_every is not None:
            if self.target_update_counter > self.settings.rl_behavior_settings.hard_update_target_every:
                # --> Update target network with weights of main network
                self.model.hard_update_target()

                # --> Reset target_update_counter
                self.target_update_counter = 0

        return

    def reset(self, random_starting_pos=False):
        # TODO: Implement random offset starting point
        # --> Reset Drone to starting position
        self.client.reset()

        # --> Restart simulation
        # self.client.simPause(False)

        # --> Enable API control and take off
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.moveToPositionAsync(0, 0, -2, 3).join()

        if random_starting_pos is True:
            pose = self.client.simGetVehiclePose()

            pose.position.x_val = random.randint(-20, 20)
            pose.position.y_val = random.randint(0, 6)
            pose.position.z_val = random.randint(-4, 4)

            self.client.simSetVehiclePose(pose, True)

        # --> Reset agent properties
        self.age = 0

        # --> Record epoque trackers to timeline trackers
        self.observation_timeline += self.observation_history
        self.action_timeline += self.action_history
        self.reward_timeline += self.reward_history

        # --> Reset step trackers
        self.observation_history = [self.observation]
        self.action_history = []
        self.reward_history = []

        # --> Update target network counter
        self.target_update_counter += 1


