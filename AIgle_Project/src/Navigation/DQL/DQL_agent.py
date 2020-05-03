
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
from AIgle_Project.src.Vision.Camera import Camera
from AIgle_Project.src.Navigation.Tools.RL_agent_abstract import RL_agent_abc
from AIgle_Project.src.Navigation.Tools.Tensor_board_gen import ModifiedTensorBoard
from AIgle_Project.src.Navigation.Tools.Reward_function_gen import Reward_function

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class DQL_agent(RL_agent_abc, Agent):
    def __init__(self, client, name):
        super().__init__(client, name)

        # --> Setup rl settings
        self.settings.rl_behavior_settings.gen_dql_settings()

        # --> Create custom tensorboard object
        # TODO: Fix tensorboard
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format("", int(time.time())))

        # --> Setup camera
        self.camera = Camera(client, "0", 0)

        # --> Setup model
        self.assigned_main_model = None
        self.assigned_target_model = None

        # --> Setup reward dict
        self.reward_function = Reward_function()

        # --> Setup trackers
        self.memory = deque(maxlen=self.settings.rl_behavior_settings.memory_size)

        # Step trackers
        # self.pos_history = []

        # self.reward_history = []
        # self.action_history = []
        # self.action_success_history = []

        # Episode trackers
        # self.reward_timeline = []
        # self.aggr_ep_reward_timeline = {'ep': [], 'avg': [], 'max': [], 'min': []}

        return

    @property
    def main_model(self):
        if self.assigned_main_model is None:
            print("!!!!! No model assigned to agent !!!!!")
            sys.exit()
        else:
            return self.assigned_main_model

    @property
    def target_model(self):
        if self.assigned_target_model is None:
            print("!!!!! No model assigned to agent !!!!!")
            sys.exit()
        else:
            return self.assigned_target_model

    @property
    def rl_state(self):

        response = self.camera.fetch_single_img()

        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # original image is fliped vertically
        img_rgb = np.flipud(img_rgb)

        return img_rgb

    @property
    def hidden_rl_state(self):

        linear_velocity_magnitude = (abs(self.state.kinematics_estimated.linear_velocity.x_val)
                                     + abs(self.state.kinematics_estimated.linear_velocity.y_val)
                                     + abs(self.state.kinematics_estimated.linear_velocity.z_val))/3

        return ((self.state.kinematics_estimated.position.x_val,
                self.state.kinematics_estimated.position.y_val,
                self.state.kinematics_estimated.position.z_val),
                linear_velocity_magnitude)

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
        return self.main_model.predict(np.array(self.rl_state).reshape(-1, *self.rl_state.shape) / 255)[0]

    def step(self, action):
        # --> Determine action requested
        action = self.action_lst[action]

        # --> Determine target new state
        current_state = self.hidden_rl_state
        next_state = [[round(current_state[0][0] + action[0][0], 1),
                      round(current_state[0][1] + action[0][1], 1),
                      round(current_state[0][2] + action[0][2], 1)],
                      action[1]]

        if next_state[0][2] < -6:
            next_state[0][2] = -6

        # --> Move to target
        self.move(next_state)

        # --> Determine reward based on resulting state
        reward = self.reward_function.get_reward(self.hidden_rl_state, self.collision)

        # --> Determine whether done or not
        done = self.reward_function.check_if_done(self.hidden_rl_state, self.collision)

        return self.rl_state, reward, done

    def remember(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))
        return

    def train(self, terminal_state, target_update_counter):
        # --> Check whether memory contains enough experience
        if len(self.memory) < self.settings.rl_behavior_settings.min_replay_memory_size:
            return
        # --> Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, self.settings.rl_behavior_settings.minibatch_size)

        # --> Get current states from minibatch (rgb normalised)
        current_states = np.array([transition[0] for transition in minibatch])/255
        
        # --> Query main model for Q values
        current_qs_list = self.main_model.predict(current_states)
        
        # --> Get next states from minibatch
        next_states = np.array([transition[3] for transition in minibatch])/255
        
        # --> Query target model for Q values
        future_qs_list = self.target_model.predict(next_states)
        
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

        # Fit main model on all samples as one batch, log only on terminal state
        # TODO: Fix tensorboard
        # self.main_model.fit(np.array(x)/255, np.array(y),
        #                     batch_size=self.settings.rl_behavior_settings.minibatch_size,
        #                     verbose=0,
        #                     shuffle=False,
        #                     callbacks=[self.tensorboard] if terminal_state else None)
        self.main_model.fit(np.array(x)/255, np.array(y),
                            batch_size=self.settings.rl_behavior_settings.minibatch_size,
                            verbose=0,
                            shuffle=False)

        # --> Update target network counter every episode
        if terminal_state:
            target_update_counter += 1

        if target_update_counter > self.settings.rl_behavior_settings.update_target_every:
            # --> Update target network with weights of main network
            self.target_model.set_weights(self.main_model.get_weights())

            # --> Reset target_update_counter
            target_update_counter = 0

        return target_update_counter

    def __str__(self):
        return self.name + " (Bot)"

    def __repr__(self):
        self.__repr__()
