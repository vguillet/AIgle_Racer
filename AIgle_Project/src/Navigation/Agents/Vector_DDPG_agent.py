
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import sys
import random

# Libs
import numpy as np
from itertools import combinations, product
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

# Own modules
from AIgle_Project.src.Tools.Agent import Agent
from AIgle_Project.src.Navigation.Tools.RL_agent_abstract import RL_agent_abc

from AIgle_Project.src.Navigation.Models.Vector_Actor_DDPG_model import Vector_Actor_DDPG_model
from AIgle_Project.src.Navigation.Models.Vector_Critic_DDQL_model import Vector_Critic_DDQL_model

from AIgle_Project.src.Navigation.Tools.Replay_memory import Replay_memory
from AIgle_Project.src.Navigation.Tools.Prioritized_experience_replay_memory import Prioritized_experience_replay_memory

from AIgle_Project.src.Navigation.Tools.Door_reward_function_gen import Door_reward_function
from AIgle_Project.src.Navigation.Tools.Track_reward_function_gen import Track_reward_function

from AIgle_Project.src.Navigation.Tools.OUAction_noise import OUAction_noise

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class Vector_DDPG_agent(RL_agent_abc, Agent):
    def __init__(self, client, name):

        super().__init__(client, name)

        # --> Setup rl settings
        self.settings.rl_behavior_settings.gen_ddpg_settings()

        # --> Setup tools
        self.noise = OUAction_noise(mu=np.zeros(len(self.action_lst)))

        # --> Setup rewards
        if self.settings.rl_behavior_settings.training_type == "Door":
            self.reward_function = Door_reward_function()

        elif self.settings.rl_behavior_settings.training_type == "Track":
            self.reward_function = Track_reward_function()

        self.goal_tracker = 0

        # ---- Setup agent properties
        # --> Setup model
        checkpoint_path = "AIgle_Project/src/Navigation/Saved_models/Vector_ddpg" \
                          + '/' + self.settings.rl_behavior_settings.training_type \
                          + "/" + self.settings.rl_behavior_settings.run_name

        print("------- Initiating actor")
        self.actor_model = Vector_Actor_DDPG_model("Actor",
                                                   self.observation.shape,
                                                   len(self.action_lst),
                                                   model_ref=self.settings.rl_behavior_settings.actor_ref,
                                                   checkpoint_directory=checkpoint_path)

        print("------- Initiating critic")
        self.critic_model = Vector_Critic_DDQL_model("Critic",
                                                     self.observation.shape,
                                                     len(self.action_lst),
                                                     model_ref=self.settings.rl_behavior_settings.critic_ref,
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

        # --> Episode trackers
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

    def get_qs(self, add_noise=False):
        # --> Queries actor main network for Q values given current observation
        state = np.expand_dims(self.observation, axis=0).astype(np.float32)
        actions_qs = self.actor_model.main_network.predict(state)

        # --> Add noise to result
        actions_qs += self.noise() * add_noise

        return actions_qs

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
        reward = self.reward_function.get_reward(self.observation,
                                                 self.goal_tracker,
                                                 collision,
                                                 self.age,
                                                 self.settings.agent_settings.max_step)

        # --> Determine whether done or not
        done, self.goal_tracker, self.age = self.reward_function.check_if_done(self.observation,
                                                                               self.goal_tracker,
                                                                               collision,
                                                                               self.age,
                                                                               self.settings.agent_settings.max_step)

        # --> Record step results
        self.observation_history.append(self.observation)
        self.action_history.append(action)
        self.reward_history.append(reward)

        return self.observation, reward, done

    def remember(self, current_state, action, reward, next_state, done):
        self.memory.remember(current_state, action, reward, next_state, done)
        return

    def train(self, discount, tau):
        # --> Check whether memory contains enough experience
        if self.memory.length < self.settings.rl_behavior_settings.min_replay_memory_size:
            return

        # --> Randomly sample minibatch from the memory
        minibatch, indices = self.memory.sample(self.settings.rl_behavior_settings.minibatch_size)

        # --> Get current states, action (from minibatch) and Qs (using critic main network)
        batch_current_states = np.array([transition[0] for transition in minibatch])
        batch_current_actions = np.array([transition[1] for transition in minibatch])
        # batch_current_qs_list = self.critic_model.main_network.predict([batch_current_states, batch_current_actions])

        batch_rewards = np.array([transition[2] for transition in minibatch])
        batch_dones = np.array([transition[4] for transition in minibatch])

        # --> Get next states (from minibatch), action (using actor target network) and Qs (using critic target network)
        batch_next_states = np.array([transition[3] for transition in minibatch])
        batch_next_actions = self.actor_model.target_network.predict(batch_next_states)
        batch_next_qs = self.critic_model.target_network.predict([batch_next_states, batch_next_actions])

        # --> Get batch target qs
        batch_target_qs = batch_rewards + batch_next_qs * discount * (1. - batch_dones)

        # --> Train critic main network
        with tf.GradientTape() as tape:
            q_values = self.critic_model.main_network([batch_current_states, batch_current_actions])

            td_error = q_values - batch_target_qs
            critic_loss = tf.reduce_mean(indices * tf.math.square(td_error))

        critic_grad = tape.gradient(critic_loss, self.critic_model.main_network.trainable_variables)  # Compute critic gradient
        critic_optimiser = Adam(self.settings.rl_behavior_settings.critic_learning_rate)
        critic_optimiser.apply_gradients(zip(critic_grad, self.critic_model.main_network.trainable_variables))

        # --> Update priorities
        if isinstance(self.memory, Prioritized_experience_replay_memory):
            abs_errors = tf.reduce_sum(tf.abs(td_error), axis=1)
            self.memory.update_priorities(indices, abs_errors)

        # --> Train train actor main network
        with tf.GradientTape() as tape:
            actions = self.actor_model.main_network(batch_current_states)
            actor_loss = -tf.reduce_mean(self.critic_model.main_network([batch_current_states, actions]))

        actor_grad = tape.gradient(actor_loss, self.actor_model.main_network.trainable_variables)  # Compute actor gradient
        actor_optimiser = Adam(self.settings.rl_behavior_settings.actor_learning_rate)
        actor_optimiser.apply_gradients(zip(actor_grad, self.actor_model.main_network.trainable_variables))

        # --> Soft update models targets
        self.actor_model.soft_update_target(tau)
        self.critic_model.soft_update_target(tau)
        return

    def reset(self, random_starting_pos=False, random_flip_track=False):
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

            pose.position.x_val = random.randint(-10, 10)
            pose.position.y_val = random.randint(-6, 6)
            pose.position.z_val = random.randint(-4, 4)

            self.client.simSetVehiclePose(pose, True)

        if random_flip_track:
            if bool(random.getrandbits(1)):
                self.reward_function.flip_track()

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
        self.goal_tracker = 0