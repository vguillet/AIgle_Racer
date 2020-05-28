
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import sys

# Libs
import numpy as np
from itertools import combinations, product
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

# Own modules
from AIgle_Project.src.Tools.Agent import Agent
from AIgle_Project.src.Navigation.Tools.RL_agent_abstract import RL_agent_abc

from AIgle_Project.src.Navigation.Models.Actor_DDQL import Actor_DDQL
from AIgle_Project.src.Navigation.Models.Critic_DDQL import Critic_DDQL

from AIgle_Project.src.Navigation.Tools.Replay_memory import Replay_memory
from AIgle_Project.src.Navigation.Tools.Prioritized_experience_replay_memory import Prioritized_experience_replay_memory

from AIgle_Project.src.Navigation.Tools.Reward_function_gen import Reward_function
from AIgle_Project.src.Navigation.Tools.OUAction_noise import OUAction_noise

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
        self.settings.rl_behavior_settings.gen_ddpg_settings()

        # --> Setup tools
        self.noise = OUAction_noise(mu=np.zeros(len(self.action_lst)))

        # --> Setup rewards
        self.reward_function = Reward_function()
        self.goal_tracker = 0

        # ---- Setup agent properties
        # --> Setup model
        self.actor_model = Actor_DDQL("Actor",
                                      len(self.observation),
                                      len(self.action_lst),
                                      model_ref=actor_ref)

        self.critic_model = Critic_DDQL("Critic",
                                        len(self.observation),
                                        len(self.action_lst),
                                        model_ref=critic_ref)

        # --> Setup memory
        if memory_type == "simple":
            self.memory = Replay_memory(self.settings.rl_behavior_settings.memory_size, memory_ref)

        elif memory_type == "prioritized":
            self.memory = Prioritized_experience_replay_memory(self.settings.rl_behavior_settings.memory_size, memory_ref)

        else:
            print("!!!!! Invalid memory setting !!!!!")
            sys.exit()

        # ---- Setup trackers
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

    def get_qs(self):
        # --> Queries actor main network for Q values given current observation
        action = self.actor_model.main_network.predict(self.observation).ravel()[0]
        noisy_action = action + self.noise()
        print(action)
        # TODO: Scale actions to match action space
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

        # --> Get current states, action and next states from minibatch
        batch_current_states = np.array([transition[0] for transition in minibatch])
        batch_actions = np.array([transition[1] for transition in minibatch])
        batch_next_states = np.array([transition[3] for transition in minibatch])

        # --> Get next actions using actor target network
        next_actions = self.actor_model.target_network.predict(batch_current_states)

        # --> Gen Q value using critic target network
        next_qs_list = self.critic_model.target_network.predict([batch_next_states, next_actions])

        # --> Creating feature set and target list
        y = []      # Resulting Q values

        # --> Enumerating the batches (tuple is content of minibatch, see remember)
        for i, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                # --> If not done, get new q from reward and Q_next
                y[i] = reward + next_qs_list[i] * self.settings.rl_behavior_settings.discount

        # --> Converting y to array
        y = np.array(y)

        # --> Updating priorities if using Prioritized experience replay
        if indices is not None:
            td_error = np.abs(y - self.critic_model.main_network.predict([batch_current_states, batch_actions]))
            self.memory.update_priorities(indices, td_error)

        # --> Training critics on batch
        self.critic_model.main_network.train_on_batch([batch_current_states, batch_actions], y)

        # --> Determining gradient difference
        with tf.GradientTape() as tape:
            a = self.actor_model.main_network(batch_current_states)
            tape.watch(a)
            q = self.critic_model.main_network([batch_current_states, a])

        # --> Getting the gradient of a and q
        dq_da_gradient = tape.gradient(q, a)

        with tf.GradientTape() as tape:
            a = self.actor_model.main_network(batch_current_states)
            theta = self.actor_model.main_network.trainable_variables

        # --> Getting the gradient of a and theta
        da_dtheta = tape.gradient(a, theta, output_gradients=-dq_da_gradient)

        actor_opt = Adam(self.settings.rl_behavior_settings.actor_learning_rate)
        actor_opt.apply_gradients(zip(da_dtheta, self.actor_model.main_network.trainable_variables))

        # --> Update models targets
        self.actor_model.soft_update_target(self.settings.rl_behavior_settings.tau)
        self.critic_model.soft_update_target(self.settings.rl_behavior_settings.tau)
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

        # Record episode trackers to timeline trackers
        self.observation_timeline += self.observation_history
        self.action_timeline += self.action_history
        self.reward_timeline += self.reward_history

        # Reset step trackers
        self.observation_history = [self.observation]
        self.action_history = []
        self.reward_history = []
