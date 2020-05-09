
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys
import time

# Libs
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Own modules
from AIgle_Project.Settings.SETTINGS import SETTINGS
from AIgle_Project.src.Navigation.Tools.ML_tools import ML_tools
from AIgle_Project.src.Navigation.Tools.RL_tools import RL_tools

from AIgle_Project.src.Navigation.Tools.Replay_memory import Replay_memory
from AIgle_Project.src.Navigation.Tools.Prioritized_experience_replay_memory import Prioritized_experience_replay_memory

from AIgle_Project.src.Navigation.Vector_DDPG.Actor_model import Actor_model
from AIgle_Project.src.Navigation.Vector_DDPG.Critic_model import Critic_model

# from AIgle_Project.src.Navigation.Vector_DDPG.DDPG_agent import DDPG_agent
# from AIgle_Project.src.Navigation.Models.DQL_models import DQL_models

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DQL_vector_based_navigation:
    def __init__(self, client):


        class DDPG():
            def __init__(self, env,
                         actor=None,
                         critic=None,
                         buffer=None,
                         action_bound_range=1,
                         max_buffer_size=100000,
                         batch_size=64,
                         replay='uniform',
                         max_time_steps=1000,
                         tow=0.001,
                         discount_factor=0.99,
                         explore_time=1000,
                         actor_learning_rate=0.0001,
                         critic_learning_rate=0.001,
                         dtype='float32',
                         n_episodes=1000,
                         verbose=True,
                         plot=False,
                         model_save_freq=10):
                '''env , # Gym environment with continous action space
                         actor(None), # Tensorflow/keras model
                         critic (None), # Tensorflow/keras model
                         buffer (None), # pre-recorded buffer
                         action_bound_range=1,
                         max_buffer_size =10000, # maximum transitions to be stored in buffer
                         batch_size =64, # batch size for training actor and critic networks
                         max_time_steps = 1000 ,# no of time steps per epoch
                         tow = 0.001, # for soft target update
                         discount_factor  = 0.99,
                         explore_time = 1000, # time steps for random actions for exploration
                         actor_learning_rate = 0.0001,
                         critic_learning_rate = 0.001
                         dtype = 'float32',
                         n_episodes = 1000 ,# no of episodes to run
                         reward_plot = True ,# (bool)  to plot reward progress per episode
                         model_save = 1) # epochs to save models and buffer'''
                #############################################
                # --------------- Parametres-----------------#
                #############################################
                self.model_save_freq = model_save_freq
                self.max_buffer_size = max_buffer_size
                self.batch_size = batch_size
                self.T = max_time_steps  ## Time limit for a episode
                self.tow = tow  ## Soft Target Update
                self.gamma = discount_factor  ## discount factor
                # self.target_update_freq = 10  ## frequency for updating target weights
                self.explore_time = explore_time
                self.act_learning_rate = actor_learning_rate
                self.critic_learning_rate = critic_learning_rate
                self.dflt_dtype = dtype
                self.n_episodes = n_episodes
                self.action_bound_range = action_bound_range
                self.plot = plot
                self.verbose = verbose
                self.actor_opt = Adam(self.act_learning_rate)
                self.critic_opt = Adam(self.critic_learning_rate)
                self.r, self.l, self.qlss = [], [], []
                self.env = env
                self.observ_min = self.env.observation_space.low
                self.observ_max = self.env.observation_space.high
                action_dim = 1
                state_dim = len(env.reset())


                if buffer is not None:
                    print('using loaded models')
                    self.buffer = buffer
                    self.actor = actor
                    self.critic = critic
                else:
                    if replay == 'prioritized':
                        self.buffer = Prioritized_experience_replay_memory(max_buffer_size, batch_size, dtype)
                    else:
                        self.buffer = Replay_Buffer(max_buffer_size, batch_size, dtype)
                    self.actor = _actor_network(state_dim, action_dim, action_bound_range).model()
                    self.critic = _critic_network(state_dim, action_dim).model()
                self.actor_target = _actor_network(state_dim, action_dim, action_bound_range).model()
                self.actor_target.set_weights(self.actor.get_weights())
                self.critic_target = _critic_network(state_dim, action_dim).model()
                self.critic.compile(loss='mse', optimizer=self.critic_opt)
                self.critic_target.set_weights(self.critic.get_weights())

            #############################################
            # ----Action based on exploration policy-----#
            #############################################
            def take_action(self, state, rand):
                actn = self.actor.predict(state).ravel()[0]
                if rand:
                    return actn + random.uniform(-self.action_bound_range, self.action_bound_range)
                else:
                    return actn

            #############################################
            # --------------Update Networks--------------#
            #############################################
            def train_networks(self, states_batch, actions_batch, rewards_batch, next_states_batch, done_batch,
                               indices=None):
                next_actions = self.actor_target(next_states_batch)

                q_t_pls_1 = self.critic_target([next_states_batch, next_actions])
                y_i = rewards_batch
                for i in range(self.batch_size):
                    if not done_batch[i]:
                        y_i[i] += q_t_pls_1[i] * self.gamma
                if isinstance(self.buffer, Prioritized_experience_replay_memory):
                    td_error = np.abs(y_i - self.critic.predict([states_batch, actions_batch]))
                    self.buffer.update_priorities(indices, td_error)
                self.critic.train_on_batch([states_batch, actions_batch], y_i)

                # '''with tf.GradientTape() as tape:
                #     #states_batch = tf.convert_to_tensor(states_batch, dtype=self.dflt_dtype)
                #     a = self.actor(states_batch)
                #     # states_batch = tf.cast(states_batch,self.dflt_dtype)
                #     lss = -tf.reduce_mean(self.critic([states_batch, a]))
                # da_dtheta = tape.gradient(lss, self.actor.trainable_variables)
                # self.actor_opt.apply_gradients(zip(da_dtheta, self.actor.trainable_variables))'''

                with tf.GradientTape() as tape:
                    a = self.actor(states_batch)
                    tape.watch(a)
                    q = self.critic([states_batch, a])
                dq_da = tape.gradient(q, a)

                with tf.GradientTape() as tape:
                    a = self.actor(states_batch)
                    theta = self.actor.trainable_variables
                da_dtheta = tape.gradient(a, theta, output_gradients=-dq_da)
                self.actor_opt.apply_gradients(zip(da_dtheta, self.actor.trainable_variables))

            def update_target(self, target, online, tow):
                init_weights = online.get_weights()
                update_weights = target.get_weights()
                weights = []
                for i in tf.range(len(init_weights)):
                    weights.append(tow * init_weights[i] + (1 - tow) * update_weights[i])
                target.set_weights(weights)
                return target

            def train(self, ):
                obs = self.env.reset()
                state_dim = len(obs)
                experience_cnt = 0
                self.ac = []
                rand = True
                for episode in range(self.n_episodes):
                    ri, li, qlssi = [], [], []
                    state_t = np.array(self.env.reset(), dtype=self.dflt_dtype).reshape(1, state_dim)
                    state_t = (state_t - self.observ_min) / (self.observ_max - self.observ_min)
                    for t in range(self.T):
                        action_t = self.take_action(state_t, rand)
                        # action_t = action_t
                        self.ac.append(action_t)
                        temp = self.env.step([action_t])  # step returns obs_t+1, reward, done
                        state_t_pls_1, rwrd_t, done_t = temp[0], temp[1], temp[2]
                        state_t_pls_1 = (state_t_pls_1 - self.observ_min) / (self.observ_max - self.observ_min)
                        ri.append(rwrd_t)
                        self.buffer.add_experience(
                            state_t.ravel(), action_t, rwrd_t, np.array(state_t_pls_1, self.dflt_dtype), done_t)

                        state_t = np.array(state_t_pls_1, dtype=self.dflt_dtype).reshape(1, state_dim)
                        if not rand:
                            if isinstance(self.buffer, Prioritized_experience_replay):
                                states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, indices = self.buffer.sample_batch()
                                self.train_networks(states_batch, actions_batch, rewards_batch, next_states_batch,
                                                    done_batch, indices)
                            else:
                                states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = self.buffer.sample_batch()
                                self.train_networks(states_batch, actions_batch, rewards_batch, next_states_batch,
                                                    done_batch, None)

                            self.actor_target = self.update_target(self.actor_target, self.actor, self.tow)
                            self.critic_target = self.update_target(self.critic_target, self.critic, self.tow)
                        if done_t or t == self.T - 1:
                            rr = np.sum(ri)
                            self.r.append(rr)
                            if self.verbose: print('Episode %d : Total Reward = %f' % (episode, rr))
                            if self.plot:
                                plt.plot(self.r)
                                plt.pause(0.0001)
                            break
                        if rand: experience_cnt += 1
                        if experience_cnt > self.explore_time: rand = False

                    if self.model_save_freq:
                        if episode % self.model_save_freq == 0:
                            self.actor.save('actor_model.h5')
                            self.critic.save('critic_model.h5')
                            self.actor_target.save('actor_model.h5')
                            self.critic_target.save('critic_model.h5')
                            with open('buffer', 'wb') as file:
                                pickle.dump({'buffer': self.buffer}, file)